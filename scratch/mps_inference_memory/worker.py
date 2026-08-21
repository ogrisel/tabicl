"""Isolated TabICLRegressor fit/predict with RSS + MPS peak sampling.

Reads one JSON config from stdin, prints one JSON result to stdout.
Aborts (exit 99) if process RSS grows past ``max_rss_delta_gb``.
"""

from __future__ import annotations

import atexit
import gc
import json
import os
import resource
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _maxrss_mb() -> float:
    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _mps_mb() -> tuple[float, float]:
    if not torch.backends.mps.is_available():
        return 0.0, 0.0
    alloc = torch.mps.current_allocated_memory() / (1024 * 1024)
    driver = torch.mps.driver_allocated_memory() / (1024 * 1024)
    return alloc, driver


def _make_split(n_train: int, n_test: int, n_features: int, seed: int):
    n_features = max(5, int(n_features))
    X, y = make_friedman1(
        n_samples=n_train + n_test,
        n_features=n_features,
        noise=0.1,
        random_state=seed,
    )
    return train_test_split(X, y, train_size=n_train, random_state=seed)


def _apply_unified_mem_cap() -> None:
    from tabicl._model.inference import InferenceManager

    orig = InferenceManager.get_available_gpu_memory

    def capped(self):
        gpu_mb = orig(self)
        cpu_mb = self.get_available_cpu_memory()
        return min(gpu_mb, cpu_mb)

    InferenceManager.get_available_gpu_memory = capped  # type: ignore[method-assign]


def _apply_query_chunks_mps() -> None:
    """Let the CPU query-chunk gate also fire on MPS (tensors stay on device)."""
    from tabicl._model.learning import ICLearning

    orig = ICLearning._icl_predictions

    def wrapped(self, R, y_train):
        mgr = self.inference_mgr
        if mgr._is_configured:
            batch_size = 1
            for dim in R.shape[:-2]:
                batch_size *= int(dim)
            bytes_per_query_row = (
                mgr.cpu_activation_factor * batch_size * R.shape[-1] * R.element_size()
            )
            chunk = max(
                1,
                int(mgr.cpu_memory_budget_mb * 1024 * 1024 // max(bytes_per_query_row, 1)),
            )
            PATCH_DEBUG["icl_seq"] = int(R.shape[-2])
            PATCH_DEBUG["icl_chunk"] = chunk
            PATCH_DEBUG["icl_dtype"] = str(R.dtype)
            PATCH_DEBUG["icl_width"] = int(R.shape[-1])
            PATCH_DEBUG["icl_budget_mb"] = mgr.cpu_memory_budget_mb
            PATCH_DEBUG["icl_would_chunk"] = chunk < int(R.shape[-2])
        if not (mgr._is_configured and mgr.exe_device.type == "mps"):
            return orig(self, R, y_train)
        real = mgr.exe_device

        class _CpuTyped:
            type = "cpu"

        mgr.exe_device = _CpuTyped()  # type: ignore[assignment]
        try:
            return orig(self, R, y_train)
        finally:
            mgr.exe_device = real

    ICLearning._icl_predictions = wrapped  # type: ignore[method-assign]


def _apply_disk_release_each_write() -> None:
    from tabicl._model.inference import DiskTensor

    orig = DiskTensor.__setitem__

    def setitem(self, indices, value):
        orig(self, indices, value)
        self.release_pages()

    DiskTensor.__setitem__ = setitem  # type: ignore[method-assign]


def _apply_activation_budget() -> None:
    """Cap accelerator batch sizing with cpu_memory_budget_mb (PR #1 idea on MPS)."""
    from tabicl._model.inference import InferenceManager, MemoryEstimator

    def budgeted(
        self,
        seq_len: int,
        include_inputs: bool = True,
        in_dim=None,
        max_bs: int = 50000,
    ):
        available_mem = self.get_available_gpu_memory()
        budget = float(getattr(self, "cpu_memory_budget_mb", 512.0))
        target_mem = min(available_mem * self.safety_factor, budget)
        estimated_bs = MemoryEstimator.estimate_batch_size(
            seq_len, target_mem, self.enc_name, include_inputs, in_dim
        )
        safe_bs = min(max(self.min_batch_size, estimated_bs), max_bs)
        PATCH_DEBUG["batch_target_mb"] = target_mem
        PATCH_DEBUG[f"est_bs_{self.enc_name}"] = safe_bs
        PATCH_DEBUG[f"seq_{self.enc_name}"] = seq_len
        return available_mem, safe_bs

    InferenceManager.estimate_safe_batch_size = budgeted  # type: ignore[method-assign]


def _apply_no_pin_memory() -> None:
    from tabicl._model.inference import InferenceManager

    orig = InferenceManager._allocate_output_buffer

    def no_pin(self, mode, shape, dtype):
        saved = self.max_pinned_memory_mb
        if self.exe_device.type == "mps":
            self.max_pinned_memory_mb = 0.0
        try:
            return orig(self, mode, shape, dtype)
        finally:
            self.max_pinned_memory_mb = saved

    InferenceManager._allocate_output_buffer = no_pin  # type: ignore[method-assign]


PATCH_DEBUG: dict[str, Any] = {}

PATCHES = {
    "unified_mem_cap": _apply_unified_mem_cap,
    "query_chunks_mps": _apply_query_chunks_mps,
    "disk_release_each_write": _apply_disk_release_each_write,
    "no_pin_memory": _apply_no_pin_memory,
    "activation_budget": _apply_activation_budget,
}


def apply_patches(names: list[str]) -> None:
    unknown = [n for n in names if n not in PATCHES]
    if unknown:
        raise ValueError(f"unknown patches: {unknown}")
    for name in names:
        PATCHES[name]()


def _abort(code: int, payload: dict[str, Any]) -> None:
    print(json.dumps({"ok": False, **payload}, default=str), flush=True)
    os._exit(code)


class MemoryWatch:
    """Sample RSS / MPS / host RAM / disk and abort before the machine saturates."""

    def __init__(
        self,
        rss0_mb: float,
        *,
        max_delta_gb: float,
        min_host_avail_gb: float,
        min_disk_free_gb: float,
        max_mps_driver_gb: float,
        interval_s: float = 0.05,
    ):
        self.rss0_mb = rss0_mb
        self.max_delta_mb = max_delta_gb * 1024
        self.min_host_avail_mb = min_host_avail_gb * 1024
        self.min_disk_free_mb = min_disk_free_gb * 1024
        self.max_mps_driver_mb = max_mps_driver_gb * 1024
        self.interval_s = interval_s
        self.peak_rss_mb = rss0_mb
        self.peak_mps_alloc_mb = 0.0
        self.peak_mps_driver_mb = 0.0
        self.min_host_avail_seen_mb = psutil.virtual_memory().available / (1024 * 1024)
        self.min_disk_free_seen_mb = psutil.disk_usage("/").free / (1024 * 1024)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sample()
        return {
            "peak_rss_mb": self.peak_rss_mb,
            "peak_rss_delta_mb": self.peak_rss_mb - self.rss0_mb,
            "peak_mps_alloc_mb": self.peak_mps_alloc_mb,
            "peak_mps_driver_mb": self.peak_mps_driver_mb,
            "min_host_avail_mb": self.min_host_avail_seen_mb,
            "min_disk_free_mb": self.min_disk_free_seen_mb,
            "maxrss_mb": _maxrss_mb(),
        }

    def _sample(self) -> None:
        rss = _rss_mb()
        self.peak_rss_mb = max(self.peak_rss_mb, rss)
        alloc, driver = _mps_mb()
        self.peak_mps_alloc_mb = max(self.peak_mps_alloc_mb, alloc)
        self.peak_mps_driver_mb = max(self.peak_mps_driver_mb, driver)
        host_avail = psutil.virtual_memory().available / (1024 * 1024)
        disk_free = psutil.disk_usage("/").free / (1024 * 1024)
        self.min_host_avail_seen_mb = min(self.min_host_avail_seen_mb, host_avail)
        self.min_disk_free_seen_mb = min(self.min_disk_free_seen_mb, disk_free)

        if rss - self.rss0_mb > self.max_delta_mb:
            _abort(99, {"error": "rss_cap", "rss_mb": rss, "rss0_mb": self.rss0_mb,
                        "max_delta_mb": self.max_delta_mb})
        if host_avail < self.min_host_avail_mb:
            _abort(98, {"error": "host_ram_cap", "host_avail_mb": host_avail,
                        "min_host_avail_mb": self.min_host_avail_mb})
        if disk_free < self.min_disk_free_mb:
            _abort(97, {"error": "disk_cap", "disk_free_mb": disk_free,
                        "min_disk_free_mb": self.min_disk_free_mb})
        if driver > self.max_mps_driver_mb:
            _abort(96, {"error": "mps_driver_cap", "mps_driver_mb": driver,
                        "max_mps_driver_mb": self.max_mps_driver_mb})

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()


def _install_src_root(src_root: str | None) -> str | None:
    """Prefer a given checkout's ``src/`` over the env's editable install."""
    if not src_root:
        return None
    root = Path(src_root)
    src = root / "src" if (root / "src" / "tabicl").is_dir() else root
    sys.path.insert(0, str(src))
    for key in list(sys.modules):
        if key == "tabicl" or key.startswith("tabicl."):
            del sys.modules[key]
    return str(src)


def run_once(cfg: dict[str, Any]) -> dict[str, Any]:
    src_used = _install_src_root(cfg.get("src_root"))
    from tabicl import TabICLRegressor
    import tabicl as tabicl_mod

    apply_patches(list(cfg.get("patches") or []))

    n_train = int(cfg["n_train"])
    n_test = int(cfg.get("n_test", 64))
    n_features = int(cfg.get("n_features", 10))
    device = str(cfg.get("device", "mps"))
    n_estimators = int(cfg.get("n_estimators", 1))
    use_amp = cfg.get("use_amp", "auto")
    offload_mode = cfg.get("offload_mode", "auto")
    disk_offload_dir = cfg.get("disk_offload_dir")
    kv_cache = cfg.get("kv_cache", False)
    seed = int(cfg.get("seed", 0))
    budget_mb = cfg.get("cpu_memory_budget_mb")
    inference_config = None
    if budget_mb is not None:
        budget = {"cpu_memory_budget_mb": float(budget_mb)}
        inference_config = {
            "COL_CONFIG": dict(budget),
            "ROW_CONFIG": dict(budget),
            "ICL_CONFIG": dict(budget),
        }

    tmp_disk = None
    if offload_mode == "disk" and not disk_offload_dir:
        tmp_disk = tempfile.mkdtemp(prefix="tabicl-mps-mem-")
        disk_offload_dir = tmp_disk

        def _cleanup():
            shutil.rmtree(tmp_disk, ignore_errors=True)

        atexit.register(_cleanup)

    rss0 = _rss_mb()
    watch = MemoryWatch(
        rss0,
        max_delta_gb=float(cfg.get("max_rss_delta_gb", 4.0)),
        min_host_avail_gb=float(cfg.get("min_host_avail_gb", 4.0)),
        min_disk_free_gb=float(cfg.get("min_disk_free_gb", 15.0)),
        max_mps_driver_gb=float(cfg.get("max_mps_driver_gb", 8.0)),
    )
    watch.start()

    X_train, X_test, y_train, _ = _make_split(n_train, n_test, n_features, seed)
    model = TabICLRegressor(
        n_estimators=n_estimators,
        norm_methods="none",
        feat_shuffle_method="none",
        batch_size=min(8, n_estimators),
        kv_cache=kv_cache,
        device=device,
        use_amp=use_amp,
        offload_mode=offload_mode,
        disk_offload_dir=disk_offload_dir,
        n_jobs=1,
        verbose=False,
        allow_auto_download=True,
        inference_config=inference_config,
    )

    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    if device == "mps":
        torch.mps.synchronize()
    fit_s = time.perf_counter() - t0
    rss_after_fit = _rss_mb()

    if device == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    pred = model.predict(X_test)
    if device == "mps":
        torch.mps.synchronize()
    predict_s = time.perf_counter() - t1
    rss_after_predict = _rss_mb()

    peaks = watch.stop()
    pred = np.asarray(pred)

    results_dir = Path(cfg.get("results_dir", "."))
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.get("tag", f"{device}_n{n_train}_p{n_features}")
    pred_path = results_dir / f"pred_{tag}.npy"
    np.save(pred_path, pred)

    del model, X_train, X_test, y_train
    gc.collect()
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if tmp_disk is not None:
        shutil.rmtree(tmp_disk, ignore_errors=True)

    return {
        "ok": True,
        "tag": tag,
        "device": device,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "n_estimators": n_estimators,
        "use_amp": use_amp,
        "offload_mode": offload_mode,
        "cpu_memory_budget_mb": budget_mb,
        "patches": list(cfg.get("patches") or []),
        "fit_s": fit_s,
        "predict_s": predict_s,
        "rss0_mb": rss0,
        "rss_after_fit_mb": rss_after_fit,
        "rss_after_predict_mb": rss_after_predict,
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "pred_path": str(pred_path),
        "tabicl_file": getattr(tabicl_mod, "__file__", None),
        "src_root": src_used,
        **peaks,
        **{k: v for k, v in PATCH_DEBUG.items()},
    }


def main() -> int:
    cfg = json.load(sys.stdin)
    try:
        result = run_once(cfg)
    except SystemExit:
        raise
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, default=str), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
