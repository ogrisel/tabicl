"""Helpers for measuring TabICL inference RSS + MPS allocations on unified memory.

Child processes abort if host RAM, disk, or process RSS would blow the caps
in this module. Results land next to this file under ``results/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
WORKER = HERE / "worker.py"

PYTHON = os.environ.get("TABICL_PYTHON", sys.executable)

# Resource caps: abort rather than saturate the machine.
MIN_HOST_RAM_GB = 6.0
ABORT_HOST_RAM_GB = 4.0
MIN_DISK_GB = 20.0
ABORT_DISK_GB = 15.0
MAX_RSS_DELTA_GB = 4.0
MAX_MPS_DRIVER_GB = 8.0
MAX_MMAP_GB = 2.0
WORKER_TIMEOUT_S = 600


def host_ram_gb() -> dict[str, float]:
    vm = psutil.virtual_memory()
    return {
        "total_gb": vm.total / 1024**3,
        "available_gb": vm.available / 1024**3,
        "used_gb": vm.used / 1024**3,
    }


def host_disk_gb(path: Path | str | None = None) -> dict[str, float]:
    usage = psutil.disk_usage(str(path or "/"))
    return {
        "total_gb": usage.total / 1024**3,
        "free_gb": usage.free / 1024**3,
        "used_gb": usage.used / 1024**3,
    }


def preflight_or_skip(
    *,
    min_ram_gb: float = MIN_HOST_RAM_GB,
    min_disk_gb: float = MIN_DISK_GB,
) -> dict[str, Any]:
    """Return host gauges, or raise SystemExit if the machine is too tight."""
    ram = host_ram_gb()
    disk = host_disk_gb("/")
    info = {"ram": ram, "disk": disk}
    if ram["available_gb"] < min_ram_gb:
        raise SystemExit(
            f"skip: only {ram['available_gb']:.1f} GB RAM free (need {min_ram_gb})"
        )
    if disk["free_gb"] < min_disk_gb:
        raise SystemExit(
            f"skip: only {disk['free_gb']:.1f} GB disk free (need {min_disk_gb})"
        )
    return info


def make_friedman_split(
    n_train: int,
    n_test: int,
    n_features: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_features = max(5, int(n_features))
    X, y = make_friedman1(
        n_samples=n_train + n_test,
        n_features=n_features,
        noise=0.1,
        random_state=seed,
    )
    return train_test_split(X, y, train_size=n_train, random_state=seed)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(proc.pid)
    except (psutil.NoSuchProcess, psutil.Error):
        proc.kill()
        return
    for child in parent.children(recursive=True):
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        parent.kill()
    except psutil.Error:
        pass


def run_isolated(cfg: dict[str, Any], *, timeout_s: int = WORKER_TIMEOUT_S) -> dict[str, Any]:
    """Run one worker in a fresh process; kill it if host RAM/disk get too low."""
    host = preflight_or_skip()
    abort_ram_gb = float(cfg.get("min_host_avail_gb", ABORT_HOST_RAM_GB))
    abort_disk_gb = float(cfg.get("min_disk_free_gb", ABORT_DISK_GB))
    cfg = {
        **cfg,
        "max_rss_delta_gb": cfg.get("max_rss_delta_gb", MAX_RSS_DELTA_GB),
        "max_mps_driver_gb": cfg.get("max_mps_driver_gb", MAX_MPS_DRIVER_GB),
        "min_host_avail_gb": abort_ram_gb,
        "min_disk_free_gb": abort_disk_gb,
        "max_mmap_gb": cfg.get("max_mmap_gb", MAX_MMAP_GB),
        "results_dir": str(RESULTS),
    }
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    src_root = cfg.get("src_root")
    if src_root:
        src = Path(src_root) / "src"
        if src.is_dir():
            env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [PYTHON, str(WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(HERE),
        env=env,
    )
    holder: dict[str, str] = {}

    def _communicate() -> None:
        out, err = proc.communicate(input=json.dumps(cfg))
        holder["stdout"] = out
        holder["stderr"] = err

    reader = threading.Thread(target=_communicate, daemon=True)
    reader.start()
    deadline = time.time() + timeout_s
    aborted: str | None = None
    while reader.is_alive():
        ram = host_ram_gb()
        disk = host_disk_gb("/")
        if ram["available_gb"] < abort_ram_gb:
            aborted = f"parent_host_ram:{ram['available_gb']:.2f}GB"
            _kill_process_tree(proc)
            break
        if disk["free_gb"] < abort_disk_gb:
            aborted = f"parent_disk:{disk['free_gb']:.2f}GB"
            _kill_process_tree(proc)
            break
        if time.time() > deadline:
            aborted = "timeout"
            _kill_process_tree(proc)
            break
        reader.join(0.2)
    reader.join(timeout=2.0)
    elapsed = time.perf_counter() - t0
    stdout = holder.get("stdout", "")
    stderr = holder.get("stderr", "")
    if aborted:
        return {
            "ok": False,
            "error": aborted,
            "returncode": proc.returncode,
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
            "elapsed_s": elapsed,
            "host": host,
            "cfg": {k: v for k, v in cfg.items() if k != "results_dir"},
        }
    if proc.returncode not in (0, None):
        lines = [ln for ln in stdout.splitlines() if ln.strip()]
        if lines:
            try:
                parsed = json.loads(lines[-1])
                parsed["returncode"] = proc.returncode
                parsed["elapsed_s"] = elapsed
                parsed["host"] = host
                parsed["cfg"] = {k: v for k, v in cfg.items() if k != "results_dir"}
                return parsed
            except json.JSONDecodeError:
                pass
        return {
            "ok": False,
            "returncode": proc.returncode,
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
            "elapsed_s": elapsed,
            "host": host,
            "cfg": {k: v for k, v in cfg.items() if k != "results_dir"},
        }
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return {
            "ok": False,
            "error": "empty worker stdout",
            "stderr": stderr[-4000:],
            "elapsed_s": elapsed,
            "host": host,
            "cfg": {k: v for k, v in cfg.items() if k != "results_dir"},
        }
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error": f"invalid worker json: {exc}",
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
            "elapsed_s": elapsed,
            "host": host,
            "cfg": {k: v for k, v in cfg.items() if k != "results_dir"},
        }
    result["host"] = host
    result["elapsed_s"] = elapsed
    result["cfg"] = {k: v for k, v in cfg.items() if k != "results_dir"}
    return result


def max_abs_diff(path_a: str | Path | None, path_b: str | Path | None) -> float | None:
    if not path_a or not path_b:
        return None
    a = np.load(path_a)
    b = np.load(path_b)
    if a.shape != b.shape:
        return None
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


if __name__ == "__main__":
    print(json.dumps({"python": sys.executable, "ram": host_ram_gb(), "disk": host_disk_gb()}, indent=2))
