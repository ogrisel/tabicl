"""Compare this PR vs its parent on growing Friedman tables, with a hard abort.

Each (variant, n_train) run is a subprocess. The child watchdog and this
parent both kill the worker if host RAM, disk, process RSS, or MPS driver
allocations approach the caps below. The ladder stops a variant after an abort.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from common import (
    RESULTS,
    WORKER_TIMEOUT_S,
    append_jsonl,
    dump_json,
    host_disk_gb,
    host_ram_gb,
    max_abs_diff,
    preflight_or_skip,
    run_isolated,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = RESULTS / "06_scale_pr.jsonl"
SUMMARY = RESULTS / "06_scale_pr.json"
PARENT_WORKTREE = Path(
    os.environ.get("TABICL_PARENT_WORKTREE", "/tmp/tabicl-parent-cpu-stream")
)
PARENT_REF = os.environ.get("TABICL_PARENT_REF", "cursor/cpu-streaming-followup-e710")

# Caps: leave several GB for the rest of the machine.
ABORT_HOST_RAM_GB = 4.0
ABORT_DISK_GB = 15.0
MAX_MPS_DRIVER_GB = 8.0
N_FEATURES = 30
N_TEST = 64
N_ESTIMATORS = 1

# Start small; stop a variant after the first abort or skip.
LADDER = [512, 1024, 2048, 4096, 8192, 12288]


def ensure_parent_worktree() -> Path:
    marker = PARENT_WORKTREE / "src" / "tabicl"
    if marker.is_dir():
        return PARENT_WORKTREE
    subprocess.run(
        ["git", "worktree", "add", str(PARENT_WORKTREE), PARENT_REF],
        cwd=str(REPO),
        check=True,
    )
    return PARENT_WORKTREE


def rss_cap_gb(n_train: int) -> float:
    if n_train >= 8192:
        return 6.0
    if n_train >= 4096:
        return 5.0
    return 4.0


def timeout_s(n_train: int) -> int:
    if n_train >= 12288:
        return 420
    if n_train >= 8192:
        return 300
    return WORKER_TIMEOUT_S


def main() -> None:
    preflight_or_skip()
    parent_root = ensure_parent_worktree()
    pr_root = REPO
    if OUT.exists():
        OUT.unlink()

    variants = [
        ("pr", pr_root),
        ("parent", parent_root),
    ]
    stopped = set()
    rows = []
    refs: dict[int, str] = {}

    for n_train in LADDER:
        ram = host_ram_gb()
        disk = host_disk_gb("/")
        print(
            f"\n--- n_train={n_train} host_avail={ram['available_gb']:.2f}GB "
            f"disk_free={disk['free_gb']:.2f}GB ---",
            flush=True,
        )
        if ram["available_gb"] < ABORT_HOST_RAM_GB + 2.0:
            print("skip remaining ladder: host RAM too tight before start", flush=True)
            break
        if disk["free_gb"] < ABORT_DISK_GB + 5.0:
            print("skip remaining ladder: disk too tight before start", flush=True)
            break

        for name, src_root in variants:
            if name in stopped:
                continue
            tag = f"{name}_n{n_train}_p{N_FEATURES}"
            result = run_isolated(
                {
                    "n_train": n_train,
                    "n_features": N_FEATURES,
                    "n_test": N_TEST,
                    "n_estimators": N_ESTIMATORS,
                    "device": "mps",
                    "use_amp": True,
                    "offload_mode": False,
                    "tag": tag,
                    "src_root": str(src_root),
                    "max_rss_delta_gb": rss_cap_gb(n_train),
                    "max_mps_driver_gb": MAX_MPS_DRIVER_GB,
                    "min_host_avail_gb": ABORT_HOST_RAM_GB,
                    "min_disk_free_gb": ABORT_DISK_GB,
                },
                timeout_s=timeout_s(n_train),
            )
            result["variant"] = name
            if name == "pr" and result.get("ok"):
                refs[n_train] = result.get("pred_path")
            result["max_abs_diff_vs_pr"] = max_abs_diff(refs.get(n_train), result.get("pred_path"))
            append_jsonl(OUT, result)
            rows.append(result)
            print(
                f"{tag:28s} ok={result.get('ok')} err={result.get('error')} "
                f"rssΔ={result.get('peak_rss_delta_mb')} "
                f"mps={result.get('peak_mps_alloc_mb')} "
                f"drv={result.get('peak_mps_driver_mb')} "
                f"fit={result.get('fit_s')} pred={result.get('predict_s')} "
                f"src={result.get('tabicl_file')} diff={result.get('max_abs_diff_vs_pr')}",
                flush=True,
            )
            if not result.get("ok"):
                stopped.add(name)
                print(f"stop {name} ladder after abort/failure", flush=True)

    dump_json(SUMMARY, rows)
    print(f"\nwrote {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
