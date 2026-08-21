"""Baseline CPU vs MPS RSS for small Friedman tables (no source patches)."""

from __future__ import annotations

from common import RESULTS, append_jsonl, dump_json, max_abs_diff, run_isolated

OUT = RESULTS / "01_baseline.jsonl"
SUMMARY = RESULTS / "01_baseline.json"

LADDER = [
    {"n_train": 128, "n_features": 10},
    {"n_train": 256, "n_features": 10},
    {"n_train": 512, "n_features": 10},
    {"n_train": 512, "n_features": 30},
]


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    rows = []
    for spec in LADDER:
        cpu_tag = f"cpu_n{spec['n_train']}_p{spec['n_features']}"
        mps_tag = f"mps_n{spec['n_train']}_p{spec['n_features']}"
        cpu = run_isolated(
            {
                **spec,
                "n_test": 64,
                "n_estimators": 1,
                "device": "cpu",
                "tag": cpu_tag,
                "use_amp": False,
            }
        )
        append_jsonl(OUT, cpu)
        rows.append(cpu)
        print(
            f"CPU  n={spec['n_train']:4d} p={spec['n_features']:2d} "
            f"ok={cpu.get('ok')} peak_rss={cpu.get('peak_rss_delta_mb')} "
            f"fit={cpu.get('fit_s')} pred={cpu.get('predict_s')}",
            flush=True,
        )
        if not cpu.get("ok"):
            break

        mps = run_isolated(
            {
                **spec,
                "n_test": 64,
                "n_estimators": 1,
                "device": "mps",
                "tag": mps_tag,
                "use_amp": "auto",
            }
        )
        mps["max_abs_diff_vs_cpu"] = max_abs_diff(cpu.get("pred_path"), mps.get("pred_path"))
        append_jsonl(OUT, mps)
        rows.append(mps)
        print(
            f"MPS  n={spec['n_train']:4d} p={spec['n_features']:2d} "
            f"ok={mps.get('ok')} peak_rss={mps.get('peak_rss_delta_mb')} "
            f"mps_alloc={mps.get('peak_mps_alloc_mb')} "
            f"fit={mps.get('fit_s')} pred={mps.get('predict_s')} "
            f"diff={mps.get('max_abs_diff_vs_cpu')}",
            flush=True,
        )
        if not mps.get("ok"):
            break

    dump_json(SUMMARY, rows)
    print(f"wrote {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
