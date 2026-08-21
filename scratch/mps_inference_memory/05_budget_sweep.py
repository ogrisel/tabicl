"""Sweep activation budgets at 4096 rows to pick a default for MPS."""

from __future__ import annotations

from common import RESULTS, append_jsonl, dump_json, max_abs_diff, run_isolated

OUT = RESULTS / "05_budget_sweep.jsonl"
SUMMARY = RESULTS / "05_budget_sweep.json"


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    rows = []
    ref = None
    for name, patches, budget in [
        ("baseline", [], 512.0),
        ("budget512", ["activation_budget"], 512.0),
        ("budget256", ["activation_budget"], 256.0),
        ("budget128", ["activation_budget"], 128.0),
    ]:
        result = run_isolated(
            {
                "n_train": 4096,
                "n_features": 30,
                "n_test": 64,
                "n_estimators": 1,
                "device": "mps",
                "use_amp": True,
                "offload_mode": False,
                "patches": patches,
                "tag": f"{name}_n4096",
                "max_rss_delta_gb": 8.0,
                "cpu_memory_budget_mb": budget,
            }
        )
        if name == "baseline" and result.get("ok"):
            ref = result.get("pred_path")
        result["probe"] = name
        result["max_abs_diff_vs_baseline"] = max_abs_diff(ref, result.get("pred_path"))
        append_jsonl(OUT, result)
        rows.append(result)
        print(
            f"{name:12s} ok={result.get('ok')} rssΔ={result.get('peak_rss_delta_mb')} "
            f"mps={result.get('peak_mps_alloc_mb')} pred_s={result.get('predict_s')} "
            f"bs_col={result.get('est_bs_tf_col')} bs_row={result.get('est_bs_tf_row')} "
            f"diff={result.get('max_abs_diff_vs_baseline')}",
            flush=True,
        )
        if not result.get("ok") and name == "baseline":
            break
    dump_json(SUMMARY, rows)


if __name__ == "__main__":
    main()
