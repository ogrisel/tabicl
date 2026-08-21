"""Force ICL query-chunking to fire and compare against unified-memory capping."""

from __future__ import annotations

from common import RESULTS, append_jsonl, dump_json, max_abs_diff, run_isolated

OUT = RESULTS / "03_icl_force.jsonl"
SUMMARY = RESULTS / "03_icl_force.json"

# 8 MB budget: chunk ≈ 32k / (12 * 512 * 2 bytes) ≈ 680 query rows at fp16.
WORKLOADS = [
    {
        "n_train": 2048,
        "n_features": 30,
        "cpu_memory_budget_mb": 8.0,
        "max_rss_delta_gb": 6.0,
    },
    {
        "n_train": 4096,
        "n_features": 30,
        "cpu_memory_budget_mb": 8.0,
        "max_rss_delta_gb": 8.0,
    },
]

PROBES = [
    {"name": "baseline", "patches": [], "offload_mode": False},
    {"name": "H2_unified_cap", "patches": ["unified_mem_cap"], "offload_mode": False},
    {"name": "H4_query_chunks", "patches": ["query_chunks_mps"], "offload_mode": False},
    {
        "name": "combo",
        "patches": ["unified_mem_cap", "query_chunks_mps"],
        "offload_mode": False,
    },
]


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    rows = []
    refs: dict[int, str] = {}
    for work in WORKLOADS:
        for probe in PROBES:
            tag = f"{probe['name']}_n{work['n_train']}_b{int(work['cpu_memory_budget_mb'])}"
            result = run_isolated(
                {
                    "n_train": work["n_train"],
                    "n_features": work["n_features"],
                    "n_test": 64,
                    "n_estimators": 1,
                    "device": "mps",
                    "use_amp": True,
                    "offload_mode": probe["offload_mode"],
                    "patches": probe["patches"],
                    "tag": tag,
                    "max_rss_delta_gb": work["max_rss_delta_gb"],
                    "cpu_memory_budget_mb": work["cpu_memory_budget_mb"],
                }
            )
            if probe["name"] == "baseline" and result.get("ok"):
                refs[work["n_train"]] = result.get("pred_path")
            result["probe"] = probe["name"]
            result["max_abs_diff_vs_baseline"] = max_abs_diff(
                refs.get(work["n_train"]), result.get("pred_path")
            )
            append_jsonl(OUT, result)
            rows.append(result)
            print(
                f"{tag:36s} ok={result.get('ok')} rssΔ={result.get('peak_rss_delta_mb')} "
                f"mps={result.get('peak_mps_alloc_mb')} pred_s={result.get('predict_s')} "
                f"chunk={result.get('icl_would_chunk')} seq={result.get('icl_seq')} "
                f"cs={result.get('icl_chunk')} diff={result.get('max_abs_diff_vs_baseline')}",
                flush=True,
            )
            if not result.get("ok") and probe["name"] == "baseline":
                print("baseline failed; skip remaining probes for this n", flush=True)
                break
    dump_json(SUMMARY, rows)
    print(f"wrote {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
