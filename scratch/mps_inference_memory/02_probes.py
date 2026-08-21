"""Ablate unified-memory hypotheses H2–H6 against a fixed MPS workload."""

from __future__ import annotations

from common import RESULTS, append_jsonl, dump_json, max_abs_diff, run_isolated

OUT = RESULTS / "02_probes.jsonl"
SUMMARY = RESULTS / "02_probes.json"

# 512-row runs check correctness. 2048-row runs are large enough for ICL
# query-chunking to fire once cpu_memory_budget_mb is tightened to 32 MB.
WORKLOADS = [
    {
        "n_train": 512,
        "n_features": 10,
        "max_rss_delta_gb": 4.0,
        "cpu_memory_budget_mb": 32.0,
    },
    {
        "n_train": 2048,
        "n_features": 30,
        "max_rss_delta_gb": 6.0,
        "cpu_memory_budget_mb": 32.0,
    },
]

PROBES = [
    {"name": "baseline", "patches": [], "offload_mode": "auto"},
    {"name": "H2_unified_cap", "patches": ["unified_mem_cap"], "offload_mode": "auto"},
    {"name": "H3_offload_gpu", "patches": [], "offload_mode": False},
    {"name": "H3_offload_cpu", "patches": [], "offload_mode": True},
    {"name": "H3_offload_disk", "patches": [], "offload_mode": "disk"},
    {
        "name": "H5_disk_release",
        "patches": ["disk_release_each_write"],
        "offload_mode": "disk",
    },
    {"name": "H4_query_chunks", "patches": ["query_chunks_mps"], "offload_mode": "auto"},
    {"name": "H6_no_pin", "patches": ["no_pin_memory"], "offload_mode": True},
    {
        "name": "combo_cap_chunks",
        "patches": ["unified_mem_cap", "query_chunks_mps"],
        "offload_mode": "auto",
    },
]


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    rows = []
    refs: dict[tuple[int, int], str] = {}

    for work in WORKLOADS:
        for probe in PROBES:
            tag = f"{probe['name']}_n{work['n_train']}_p{work['n_features']}"
            result = run_isolated(
                {
                    "n_train": work["n_train"],
                    "n_features": work["n_features"],
                    "n_test": 64,
                    "n_estimators": 1,
                    "device": "mps",
                    "use_amp": "auto",
                    "offload_mode": probe["offload_mode"],
                    "patches": probe["patches"],
                    "tag": tag,
                    "max_rss_delta_gb": work["max_rss_delta_gb"],
                    "cpu_memory_budget_mb": work["cpu_memory_budget_mb"],
                }
            )
            key = (work["n_train"], work["n_features"])
            if probe["name"] == "baseline" and result.get("ok"):
                refs[key] = result.get("pred_path")
            result["probe"] = probe["name"]
            result["max_abs_diff_vs_baseline"] = max_abs_diff(
                refs.get(key), result.get("pred_path")
            )
            append_jsonl(OUT, result)
            rows.append(result)
            print(
                f"{tag:40s} ok={result.get('ok')} "
                f"rssΔ={result.get('peak_rss_delta_mb')} "
                f"mps={result.get('peak_mps_alloc_mb')} "
                f"fit={result.get('fit_s')} pred={result.get('predict_s')} "
                f"diff={result.get('max_abs_diff_vs_baseline')}",
                flush=True,
            )
            if not result.get("ok") and probe["name"] == "baseline":
                print("baseline failed; stopping this workload", flush=True)
                break

    dump_json(SUMMARY, rows)
    print(f"wrote {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
