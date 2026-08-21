# MPS inference memory harness

Temporary probe scripts for comparing TabICL MPS memory on unified-memory
hosts. Not part of the library API. Safe to drop from the PR later.

Each numbered script launches `worker.py` in a **subprocess**. The child
samples RSS, host RAM, disk, and `torch.mps.driver_allocated_memory` and
aborts (exit 96–99) before saturating the machine. `common.run_isolated`
also kills the process tree if host RAM or disk get too low.

## Run

Use the same Python that has this checkout installed (conda env `tabicl`
on the original host):

```bash
cd scratch/mps_inference_memory
python 01_baseline.py          # small CPU vs MPS ladder
python 06_scale_compare.py     # this PR vs parent, growing n_train
```

Optional environment:

| Variable | Default | Meaning |
|---|---|---|
| `TABICL_PYTHON` | `sys.executable` | Interpreter for worker subprocesses |
| `TABICL_PARENT_REF` | `cursor/cpu-streaming-followup-e710` | Git ref for the pre-PR tree |
| `TABICL_PARENT_WORKTREE` | `/tmp/tabicl-parent-cpu-stream` | Where `06_scale_compare.py` adds that worktree |

`06_scale_compare.py` runs `git worktree add` once if the parent tree is
missing. Override `TABICL_PARENT_REF` if this branch is rebased.

Default abort caps (overridable per worker config): host available RAM
4 GB, disk free 15 GB, process RSS delta 4–6 GB, MPS driver 8 GB. Raise
`MAX_MPS_DRIVER_GB` in `06_scale_compare.py` only if the machine has
plenty of unified RAM left.

## Layout

- `common.py` / `worker.py` — isolated fit/predict + watchdogs
- `01_baseline.py` … `05_budget_sweep.py` — ablations that led to the PR
- `06_scale_compare.py` — PR vs parent scale ladder
- `results/*.json{,l}` — last run on Apple M4, 32 GB, torch 2.10.0
- Prediction `.npy` files are gitignored (tiny test slices, regenerable)
