# MPS inference memory (unified RAM)

Local M4, 32 GB unified, torch 2.10.0, conda env `tabicl`.
Branch at the time of probes: `cursor/cpu-streaming-followup-e710` (fork PR #2).

Isolated-process RSS + `torch.mps.current_allocated_memory` on
`TabICLRegressor` (1 estimator, `norm_methods="none"`, Friedman tables).

## Baseline (no patches)

| n_train | features | CPU RSS Δ | MPS RSS Δ | MPS alloc | predict CPU | predict MPS |
|--------:|---------:|----------:|----------:|----------:|------------:|------------:|
| 128 | 10 | 236 MB | 315 MB | 116 MB | 0.09 s | 0.47 s |
| 256 | 10 | 277 MB | 322 MB | 117 MB | 0.14 s | 0.51 s |
| 512 | 10 | 333 MB | 313 MB | 189 MB | 0.27 s | 0.60 s |
| 512 | 30 | 463 MB | 316 MB | 251 MB | 0.48 s | 0.43 s |

Predictions agree within ~1e-4 (MPS numerics). Process RSS on MPS is almost
flat at this scale; the useful signal is `current_allocated_memory`.

`torch.mps.recommended_max_memory()` is ~25 GB while the host often has ~10 GB
free. AUTO offload treats CPU as a second pool; on unified memory it is a copy.

## Probes (2048×30 and 4096×30, AMP on)

- **CPU offload / pin_memory:** no RSS reduction. Disk offload at this size
  was slightly heavier and showed ~3e-2 pred drift from mmap copies.
- **ICL query chunking on MPS:** algebraically exact (`max_abs_diff=0` vs
  full SDPA). Did **not** cut the peak at 2k–4k rows: the column/row stages
  still dominate, and default `cpu_memory_budget_mb=512` only chunks ICL
  past ~20k rows. Slightly slower when forced with an 8 MB budget.
- **Unified-memory cap** (`min(mps_free, host_available)`): sometimes lower
  MPS alloc at 2048 (e.g. 426→311 MB), little effect at 4096 because
  `MemoryEstimator` still fits the whole feature batch in several GB.
- **Activation-budgeted batching** (PR #1 idea on the accelerator path):
  `target_mem = min(available * safety, budget)` using `MemoryEstimator`.

  At 4096×30, stay-on-GPU, AMP:

  | budget | MPS alloc | predict | col batch | row batch |
  |--------|----------:|--------:|----------:|----------:|
  | none (baseline) | 426 MB | 2.52 s | all | all |
  | 512 MB | 368 MB | 2.53 s | 18 | 2045 |
  | 128 MB | 218 MB | 14.3 s | 1 | 1 |

  512 MB is the CPU default: ~14% less MPS alloc, same speed.
  128 MB is ~2× less MPS alloc and ~5.7× slower. Pred diffs ~0.02 from
  batch reassociation under AMP.

## PR #3 takeaway

Port the CPU *policy*, not the CPU *path*:

1. Cap MPS free memory by host available RAM.
2. On unified-memory devices, also cap auto-batch `target_mem` by
   `cpu_memory_budget_mb` (default 512).
3. AUTO offload must not treat CPU as a distinct pool (GPU or disk).
4. Do not pin host buffers on MPS.
5. Allow ICL query chunking on MPS (same budget); it is exact and becomes
   the ICL safety net once `n_rows` exceeds the budgeted chunk size.

## Scale: this PR vs parent (`722e9e1`)

Isolated subprocesses, 1 estimator, 30 features, 64 test rows, AMP on,
`offload=False`. Child + parent watchdogs abort if host available RAM < 4 GB,
disk free < 15 GB, process RSS Δ exceeds 4–6 GB, or MPS **driver** allocated
exceeds **8 GB**.

| n_train | PR MPS alloc | parent MPS alloc | PR predict | parent predict | notes |
|--------:|-------------:|-----------------:|-----------:|---------------:|---|
| 512 | 235 MB | 165 MB | 0.69 s | 0.54 s | small-n noise; both fine |
| 1024 | 246 MB | 283 MB | 0.64 s | 0.62 s | |
| 2048 | 405 MB | 426 MB | 1.13 s | 1.07 s | pred diff 0.01 |
| 4096 | **364 MB** | 636 MB | 2.54 s | 2.48 s | **43% less MPS alloc**; pred diff 0.019 |
| 8192 | **558 MB** | aborted | 7.12 s | — | parent hit **11.4 GB** MPS driver (cap 8 GB) |
| 12288 | aborted | not run | — | — | PR hit **14.1 GB** MPS driver (cap 8 GB) |

Process RSS Δ stayed ~320–340 MB even at 8k rows: the growing working set
shows up in `torch.mps.driver_allocated_memory`, not in `psutil` RSS. That
is why the driver cap is the one that fires.

The PR does **not** make 12k-row ICL cheap — quadratic attention still fills
the Metal heap — but it **does** complete 8k×30 where the parent path does
not, under the same 8 GB driver abort. Default `cpu_memory_budget_mb=512`
is enough for column/row batching at 4k–8k; ICL query chunking at the
default budget is still a no-op until ~20k rows, so 12k remains ICL-bound.

Disk was never the limiter (~34 GB free throughout). After the 8k parent
abort, host available RAM dipped to 5.5 GB and the 12k step was postponed
until RAM recovered (~13 GB), then PR-only 12k was attempted and aborted
on the driver cap.

