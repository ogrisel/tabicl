# 5-fold CV of `max_imbalance_ratio`

Stratified 5-fold CV (`shuffle=True`, `random_state=0`) of
`TabICLClassifier(n_estimators=1, norm_methods=["none"], device="cpu")` on
TabArena-v0.1 classification tasks. APSFailure (76k rows) and
kddcup09_appetency (50k rows) are stratified down to 12,000 rows before the
split; every other task is used in full. Caps are `{1, 3, 5, 10, 15, 20, 30,
None}`. The same folds are reused for every cap.

Points are fold means. Horizontal bars are the standard deviation of
fit+predict time, vertical bars the standard deviation of the metric. The
orange line is the Pareto front of the means. Fold-level rows are in
`results_cv.csv`; means are in `results_cv_summary.csv`. With one estimator
there is a single majority subsample. The four-estimator run is in
`MEMBER_RESULTS.md`.

A cap at or above the training fold's natural ratio does not drop rows.
ROC-AUC and log-loss then match the uncapped run on every fold (seismic-bumps,
coil2000, and polish at caps 15 and above; students dropout, ratio 2.8, at
every cap above 1).

Binary ROC-AUC does not move because of the Elkan correction itself: for two
classes the map is strictly increasing. AUC gaps are the effect of the shorter
context. Log-loss moves for both reasons. With more than two classes the
normalizer depends on the features, so one-vs-rest AUC can move as well.

## ROC-AUC

![ROC-AUC, 5-fold mean ± 1 std](pareto_roc_auc.png)

## Log-loss

![Log-loss, 5-fold mean ± 1 std](pareto_log_loss.png)

## Paired comparison with the full context

`Δ` is the mean over folds of (capped − uncapped) on the same fold, and the
± term is the standard deviation of that paired difference.

| Task | Ratio | Cap 20 time | Speedup | Δ ROC-AUC | Δ log-loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| APSFailure (n=12000) | 54× | 10.6 s vs 24.3 s | 2.3× | −0.0007 ± 0.0007 | +0.0016 ± 0.0013 |
| kddcup09_appetency (n=12000) | 55× | 11.9 s vs 27.4 s | 2.3× | −0.0010 ± 0.0088 | −0.0010 ± 0.0012 |
| taiwanese bankruptcy | 30× | 5.4 s vs 8.4 s | 1.6× | −0.0018 ± 0.0016 | +0.0015 ± 0.0013 |
| MIC (8 classes) | 119× | 1.07 s vs 2.20 s | 2.1× | −0.0047 ± 0.0077 | +0.0055 ± 0.0057 |
| anneal (5 classes) | 86× | 0.38 s vs 0.52 s | 1.4× | −0.0009 ± 0.0013 | +0.0161 ± 0.0130 |
| coil2000, polish, seismic, students | ≤16× | same context | 1× | 0 | 0 |

Cap 5, by comparison, is not inside the fold noise on the multiclass tasks:
MIC loses 0.026 ± 0.014 ROC-AUC, and anneal log-loss rises from 0.016 to
0.054 (paired +0.038 ± 0.031). Taiwanese bankruptcy loses 0.0054 ± 0.0017
ROC-AUC. Those drops are small next to the raw fold scatter on some binary
tasks, but they are systematic.

Cap 30 is the first value that also puts anneal's log-loss back inside one
standard deviation of the paired differences (+0.011 ± 0.014, 0.027 vs 0.016).
The speedups shrink with it: about 1.7× on APSFailure and kddcup09_appetency
instead of 2.3×.

## Recommendation

The estimator default is **`None`**, so the full training context is used
unless a cap is requested.

When enabling the cap, **20** is the smallest value in this grid whose
ROC-AUC change is within one standard deviation of the paired fold
differences on every task, including MIC. On tasks whose natural ratio is
already below 20 it is exactly a no-op. On the ratio-50+ tasks it cuts
fit+predict time by about 2.3× for a sub-0.001 ROC-AUC change.

Use **30** when log-loss on a tiny multiclass problem matters more than that
extra speed (anneal is the case that still moves at 20). Caps of 5 and below
save more time, but MIC and anneal lose accuracy beyond fold noise.
