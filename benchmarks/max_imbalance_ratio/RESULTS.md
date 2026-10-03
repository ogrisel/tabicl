# 5-fold CV of `max_imbalance_ratio`

Stratified 5-fold CV (`shuffle=True`, `random_state=0`) of
`TabICLClassifier(n_estimators=4, device="cpu")` on TabArena-v0.1
classification tasks. Each ensemble member draws its own majority subsample.
APSFailure (76k rows) and kddcup09_appetency (50k rows) are stratified down
to 12,000 rows before the split; every other task is used in full. Caps are
`{1, 5, 20, None}`. The same folds are reused for every cap.

Points are fold means. Horizontal bars are the standard deviation of
fit+predict time, vertical bars the standard deviation of the metric. The
orange line is the Pareto front of the means. Fold-level rows are in
`results_cv.csv`; means are in `results_cv_summary.csv`.

A cap at or above the training fold's natural ratio does not drop rows.
ROC-AUC and log-loss then match the uncapped run on every fold (seismic-bumps,
coil2000, polish, and students dropout at cap 20; students dropout, ratio
2.8, also at cap 5).

Binary ROC-AUC does not move because of the Elkan correction itself: for two
classes the map is strictly increasing. AUC gaps are the effect of the shorter
context. Log-loss moves for both reasons. With more than two classes the
normalizer depends on the features, so one-vs-rest AUC can move as well.

## ROC-AUC

![ROC-AUC, 5-fold mean ± 1 std](pareto_member_roc_auc.png)

## Log-loss

![Log-loss, 5-fold mean ± 1 std](pareto_member_log_loss.png)

## Paired comparison with the full context

`Δ` is the mean over folds of (capped − uncapped) on the same fold, and the
± term is the standard deviation of that paired difference.

| Task | Ratio | Cap 20 time | Speedup | Δ ROC-AUC | Δ log-loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| APSFailure (n=12000) | 54× | 45.5 s vs 93.3 s | 2.1× | +0.0000 ± 0.0005 | +0.0004 ± 0.0016 |
| kddcup09_appetency (n=12000) | 55× | 50.7 s vs 106.0 s | 2.1× | +0.0025 ± 0.0021 | −0.0013 ± 0.0007 |
| taiwanese bankruptcy | 30× | 22.6 s vs 31.7 s | 1.4× | −0.0002 ± 0.0009 | −0.0003 ± 0.0006 |
| MIC (8 classes) | 119× | 4.40 s vs 8.06 s | 1.8× | −0.0018 ± 0.0038 | −0.0046 ± 0.0074 |
| anneal (5 classes) | 86× | 1.13 s vs 1.59 s | 1.4× | −0.0004 ± 0.0009 | +0.0109 ± 0.0121 |
| coil2000, polish, seismic, students | ≤16× | same context | 1× | 0 | 0 |

Cap 5 is faster (about 4.1× on APSFailure and 3.9× on kddcup09_appetency) but
MIC loses 0.0078 ± 0.0068 ROC-AUC, and anneal log-loss rises by 0.036 ± 0.040.
Cap 1 is faster still and loses more, especially on MIC (−0.037 ± 0.016
ROC-AUC) and anneal (log-loss +0.165 ± 0.088).

## Recommendation

The estimator default is **`None`**, so the full training context is used
unless a cap is requested.

When enabling the cap, **20** is the value in this grid whose ROC-AUC and
log-loss stay within one standard deviation of the paired fold differences
on every task. On tasks whose natural ratio is already below 20 it is exactly
a no-op. On the ratio-50+ tasks it cuts fit+predict time by about 2.1× with
no ROC-AUC loss. Caps of 5 and below save more time, but MIC and anneal move
beyond that fold noise.
