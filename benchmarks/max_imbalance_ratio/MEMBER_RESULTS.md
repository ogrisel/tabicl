# Independent subsamples, 4 estimators

Same TabArena tasks and 5-fold protocol as `RESULTS.md`, with
`n_estimators=4`. Each member draws its own majority subsample. Caps are
`{1, 5, 20, None}`. APSFailure and kddcup09_appetency are stratified to
12,000 rows. Fold rows are in `results_member_cv.csv`; means are in
`results_member_cv_summary.csv`.

Points are fold means. Horizontal bars are the standard deviation of
fit+predict time, vertical bars the standard deviation of the metric.

![ROC-AUC, 4 estimators](pareto_member_roc_auc.png)

![Log-loss, 4 estimators](pareto_member_log_loss.png)

With one estimator the ratio sweep in `RESULTS.md` is the whole story. With
four estimators the cap still trades a shorter context for speed. At cap 20,
tasks whose natural ratio is already below 20 match the uncapped run. On
APSFailure (ratio 54×, n=12000) cap 20 is 45 s versus 93 s for the full
context, with ROC-AUC 0.992 either way. On MIC (8 classes, ratio 119×) cap 1
reaches ROC-AUC 0.889 in 2.8 s, against 0.926 and 8.1 s with no cap.
