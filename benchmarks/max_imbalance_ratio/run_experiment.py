"""5-fold CV of class-aware vs size-matched uniform subsampling.

Each ensemble member draws its own subsample. The class-aware draw is
``TabICLClassifier(max_imbalance_ratio=...)``. The uniform draw keeps the
same number of rows but chooses them uniformly, with no Elkan correction.
Caps that do not change the training set are evaluated once and reused.

``n_estimators`` in ``{1, 4, 8}`` and caps in
``{1, 3, 5, 10, 15, 20, 30, None}``. Completed folds are appended to
``results_cv.csv`` and skipped on a rerun.

Fits expected to take more than two minutes are not started. The estimate
scales the previous ``n_estimators=4`` timings on this machine by the
observed cost of 1 and 8 members. Those skips are listed in
``skipped_cv.csv``.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from tabicl import TabICLClassifier

warnings.filterwarnings(
    "ignore",
    message="The following categorical columns have a cardinality above 40",
)


# TabArena-v0.1 classification tasks with a heavy class skew, plus one mild
# multiclass control (students dropout, ratio about 3).
TABARENA_TASKS = {
    "seismic-bumps": 46956,
    "taiwanese_bankruptcy_prediction": 46962,
    "polish_companies_bankruptcy": 46950,
    "coil2000_insurance_policies": 46916,
    "anneal": 46906,
    "students_dropout_and_academic_success": 46960,
    "MIC": 46980,
    "kddcup09_appetency": 46939,
    "APSFailure": 46908,
}

DATASET_ORDER = [
    "APSFailure",
    "kddcup09_appetency",
    "taiwanese_bankruptcy_prediction",
    "coil2000_insurance_policies",
    "polish_companies_bankruptcy",
    "seismic-bumps",
    "anneal",
    "MIC",
    "students_dropout_and_academic_success",
]

# Skip a fit whose expected fit+predict time exceeds two minutes.
TIME_LIMIT_S = 120.0

# Mean fit+predict seconds at n_estimators=4 on this CPU, as (n_context, seconds).
# Measured in the previous 5-fold run and used to decide which large-task
# configs to skip before paying for them.
_N4_ANCHORS: dict[str, list[tuple[float, float]]] = {
    "APSFailure": [(347.2, 17.80), (1041.6, 22.63), (3645.6, 45.49), (9600.0, 93.26)],
    "kddcup09_appetency": [(342.4, 21.63), (1027.2, 27.33), (3595.2, 50.73), (9600.0, 106.04)],
    "coil2000_insurance_policies": [(937.6, 10.65), (2812.8, 19.12), (7857.6, 48.36)],
    "taiwanese_bankruptcy_prediction": [(352.0, 6.66), (1056.0, 9.71), (3696.0, 22.61), (5455.2, 31.66)],
    "polish_companies_bankruptcy": [(656.0, 5.28), (1968.0, 10.00), (4728.0, 20.71)],
    "students_dropout_and_academic_success": [(1905.6, 5.91), (3539.2, 10.30)],
    "seismic-bumps": [(272.0, 1.01), (816.0, 1.58), (2067.2, 3.58)],
    "MIC": [(76.8, 2.83), (224.8, 3.51), (408.8, 4.40), (1359.2, 8.06)],
    "anneal": [(32.0, 0.72), (133.2, 0.89), (299.2, 1.13), (718.4, 1.59)],
}

# Cost relative to n_estimators=4, measured on the small tasks in this run.
_N_EST_SCALE = {1: 0.32, 4: 1.0, 8: 2.1}


def _parse_ratio(text: str):
    if text.lower() in {"none", "inf", "unlimited"}:
        return None
    return float(text)


def _ratio_label(value) -> str:
    if value is None or (isinstance(value, float) and np.isinf(value)):
        return "none"
    return f"{float(value):g}"


def _class_counts(y) -> np.ndarray:
    _, counts = np.unique(np.asarray(y), return_counts=True)
    return counts.astype(np.int64)


def _scores(y_true, proba, classes) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    if len(classes) == 2:
        auc = float(roc_auc_score(y_true == classes[1], proba[:, 1]))
    else:
        auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=classes))
    loss = float(log_loss(y_true, proba, labels=classes))
    return auc, loss


def _load_task(name: str):
    data_id = TABARENA_TASKS[name]
    frame, y = fetch_openml(data_id=data_id, as_frame=True, return_X_y=True, parser="auto")
    y = pd.Series(np.asarray(y), name="target").reset_index(drop=True)
    return frame.reset_index(drop=True), y


def _stratified_take(X, y, n_samples: int, random_state: int):
    if n_samples >= len(y):
        return X, y
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
    index, _ = next(splitter.split(np.zeros(len(y)), y))
    return X.iloc[index].reset_index(drop=True), y.iloc[index].reset_index(drop=True)


def _already_ran(results: pd.DataFrame, row: dict) -> bool:
    if results.empty:
        return False
    keys = [
        "dataset",
        "strategy",
        "fold",
        "max_imbalance_ratio",
        "n_estimators",
        "n_splits",
        "max_rows",
        "seed",
    ]
    mask = np.ones(len(results), dtype=bool)
    for key in keys:
        mask &= results[key].astype(str) == str(row[key])
    return bool(mask.any())


def _context_size(y, ratio) -> int:
    """Rows kept by the class-aware cap. ``ratio is None`` keeps every row."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    if ratio is None or counts.max() / counts.min() <= float(ratio):
        return int(counts.sum())
    cap = max(int(np.floor(float(ratio) * float(counts.min()) + 1e-8)), 1)
    return int(sum(min(int(count), cap) for count in counts))


def _interp_time(points: list[tuple[float, float]], n_context: float) -> float:
    """Piecewise-linear fit+predict time as a function of context length."""
    points = sorted(points)
    if n_context <= points[0][0]:
        return float(points[0][1])
    if len(points) == 1 or n_context >= points[-1][0]:
        if len(points) == 1:
            return float(points[0][1])
        (x0, y0), (x1, y1) = points[-2], points[-1]
        slope = (y1 - y0) / max(x1 - x0, 1e-9)
        return float(max(y1, y1 + slope * (n_context - x1)))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= n_context <= x1:
            weight = (n_context - x0) / (x1 - x0)
            return float(y0 + weight * (y1 - y0))
    return float(points[-1][1])


def _n_est_scale(results: pd.DataFrame, dataset: str, n_estimators: int) -> float:
    """Seconds at this ensemble size divided by the n_estimators=4 anchor."""
    default = _N_EST_SCALE.get(int(n_estimators), (n_estimators / 4) ** 1.07)
    anchors = _N4_ANCHORS.get(dataset)
    if anchors is None or results.empty:
        return default
    block = results[(results["dataset"] == dataset) & (results["n_estimators"] == n_estimators)]
    if block.empty:
        return default
    ratios = []
    grouped = block.groupby("n_context")["fit_predict_seconds"].mean()
    for n_context, seconds in grouped.items():
        base = _interp_time(anchors, float(n_context))
        if base > 0:
            ratios.append(float(seconds) / base)
    if not ratios:
        return default
    # A larger measured ratio means this machine is slower than the anchor.
    # A smaller one is usually the fixed overhead on a short context, so keep
    # the default rather than under-predicting the long contexts.
    return float(max(default, np.median(ratios)))


def _expected_seconds(results: pd.DataFrame, dataset: str, n_estimators: int, n_context: int) -> float | None:
    """Predicted fit+predict seconds, or None when the task has no anchor."""
    anchors = _N4_ANCHORS.get(dataset)
    if anchors is None:
        return None
    return _interp_time(anchors, float(n_context)) * _n_est_scale(results, dataset, n_estimators)


def _align_proba(proba, model_classes, reference_classes):
    """Add epsilon columns for classes a uniform subsample dropped."""
    model_classes = np.asarray(model_classes)
    reference_classes = np.asarray(reference_classes)
    if model_classes.shape == reference_classes.shape and np.array_equal(model_classes, reference_classes):
        return proba
    aligned = np.full((len(proba), len(reference_classes)), 1e-15, dtype=np.float64)
    position = {label: i for i, label in enumerate(reference_classes)}
    for column, label in enumerate(model_classes):
        aligned[:, position[label]] = np.clip(proba[:, column], 1e-15, None)
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _pareto_mask(time_s: np.ndarray, score: np.ndarray, *, maximize: bool) -> np.ndarray:
    keep = np.ones(len(time_s), dtype=bool)
    for i in range(len(time_s)):
        others = np.ones(len(time_s), dtype=bool)
        others[i] = False
        not_worse_time = time_s[others] <= time_s[i] + 1e-12
        if maximize:
            not_worse_score = score[others] >= score[i] - 1e-12
            strictly = (time_s[others] < time_s[i] - 1e-12) | (score[others] > score[i] + 1e-12)
        else:
            not_worse_score = score[others] <= score[i] + 1e-12
            strictly = (time_s[others] < time_s[i] - 1e-12) | (score[others] < score[i] - 1e-12)
        if np.any(not_worse_time & not_worse_score & strictly):
            keep[i] = False
    return keep


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation of each cap across CV folds."""
    if results.empty:
        return results
    grouped = results.groupby(["dataset", "strategy", "n_estimators", "max_imbalance_ratio"], sort=False)
    rows = []
    for (dataset, strategy, n_estimators, ratio), group in grouped:
        rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "n_estimators": int(n_estimators),
                "max_imbalance_ratio": ratio,
                "n_folds": int(len(group)),
                "n_classes": int(group["n_classes"].iloc[0]),
                "n_rows": int(group["n_rows"].iloc[0]),
                "natural_ratio_mean": float(group["natural_ratio"].mean()),
                "n_context_mean": float(group["n_context"].mean()),
                "time_mean": float(group["fit_predict_seconds"].mean()),
                "time_std": float(group["fit_predict_seconds"].std(ddof=1)),
                "roc_auc_mean": float(group["roc_auc"].mean()),
                "roc_auc_std": float(group["roc_auc"].std(ddof=1)),
                "log_loss_mean": float(group["log_loss"].mean()),
                "log_loss_std": float(group["log_loss"].std(ddof=1)),
            }
        )
    summary = pd.DataFrame(rows)

    def _cap_order(value: object) -> float:
        text = str(value)
        if text == "none":
            return 1e9
        return float(text)

    summary["_order"] = summary["max_imbalance_ratio"].map(_cap_order)
    return summary.sort_values(["dataset", "_order"]).drop(columns="_order")


def _panel_figure(summary: pd.DataFrame, score: str, *, maximize: bool, title: str, ylabel: str, path: Path) -> None:
    present = [name for name in DATASET_ORDER if name in set(summary["dataset"])]
    if not present:
        return
    n_cols = 3 if len(present) > 3 else len(present)
    n_rows = int(np.ceil(len(present) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.6 * n_rows), squeeze=False)
    for axis in axes.ravel():
        axis.set_visible(False)
    mean_col = f"{score}_mean"
    std_col = f"{score}_std"
    for axis, dataset in zip(axes.ravel(), present):
        axis.set_visible(True)
        group = summary[summary["dataset"] == dataset].sort_values("time_mean")
        times = group["time_mean"].to_numpy()
        scores = group[mean_col].to_numpy()
        axis.errorbar(
            times,
            scores,
            xerr=group["time_std"].to_numpy(),
            yerr=group[std_col].to_numpy(),
            fmt="o",
            color="#1f4e79",
            ecolor="#8aa0b4",
            elinewidth=1.0,
            capsize=3,
            markersize=5,
            zorder=3,
        )
        for _, row in group.iterrows():
            axis.annotate(
                str(row["max_imbalance_ratio"]),
                (row["time_mean"], row[mean_col]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )
        front = group.iloc[np.flatnonzero(_pareto_mask(times, scores, maximize=maximize))]
        front = front.sort_values("time_mean")
        axis.plot(front["time_mean"], front[mean_col], color="#c45c26", linewidth=1.4, zorder=2)
        n_rows_used = int(group["n_rows"].iloc[0])
        ratio = float(group["natural_ratio_mean"].iloc[0])
        n_classes = int(group["n_classes"].iloc[0])
        short_name = dataset.replace("_and_academic_success", "").replace("_prediction", "").replace("_insurance_policies", "")
        axis.set_title(f"{short_name}\n{n_classes} classes, ratio {ratio:.0f}×, n={n_rows_used}", fontsize=9)
        axis.set_xlabel("fit + predict time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cv(results: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = summarize(results)
    if summary.empty:
        return summary
    summary = summary.loc[summary["n_folds"] >= 5].copy()
    summary.to_csv(output_dir / "results_cv_summary.csv", index=False)
    for n_estimators in sorted(summary["n_estimators"].unique()):
        subset = summary[summary["n_estimators"] == n_estimators]
        _strategy_figure(
            subset,
            "roc_auc",
            maximize=True,
            title=f"n_estimators={int(n_estimators)}: ROC-AUC vs fit+predict time",
            ylabel="ROC-AUC",
            path=output_dir / f"pareto_roc_auc_n{int(n_estimators)}.png",
        )
        _strategy_figure(
            subset,
            "log_loss",
            maximize=False,
            title=f"n_estimators={int(n_estimators)}: log-loss vs fit+predict time",
            ylabel="log-loss",
            path=output_dir / f"pareto_log_loss_n{int(n_estimators)}.png",
        )
    return summary


def _strategy_figure(summary, score, *, maximize, title, ylabel, path):
    datasets = [name for name in DATASET_ORDER if name in set(summary["dataset"])]
    if not datasets:
        return
    n_cols = 3
    n_rows = int(np.ceil(len(datasets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.6 * n_rows), squeeze=False)
    for axis in axes.ravel():
        axis.set_visible(False)
    colors = {"class_aware": "#1f4e79", "uniform": "#c45c26"}
    mean_col = f"{score}_mean"
    std_col = f"{score}_std"
    for axis, dataset in zip(axes.ravel(), datasets):
        axis.set_visible(True)
        block = summary[summary["dataset"] == dataset]
        # Uniform of a non-binding cap is the full context. Show that point
        # on both curves when only the class-aware row was stored.
        none_rows = block[block["max_imbalance_ratio"].astype(str) == "none"]
        for strategy, color in colors.items():
            group = block[block["strategy"] == strategy]
            if strategy == "uniform" and not none_rows.empty and not (group["max_imbalance_ratio"].astype(str) == "none").any():
                group = pd.concat([group, none_rows], ignore_index=True)
            if group.empty:
                continue
            group = group.sort_values("time_mean")
            times = group["time_mean"].to_numpy()
            scores = group[mean_col].to_numpy()
            axis.errorbar(
                times,
                scores,
                xerr=group["time_std"].to_numpy(),
                yerr=group[std_col].to_numpy(),
                fmt="o",
                color=color,
                ecolor=color,
                alpha=0.85,
                elinewidth=0.8,
                capsize=2,
                markersize=4,
                label=strategy,
                zorder=3,
            )
            front = group.iloc[np.flatnonzero(_pareto_mask(times, scores, maximize=maximize))]
            axis.plot(
                front.sort_values("time_mean")["time_mean"],
                front.sort_values("time_mean")[mean_col],
                color=color,
                linewidth=1.1,
                zorder=2,
            )
        ratio = float(block["natural_ratio_mean"].iloc[0])
        short_name = dataset.replace("_and_academic_success", "").replace("_prediction", "").replace(
            "_insurance_policies", ""
        )
        axis.set_title(f"{short_name}\nratio {ratio:.0f}×, n={int(block['n_rows'].iloc[0])}", fontsize=9)
        axis.set_xlabel("fit + predict time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(title + "\n5-fold mean ± 1 std", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fit_one(strategy, ratio, X_train, y_train, X_test, n_estimators, seed):
    """Fit one fold. Uniform draws are size-matched to the class-aware cap."""
    import tabicl._sklearn.classifier as classifier_module

    reference_classes = np.unique(np.asarray(y_train))
    common = dict(
        n_estimators=n_estimators,
        norm_methods=["none", "power"] if n_estimators > 1 else ["none"],
        device="cpu",
        use_amp=False,
        use_fa3=False,
        random_state=seed,
        verbose=False,
    )
    n_context = _context_size(y_train, ratio)
    original = classifier_module.majority_undersample_index_sets
    if strategy == "uniform" and ratio is not None and n_context < len(y_train):

        def uniform_sets(y, max_imbalance_ratio, n_sets, random_state):
            y = np.asarray(y)
            n_classes = int(y.max()) + 1
            class_counts = np.bincount(y, minlength=n_classes).astype(np.int64, copy=False)
            class_rows = [np.flatnonzero(y == class_id) for class_id in range(n_classes)]
            rng = np.random.RandomState(random_state)
            index_sets = []
            for _ in range(n_sets):
                chosen = None
                for _attempt in range(10000):
                    candidate = rng.choice(len(y), size=n_context, replace=False)
                    if np.array_equal(np.unique(y[candidate]), np.arange(n_classes)):
                        chosen = candidate
                        break
                if chosen is None:
                    # Keep the requested size and force one row of each class.
                    chosen = rng.choice(len(y), size=n_context, replace=False)
                    present = set(np.unique(y[chosen]).tolist())
                    for class_id in range(n_classes):
                        if class_id in present:
                            continue
                        replace_at = int(np.flatnonzero(y[chosen] != class_id)[0])
                        chosen[replace_at] = int(rng.choice(class_rows[class_id]))
                        present.add(class_id)
                chosen = np.asarray(chosen, dtype=np.int64)
                chosen.sort()
                index_sets.append(chosen)
            # Keep the original prior so the Elkan map is not applied. Uniform
            # draws are not a class-conditional subsample.
            return index_sets, class_counts, class_counts.copy()

        classifier_module.majority_undersample_index_sets = uniform_sets
    try:
        clf = TabICLClassifier(max_imbalance_ratio=ratio, **common)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        proba = clf.predict_proba(X_test)
        predict_seconds = time.perf_counter() - t1
    finally:
        classifier_module.majority_undersample_index_sets = original
    proba = _align_proba(proba, clf.classes_, reference_classes)
    if clf.context_indices_ is None:
        context_rows = len(y_train)
    else:
        context_rows = int(len(clf.context_indices_[0]))
    return clf, proba, fit_seconds, predict_seconds, context_rows, reference_classes


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results_cv.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
    else:
        results = pd.DataFrame()

    ratios = [_parse_ratio(text) for text in args.ratios]
    skipped_path = output_dir / "skipped_cv.csv"
    if skipped_path.exists():
        skipped = pd.read_csv(skipped_path)
    else:
        skipped = pd.DataFrame()
    # Configs whose first measured fold already exceeded the time limit.
    too_slow: set[tuple] = set()
    for name in args.datasets:
        if name not in TABARENA_TASKS:
            known = ", ".join(sorted(TABARENA_TASKS))
            raise SystemExit(f"Unknown dataset {name!r}. Known tasks: {known}")
        print(f"\n=== {name} (OpenML {TABARENA_TASKS[name]}) ===", flush=True)
        X, y = _load_task(name)
        counts = _class_counts(y)
        print(
            f"n={len(y)} p={X.shape[1]} classes={len(counts)} "
            f"counts={sorted(counts.tolist(), reverse=True)} ratio={counts.max() / counts.min():.2f}",
            flush=True,
        )
        n_rows = len(y)
        if args.max_rows is not None and n_rows > args.max_rows:
            X, y = _stratified_take(X, y, args.max_rows, args.seed)
            counts = _class_counts(y)
            print(
                f"stratified cap to n={len(y)} counts={sorted(counts.tolist(), reverse=True)} "
                f"ratio={counts.max() / counts.min():.2f}",
                flush=True,
            )
        if counts.min() < args.n_splits:
            print(
                f"skip: rarest class has {int(counts.min())} rows, fewer than n_splits={args.n_splits}",
                flush=True,
            )
            continue

        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.zeros(len(y)), y))
        for n_estimators in args.n_estimators:
            for ratio in ratios:
                label = _ratio_label(ratio)
                for fold, (train_idx, test_idx) in enumerate(splits):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    n_context = _context_size(y_train, ratio)
                    strategies = ["class_aware"]
                    if ratio is not None and n_context < len(y_train):
                        strategies.append("uniform")
                    for strategy in strategies:
                        row_key = {
                            "dataset": name,
                            "strategy": strategy,
                            "fold": int(fold),
                            "max_imbalance_ratio": label,
                            "n_estimators": int(n_estimators),
                            "n_splits": int(args.n_splits),
                            "max_rows": int(args.max_rows) if args.max_rows is not None else -1,
                            "seed": int(args.seed),
                        }
                        if _already_ran(results, row_key):
                            print(
                                f"  n_est={n_estimators} {strategy} ratio={label} fold={fold}: cached",
                                flush=True,
                            )
                            continue
                        # A cap that does not drop rows is the same fit as every
                        # other non-binding cap: same seed, full context, no
                        # Elkan correction. Reuse the first such fold.
                        if n_context == len(y_train) and not results.empty:
                            donor_mask = (
                                (results["dataset"] == name)
                                & (results["strategy"] == strategy)
                                & (results["fold"] == fold)
                                & (results["n_estimators"] == n_estimators)
                                & (results["n_splits"] == args.n_splits)
                                & (results["max_rows"] == row_key["max_rows"])
                                & (results["seed"] == args.seed)
                                & (results["n_context"] == len(y_train))
                            )
                            donors = results.loc[donor_mask]
                            if not donors.empty:
                                donor = donors.iloc[0]
                                row = donor.to_dict()
                                row["max_imbalance_ratio"] = label
                                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                                results.to_csv(results_path, index=False)
                                print(
                                    f"  n_est={n_estimators} {strategy} ratio={label} fold={fold}: "
                                    f"same context as ratio={donor['max_imbalance_ratio']}",
                                    flush=True,
                                )
                                continue
                        config_key = (name, strategy, int(n_estimators), label)
                        expected = _expected_seconds(results, name, n_estimators, n_context)
                        if config_key in too_slow or (expected is not None and expected > TIME_LIMIT_S):
                            reason = (
                                "earlier fold exceeded the time limit"
                                if config_key in too_slow
                                else f"expected {expected:.0f}s > {TIME_LIMIT_S:.0f}s"
                            )
                            already_skipped = False
                            if not skipped.empty:
                                already_skipped = bool(
                                    (
                                        (skipped["dataset"] == name)
                                        & (skipped["strategy"] == strategy)
                                        & (skipped["fold"] == fold)
                                        & (skipped["max_imbalance_ratio"].astype(str) == label)
                                        & (skipped["n_estimators"] == n_estimators)
                                    ).any()
                                )
                            if not already_skipped:
                                skip_row = {
                                    **row_key,
                                    "n_context": int(n_context),
                                    "expected_seconds": None if expected is None else float(expected),
                                    "reason": reason,
                                }
                                skipped = pd.concat([skipped, pd.DataFrame([skip_row])], ignore_index=True)
                                skipped.to_csv(skipped_path, index=False)
                            print(
                                f"  n_est={n_estimators} {strategy} ratio={label} fold={fold}: "
                                f"skip ({reason})",
                                flush=True,
                            )
                            continue
                        clf, proba, fit_seconds, predict_seconds, context_rows, reference_classes = _fit_one(
                            strategy, ratio, X_train, y_train, X_test, n_estimators, args.seed
                        )
                        auc, loss = _scores(y_test, proba, reference_classes)
                        natural = float(clf.imbalance_ratio_)
                        row = {
                            **row_key,
                            "openml_id": TABARENA_TASKS[name],
                            "n_classes": int(len(reference_classes)),
                            "n_features": int(X.shape[1]),
                            "n_rows": int(len(y)),
                            "n_train": int(len(y_train)),
                            "n_test": int(len(y_test)),
                            "natural_ratio": natural,
                            "n_context": int(context_rows),
                            "fit_seconds": fit_seconds,
                            "predict_seconds": predict_seconds,
                            "fit_predict_seconds": fit_seconds + predict_seconds,
                            "roc_auc": auc,
                            "log_loss": loss,
                        }
                        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                        results.to_csv(results_path, index=False)
                        elapsed = fit_seconds + predict_seconds
                        if elapsed > TIME_LIMIT_S:
                            too_slow.add(config_key)
                        print(
                            f"  n_est={n_estimators} {strategy} ratio={label} fold={fold}: "
                            f"context={context_rows} time={elapsed:.2f}s "
                            f"auc={auc:.4f} logloss={loss:.4f}",
                            flush=True,
                        )
            plot_cv(results, output_dir)
    summary = plot_cv(results, output_dir)
    print(f"\nWrote {results_path}", flush=True)
    if not summary.empty:
        print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "seismic-bumps",
            "anneal",
            "MIC",
            "students_dropout_and_academic_success",
            "polish_companies_bankruptcy",
            "taiwanese_bankruptcy_prediction",
            "coil2000_insurance_policies",
            "APSFailure",
            "kddcup09_appetency",
        ],
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=["1", "3", "5", "10", "15", "20", "30", "none"],
    )
    parser.add_argument("--n-estimators", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=12000,
        help="Stratified cap applied before CV. Datasets at or below this size are used in full.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
