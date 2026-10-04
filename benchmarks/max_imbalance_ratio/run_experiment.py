"""5-fold CV of class-aware vs size-matched uniform subsampling.

Each ensemble member draws its own subsample inside ``TabICLClassifier``.
The class-aware draw is ``max_imbalance_ratio``. The uniform draw passes
the same row count as ``subsample`` and leaves ``max_imbalance_ratio``
unset, so the estimator samples uniformly and skips the Elkan correction.
Datasets are not cut down before ``fit``.

``n_estimators`` in ``{1, 4, 8}`` and caps in
``{1, 3, 5, 10, 15, 20, 30, None}``. Completed folds are appended to
``results_cv.csv`` and skipped on a rerun.

Pending folds are run shortest prediction first. After each batch the
prediction is rebuilt from the folds just measured. A fold is not started
when that prediction is above 60 seconds. Skips are listed in
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
from sklearn.model_selection import StratifiedKFold

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

# Do not start a fold whose predicted fit+predict time is longer than a minute.
TIME_LIMIT_S = 60.0

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


def _global_bias(results: pd.DataFrame) -> float:
    """How much slower n_estimators=4 is than the anchor table. At least 1."""
    if results.empty:
        return 1.0
    ratios = []
    for dataset, anchors in _N4_ANCHORS.items():
        block = results[(results["dataset"] == dataset) & (results["n_estimators"] == 4)]
        if block.empty:
            continue
        for n_context, seconds in block.groupby("n_context")["fit_predict_seconds"].mean().items():
            base = _interp_time(anchors, float(n_context))
            if base > 0:
                ratios.append(float(seconds) / base)
    if not ratios:
        return 1.0
    return float(max(1.0, np.median(ratios)))


def _dataset_bias(results: pd.DataFrame, dataset: str) -> float:
    """n_estimators=4 slowdown on this task, else the slowdown on tasks already run."""
    anchors = _N4_ANCHORS.get(dataset)
    if anchors is None or results.empty:
        return _global_bias(results)
    block = results[(results["dataset"] == dataset) & (results["n_estimators"] == 4)]
    ratios = []
    for n_context, seconds in block.groupby("n_context")["fit_predict_seconds"].mean().items():
        base = _interp_time(anchors, float(n_context))
        if base > 0:
            ratios.append(float(seconds) / base)
    if not ratios:
        return _global_bias(results)
    return float(max(1.0, np.median(ratios)))


def _scale_bias(results: pd.DataFrame, dataset: str, n_estimators: int) -> float:
    """Actual time divided by the anchor at this ensemble size.

    One factor rescales the whole anchor curve. A local slope between two
    nearly identical context lengths is not used: fold noise over a couple
    of rows would otherwise predict multi-minute fits.
    """
    anchors = _N4_ANCHORS.get(dataset)
    scale = _N_EST_SCALE.get(int(n_estimators), (n_estimators / 4) ** 1.07)
    if anchors is None or results.empty:
        return _dataset_bias(results, dataset)
    block = results[(results["dataset"] == dataset) & (results["n_estimators"] == int(n_estimators))]
    ratios = []
    for n_context, seconds in block.groupby("n_context")["fit_predict_seconds"].mean().items():
        base = _interp_time(anchors, float(n_context)) * scale
        if base > 0:
            ratios.append(float(seconds) / base)
    if not ratios:
        return _dataset_bias(results, dataset)
    return float(np.clip(np.median(ratios), 0.5, 3.0))


def _expected_seconds(results: pd.DataFrame, dataset: str, n_estimators: int, n_context: int) -> float | None:
    """Predicted fit+predict seconds from anchors, rescaled by folds already run."""
    anchors = _N4_ANCHORS.get(dataset)
    if anchors is None:
        return None
    scale = _N_EST_SCALE.get(int(n_estimators), (n_estimators / 4) ** 1.07)
    predicted = _interp_time(anchors, float(n_context)) * scale * _scale_bias(results, dataset, n_estimators)
    if results.empty:
        return float(predicted)
    block = results[(results["dataset"] == dataset) & (results["n_estimators"] == int(n_estimators))]
    if block.empty:
        return float(predicted)
    # A fold that already exceeded the limit condemns this context and anything larger.
    slow = block[block["fit_predict_seconds"] > TIME_LIMIT_S]
    if not slow.empty and float(n_context) >= float(slow["n_context"].min()) * 0.98:
        return float(max(slow["fit_predict_seconds"].max(), predicted))
    means = block.groupby("n_context")["fit_predict_seconds"].mean()
    nearest = min(means.index, key=lambda ctx: abs(float(ctx) - n_context))
    if abs(float(nearest) - n_context) <= max(10.0, 0.05 * float(n_context)):
        return float(means.loc[nearest])
    return float(predicted)


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


def _configuration_label(row) -> str:
    """Short label: ensemble size, then the cap or a uniform subsample of that size."""
    members = int(row["n_estimators"])
    ratio = str(row["max_imbalance_ratio"])
    natural = float(row["natural_ratio_mean"])
    full = ratio == "none" or float(ratio) + 1e-9 >= natural
    tag = "full" if full else ratio
    if row["strategy"] == "uniform" and not full:
        tag = "u" + tag
    return f"{members}·{tag}"


def _dedupe_configs(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep one row when several caps are the same copied full-context fit."""
    ordered = summary.copy()

    def _cap_order(value: object) -> float:
        text = str(value)
        if text == "none":
            return 1e9
        return float(text)

    ordered["_order"] = ordered["max_imbalance_ratio"].map(_cap_order)
    ordered = ordered.sort_values("_order")
    identity = [
        "dataset",
        "strategy",
        "n_estimators",
        "time_mean",
        "roc_auc_mean",
        "log_loss_mean",
        "n_context_mean",
    ]
    return ordered.drop_duplicates(identity, keep="first").drop(columns="_order")


def _joint_pareto_figure(summary: pd.DataFrame, score: str, *, maximize: bool, title: str, ylabel: str, path: Path) -> pd.DataFrame:
    """Pareto front over caps, uniform subsamples, and ensemble sizes together."""
    configs = _dedupe_configs(summary)
    configs = configs.copy()
    configs["on_front"] = False
    configs["label"] = configs.apply(_configuration_label, axis=1)
    mean_col = f"{score}_mean"
    for dataset, index in configs.groupby("dataset").groups.items():
        block = configs.loc[index]
        mask = _pareto_mask(block["time_mean"].to_numpy(), block[mean_col].to_numpy(), maximize=maximize)
        configs.loc[block.index[mask], "on_front"] = True

    datasets = [name for name in DATASET_ORDER if name in set(configs["dataset"])]
    if not datasets:
        return configs
    n_cols = 3
    n_rows = int(np.ceil(len(datasets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.0 * n_rows), squeeze=False)
    for axis in axes.ravel():
        axis.set_visible(False)
    markers = {1: "o", 4: "s", 8: "D"}
    colors = {"class_aware": "#1f4e79", "uniform": "#c45c26"}
    for axis, dataset in zip(axes.ravel(), datasets):
        axis.set_visible(True)
        block = configs[configs["dataset"] == dataset]
        for strategy, color in colors.items():
            for n_estimators, marker in markers.items():
                group = block[(block["strategy"] == strategy) & (block["n_estimators"] == n_estimators)]
                if group.empty:
                    continue
                kind = "class-aware" if strategy == "class_aware" else "uniform subsample"
                axis.scatter(
                    group["time_mean"],
                    group[mean_col],
                    s=22,
                    marker=marker,
                    color=color,
                    alpha=0.28,
                    linewidths=0,
                    zorder=2,
                )
        front = block[block["on_front"]].sort_values("time_mean")
        if not front.empty:
            axis.plot(front["time_mean"], front[mean_col], color="#222222", linewidth=1.0, zorder=3)
            for _, row in front.iterrows():
                axis.scatter(
                    [row["time_mean"]],
                    [row[mean_col]],
                    s=46,
                    marker=markers[int(row["n_estimators"])],
                    facecolor=colors[row["strategy"]],
                    edgecolor="#111111",
                    linewidths=0.6,
                    zorder=4,
                )
                axis.annotate(
                    row["label"],
                    (row["time_mean"], row[mean_col]),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=7,
                )
        ratio = float(block["natural_ratio_mean"].iloc[0])
        short_name = dataset.replace("_and_academic_success", "").replace("_prediction", "").replace(
            "_insurance_policies", ""
        )
        axis.set_title(f"{short_name}\nratio {ratio:.0f}×, n={int(block['n_rows'].iloc[0])}", fontsize=9)
        axis.set_xlabel("fit + predict time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    legend_handles = []
    for strategy, color in colors.items():
        kind = "class-aware" if strategy == "class_aware" else "uniform subsample"
        for n_estimators, marker in markers.items():
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="none",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=6,
                    label=f"{kind}, {n_estimators} members",
                )
            )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(title + "\nlabels are ensemble size · cap (u = uniform subsample of that size)", fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return configs


def plot_cv(results: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = summarize(results)
    if summary.empty:
        return summary
    summary = summary.loc[summary["n_folds"] >= 5].copy()
    summary.to_csv(output_dir / "results_cv_summary.csv", index=False)
    roc_front = _joint_pareto_figure(
        summary,
        "roc_auc",
        maximize=True,
        title="Joint Pareto front: ROC-AUC vs fit+predict time",
        ylabel="ROC-AUC",
        path=output_dir / "pareto_joint_roc_auc.png",
    )
    loss_front = _joint_pareto_figure(
        summary,
        "log_loss",
        maximize=False,
        title="Joint Pareto front: log-loss vs fit+predict time",
        ylabel="log-loss",
        path=output_dir / "pareto_joint_log_loss.png",
    )
    roc_front = roc_front.rename(columns={"on_front": "on_roc_auc_front"}).drop(columns=["label"])
    loss_kept = loss_front[["dataset", "strategy", "n_estimators", "max_imbalance_ratio", "on_front"]]
    joint = roc_front.merge(
        loss_kept.rename(columns={"on_front": "on_log_loss_front"}),
        on=["dataset", "strategy", "n_estimators", "max_imbalance_ratio"],
        how="left",
    )
    joint.to_csv(output_dir / "pareto_joint.csv", index=False)
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
    """Fit one fold. Uniform draws use ``subsample`` at the class-aware size."""
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
    if strategy == "uniform":
        clf = TabICLClassifier(subsample=n_context, max_imbalance_ratio=None, **common)
    else:
        clf = TabICLClassifier(max_imbalance_ratio=ratio, **common)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = clf.predict_proba(X_test)
    predict_seconds = time.perf_counter() - t1
    proba = _align_proba(proba, clf.classes_, reference_classes)
    if clf.context_indices_ is None:
        context_rows = len(y_train)
    else:
        context_rows = int(len(clf.context_indices_[0]))
    return clf, proba, fit_seconds, predict_seconds, context_rows, reference_classes


def _dataset_is_finished(results: pd.DataFrame, name: str, n_estimators: list[int], n_splits: int) -> bool:
    """True when every ensemble size already has an uncapped class-aware run."""
    if results.empty:
        return False
    block = results[
        (results["dataset"] == name)
        & (results["strategy"] == "class_aware")
        & (results["max_imbalance_ratio"].astype(str) == "none")
    ]
    return all(int((block["n_estimators"] == n_est).sum()) >= n_splits for n_est in n_estimators)


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
    tables: dict[str, tuple] = {}
    pending: list[dict] = []
    def _remember_skip(job: dict, reason: str, expected: float | None) -> None:
        nonlocal skipped
        if not skipped.empty:
            already = (
                (skipped["dataset"] == job["dataset"])
                & (skipped["strategy"] == job["strategy"])
                & (skipped["fold"] == job["fold"])
                & (skipped["max_imbalance_ratio"].astype(str) == job["label"])
                & (skipped["n_estimators"] == job["n_estimators"])
            )
            if bool(already.any()):
                return
        skip_row = {
            **job["row_key"],
            "n_context": int(job["n_context"]),
            "expected_seconds": None if expected is None else float(expected),
            "reason": reason,
        }
        skipped = pd.concat([skipped, pd.DataFrame([skip_row])], ignore_index=True)

    def _copy_donor(job: dict) -> bool:
        """Reuse a full-context fold. Returns True when the job is finished."""
        nonlocal results
        if job["n_context"] != job["n_train"] or results.empty:
            return False
        donor_mask = (
            (results["dataset"] == job["dataset"])
            & (results["strategy"] == job["strategy"])
            & (results["fold"] == job["fold"])
            & (results["n_estimators"] == job["n_estimators"])
            & (results["n_splits"] == args.n_splits)
            & (results["max_rows"] == job["row_key"]["max_rows"])
            & (results["seed"] == args.seed)
            & (results["n_context"] == job["n_train"])
        )
        donors = results.loc[donor_mask]
        if donors.empty:
            return False
        row = donors.iloc[0].to_dict()
        row["max_imbalance_ratio"] = job["label"]
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(results_path, index=False)
        print(
            f"  n_est={job['n_estimators']} {job['strategy']} ratio={job['label']} "
            f"fold={job['fold']}: same context as ratio={donors.iloc[0]['max_imbalance_ratio']}",
            flush=True,
        )
        return True

    for name in args.datasets:
        if name not in TABARENA_TASKS:
            known = ", ".join(sorted(TABARENA_TASKS))
            raise SystemExit(f"Unknown dataset {name!r}. Known tasks: {known}")
        if _dataset_is_finished(results, name, list(args.n_estimators), args.n_splits):
            print(f"\n=== {name}: already finished ===", flush=True)
            continue
        print(f"\n=== {name} (OpenML {TABARENA_TASKS[name]}) ===", flush=True)
        X, y = _load_task(name)
        counts = _class_counts(y)
        print(
            f"n={len(y)} p={X.shape[1]} classes={len(counts)} "
            f"counts={sorted(counts.tolist(), reverse=True)} ratio={counts.max() / counts.min():.2f}",
            flush=True,
        )
        if counts.min() < args.n_splits:
            print(
                f"skip: rarest class has {int(counts.min())} rows, fewer than n_splits={args.n_splits}",
                flush=True,
            )
            continue
        tables[name] = (X, y)
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.zeros(len(y)), y))
        for n_estimators in args.n_estimators:
            for ratio in ratios:
                label = _ratio_label(ratio)
                for fold, (train_idx, test_idx) in enumerate(splits):
                    y_train = y.iloc[train_idx]
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
                            "max_rows": -1,
                            "seed": int(args.seed),
                        }
                        if _already_ran(results, row_key):
                            continue
                        pending.append(
                            {
                                "dataset": name,
                                "strategy": strategy,
                                "fold": int(fold),
                                "label": label,
                                "ratio": ratio,
                                "n_estimators": int(n_estimators),
                                "n_context": int(n_context),
                                "n_train": int(len(y_train)),
                                "train_idx": train_idx,
                                "test_idx": test_idx,
                                "row_key": row_key,
                            }
                        )

    # Identical full-context caps share one fit. Copy those before spending time.
    still_pending = []
    for job in pending:
        if _copy_donor(job):
            continue
        still_pending.append(job)
    pending = still_pending
    print(
        f"\n{len(pending)} folds still to run, shortest prediction first, "
        f"limit {TIME_LIMIT_S:.0f}s",
        flush=True,
    )

    while pending:
        for job in pending:
            job["expected"] = _expected_seconds(
                results, job["dataset"], job["n_estimators"], job["n_context"]
            )
        pending.sort(
            key=lambda job: (
                job["expected"] if job["expected"] is not None else 1e9,
                job["n_context"],
                job["n_estimators"],
                job["dataset"],
                job["fold"],
            )
        )
        if pending[0]["expected"] is None or pending[0]["expected"] > TIME_LIMIT_S:
            for job in pending:
                _remember_skip(
                    job,
                    f"expected {job['expected']:.0f}s > {TIME_LIMIT_S:.0f}s"
                    if job["expected"] is not None
                    else "no timing anchor",
                    job["expected"],
                )
            print(
                f"skip remaining {len(pending)} folds; "
                f"shortest expectation is {pending[0]['expected']:.1f}s",
                flush=True,
            )
            skipped.to_csv(skipped_path, index=False)
            pending = []
            break

        fastest = float(pending[0]["expected"])
        # The next batch stays close to the fastest remaining fold so its
        # measured time can revise the estimate before a longer fold starts.
        ceiling = min(TIME_LIMIT_S, max(fastest * 1.25, fastest + 1.5))
        batch = [job for job in pending if job["expected"] is not None and job["expected"] <= ceiling]
        batch_ids = {id(job) for job in batch}
        pending = [job for job in pending if id(job) not in batch_ids]
        print(
            f"\nbatch: {len(batch)} folds, expected {fastest:.1f}–{ceiling:.1f}s",
            flush=True,
        )
        for job in batch:
            if _copy_donor(job):
                continue
            expected = _expected_seconds(results, job["dataset"], job["n_estimators"], job["n_context"])
            if expected is None or expected > TIME_LIMIT_S:
                reason = (
                    "no timing anchor"
                    if expected is None
                    else f"expected {expected:.0f}s > {TIME_LIMIT_S:.0f}s"
                )
                _remember_skip(job, reason, expected)
                print(
                    f"  {job['dataset']} n_est={job['n_estimators']} {job['strategy']} "
                    f"ratio={job['label']} fold={job['fold']}: skip ({reason})",
                    flush=True,
                )
                continue
            X, y = tables[job["dataset"]]
            X_train, X_test = X.iloc[job["train_idx"]], X.iloc[job["test_idx"]]
            y_train, y_test = y.iloc[job["train_idx"]], y.iloc[job["test_idx"]]
            clf, proba, fit_seconds, predict_seconds, context_rows, reference_classes = _fit_one(
                job["strategy"],
                job["ratio"],
                X_train,
                y_train,
                X_test,
                job["n_estimators"],
                args.seed,
            )
            auc, loss = _scores(y_test, proba, reference_classes)
            elapsed = fit_seconds + predict_seconds
            row = {
                **job["row_key"],
                "openml_id": TABARENA_TASKS[job["dataset"]],
                "n_classes": int(len(reference_classes)),
                "n_features": int(X.shape[1]),
                "n_rows": int(len(y)),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "natural_ratio": float(clf.imbalance_ratio_),
                "n_context": int(context_rows),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "fit_predict_seconds": elapsed,
                "roc_auc": auc,
                "log_loss": loss,
            }
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            results.to_csv(results_path, index=False)
            print(
                f"  {job['dataset']} n_est={job['n_estimators']} {job['strategy']} "
                f"ratio={job['label']} fold={job['fold']}: context={context_rows} "
                f"time={elapsed:.2f}s expected={expected:.1f}s auc={auc:.4f} logloss={loss:.4f}",
                flush=True,
            )
        if not skipped.empty:
            skipped.to_csv(skipped_path, index=False)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
