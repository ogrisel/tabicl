"""5-fold CV of ``max_imbalance_ratio`` on imbalanced TabArena tasks.

Each task is optionally stratified-capped (``--max-rows``) and then split with
``StratifiedKFold``. Every cap is fit on the same training folds. The plotted
points are means across folds. Horizontal error bars are the standard
deviation of fit+predict time, and vertical error bars are the standard
deviation of the metric.

Examples
--------
::

    python benchmarks/max_imbalance_ratio/run_experiment.py \\
        --datasets seismic-bumps anneal MIC \\
        --ratios 1 5 20 none \\
        --n-estimators 4 \\
        --n-splits 5

Completed folds are appended to ``results_cv.csv`` and skipped on a rerun.
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
    keys = ["dataset", "fold", "max_imbalance_ratio", "n_estimators", "n_splits", "max_rows", "seed"]
    mask = np.ones(len(results), dtype=bool)
    for key in keys:
        mask &= results[key].astype(str) == str(row[key])
    return bool(mask.any())


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
    grouped = results.groupby(["dataset", "max_imbalance_ratio"], sort=False)
    rows = []
    for (dataset, ratio), group in grouped:
        rows.append(
            {
                "dataset": dataset,
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
    # Only draw caps that have every requested fold, so a partial rerun does
    # not mix a 2-fold mean into a 5-fold panel. The caller passes the full
    # frame; incomplete groups are those with fewer folds than the mode.
    full = summary["n_folds"] == summary["n_folds"].max()
    summary = summary.loc[full].copy()
    _panel_figure(
        summary,
        "roc_auc",
        maximize=True,
        title="5-fold CV mean ± 1 std: ROC-AUC vs fit+predict time",
        ylabel="ROC-AUC",
        path=output_dir / "pareto_roc_auc.png",
    )
    _panel_figure(
        summary,
        "log_loss",
        maximize=False,
        title="5-fold CV mean ± 1 std: log-loss vs fit+predict time",
        ylabel="log-loss",
        path=output_dir / "pareto_log_loss.png",
    )
    summary.to_csv(output_dir / "results_cv_summary.csv", index=False)
    return summary


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results_cv.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
    else:
        results = pd.DataFrame()

    ratios = [_parse_ratio(text) for text in args.ratios]
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
        for ratio in ratios:
            label = _ratio_label(ratio)
            for fold, (train_idx, test_idx) in enumerate(splits):
                row_key = {
                    "dataset": name,
                    "fold": int(fold),
                    "max_imbalance_ratio": label,
                    "n_estimators": int(args.n_estimators),
                    "n_splits": int(args.n_splits),
                    "max_rows": int(args.max_rows) if args.max_rows is not None else -1,
                    "seed": int(args.seed),
                }
                if _already_ran(results, row_key):
                    print(f"  ratio={label} fold={fold}: cached", flush=True)
                    continue
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                clf = TabICLClassifier(
                    n_estimators=args.n_estimators,
                    norm_methods=["none", "power"] if args.n_estimators > 1 else ["none"],
                    max_imbalance_ratio=ratio,
                    device="cpu",
                    use_amp=False,
                    use_fa3=False,
                    random_state=args.seed,
                    verbose=False,
                )
                t0 = time.perf_counter()
                clf.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - t0
                t1 = time.perf_counter()
                proba = clf.predict_proba(X_test)
                predict_seconds = time.perf_counter() - t1
                auc, loss = _scores(y_test, proba, clf.classes_)
                row = {
                    **row_key,
                    "openml_id": TABARENA_TASKS[name],
                    "n_classes": int(len(clf.classes_)),
                    "n_features": int(X.shape[1]),
                    "n_rows": int(len(y)),
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "natural_ratio": float(clf.imbalance_ratio_),
                    "n_context": int(clf.context_class_counts_.sum()),
                    "context_ratio": float(
                        clf.context_class_counts_.max() / clf.context_class_counts_.min()
                    ),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "fit_predict_seconds": fit_seconds + predict_seconds,
                    "roc_auc": auc,
                    "log_loss": loss,
                }
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                results.to_csv(results_path, index=False)
                print(
                    f"  ratio={label} fold={fold}: context={row['n_context']} "
                    f"({row['context_ratio']:.2f}x) time={row['fit_predict_seconds']:.2f}s "
                    f"auc={row['roc_auc']:.4f} logloss={row['log_loss']:.4f}",
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
        default=["1", "5", "20", "none"],
    )
    parser.add_argument("--n-estimators", type=int, default=4)
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
