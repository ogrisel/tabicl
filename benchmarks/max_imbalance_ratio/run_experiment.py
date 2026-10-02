"""ROC-AUC versus fit+predict time for ``max_imbalance_ratio``.

Evaluates :class:`tabicl.TabICLClassifier` on highly imbalanced TabArena
classification tasks (OpenML suite 457, ``tabarena-v0.1``). Each task is
split once. Training rows are then stratified-subsampled, and several
imbalance caps are timed on that same train/test pair.

Binary ROC-AUC is invariant to the Elkan correction itself (it is a strictly
increasing function of the positive-class probability). Differences across
caps on binary tasks therefore come from the shorter in-context set. On
multiclass tasks the correction's normalizer depends on the features, so
one-vs-rest AUC can move as well. Log-loss is recorded because that is the
metric the prior correction is meant to protect.

Examples
--------
Pilot, one small task::

    python benchmarks/max_imbalance_ratio/run_experiment.py \\
        --datasets seismic-bumps \\
        --train-sizes 400 \\
        --ratios none 5 1 \\
        --n-estimators 1

Grow the grid by repeating the command with more datasets, sizes, or caps.
Existing rows in ``results.csv`` are kept and matching runs are skipped.
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
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from tabicl import TabICLClassifier

warnings.filterwarnings(
    "ignore",
    message="The following categorical columns have a cardinality above 40",
)


# TabArena-v0.1 classification tasks that are small enough to download
# quickly and, on the curated release, carry a heavy class skew.
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


def _parse_ratio(text: str):
    if text.lower() in {"none", "inf", "unlimited"}:
        return None
    return float(text)


def _ratio_label(value) -> str:
    if value is None or (isinstance(value, float) and np.isinf(value)):
        return "none"
    return f"{float(value):g}"


def _stratified_take(X, y, n_samples: int, random_state: int):
    if n_samples >= len(y):
        return X, y
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
    index, _ = next(splitter.split(np.zeros(len(y)), y))
    if isinstance(X, pd.DataFrame):
        return X.iloc[index], y.iloc[index] if isinstance(y, pd.Series) else y[index]
    return X[index], y[index]


def _class_counts(y) -> np.ndarray:
    _, counts = np.unique(np.asarray(y), return_counts=True)
    return counts.astype(np.int64)


def _scores(y_true, proba, classes) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    if len(classes) == 2:
        # Fixed positive label (classes_[1], the greater encoded label) so
        # every cap is scored the same way. Binary AUC does not depend on
        # which label is called positive up to complement, and the complement
        # is identical for every cap.
        auc = float(roc_auc_score(y_true == classes[1], proba[:, 1]))
    else:
        auc = float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=classes)
        )
    loss = float(log_loss(y_true, proba, labels=classes))
    return auc, loss


def _load_task(name: str):
    data_id = TABARENA_TASKS[name]
    frame, y = fetch_openml(data_id=data_id, as_frame=True, return_X_y=True, parser="auto")
    y = pd.Series(np.asarray(y), name="target")
    return frame.reset_index(drop=True), y.reset_index(drop=True)


def _already_ran(results: pd.DataFrame, row: dict) -> bool:
    if results.empty:
        return False
    keys = ["dataset", "train_size", "max_imbalance_ratio", "n_estimators", "seed"]
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


def _series_pareto(axis, group: pd.DataFrame, score_column: str, *, maximize: bool) -> None:
    """Scatter one ratio sweep and draw its Pareto front."""
    group = group.sort_values("fit_predict_seconds")
    times = group["fit_predict_seconds"].to_numpy()
    score = group[score_column].to_numpy()
    axis.scatter(times, score, s=42, color="#1f4e79", zorder=3)
    for _, row in group.iterrows():
        axis.annotate(
            str(row["max_imbalance_ratio"]),
            (row["fit_predict_seconds"], row[score_column]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    front = group.iloc[np.flatnonzero(_pareto_mask(times, score, maximize=maximize))]
    front = front.sort_values("fit_predict_seconds")
    axis.plot(
        front["fit_predict_seconds"],
        front[score_column],
        color="#c45c26",
        linewidth=1.4,
        zorder=2,
    )
    axis.grid(True, alpha=0.3)


def _largest_slice(results: pd.DataFrame, n_estimators: int) -> pd.DataFrame:
    subset = results[results["n_estimators"] == n_estimators]
    if subset.empty:
        return subset
    max_size = subset.groupby("dataset")["train_size"].transform("max")
    return subset[subset["train_size"] == max_size]


def _panel_figure(results: pd.DataFrame, datasets: list[str], score_column: str, *, maximize: bool, title: str, ylabel: str, path: Path) -> None:
    present = [name for name in datasets if name in set(results["dataset"])]
    if not present:
        return
    n_cols = 4 if len(present) > 4 else len(present)
    n_rows = int(np.ceil(len(present) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.5 * n_rows), squeeze=False)
    for axis in axes.ravel():
        axis.set_visible(False)
    for axis, dataset in zip(axes.ravel(), present):
        axis.set_visible(True)
        group = results[results["dataset"] == dataset]
        _series_pareto(axis, group, score_column, maximize=maximize)
        n_train = int(group["train_size"].iloc[0])
        ratio = float(group["natural_ratio"].iloc[0])
        n_classes = int(group["n_classes"].iloc[0])
        axis.set_title(f"{dataset}\n{n_classes} classes, ratio {ratio:.0f}×, n={n_train}", fontsize=9)
        axis.set_xlabel("fit + predict time (s)")
        axis.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot(results: pd.DataFrame, output_dir: Path) -> None:
    if results.empty:
        return
    dataset_order = [
        "APSFailure",
        "kddcup09_appetency",
        "taiwanese_bankruptcy_prediction",
        "coil2000_insurance_policies",
        "polish_companies_bankruptcy",
        "seismic-bumps",
        "anneal",
        "MIC",
    ]
    largest = _largest_slice(results, n_estimators=1)
    _panel_figure(
        largest,
        dataset_order,
        "roc_auc",
        maximize=True,
        title="Largest training slice, 1 estimator: ROC-AUC vs fit+predict time",
        ylabel="ROC-AUC",
        path=output_dir / "pareto_roc_auc.png",
    )
    _panel_figure(
        largest,
        dataset_order,
        "log_loss",
        maximize=False,
        title="Largest training slice, 1 estimator: log-loss vs fit+predict time",
        ylabel="log-loss",
        path=output_dir / "pareto_log_loss.png",
    )
    ensemble = _largest_slice(results, n_estimators=4)
    _panel_figure(
        ensemble,
        dataset_order,
        "roc_auc",
        maximize=True,
        title="4 estimators: ROC-AUC vs fit+predict time",
        ylabel="ROC-AUC",
        path=output_dir / "pareto_roc_auc_ensemble.png",
    )
    _panel_figure(
        ensemble,
        dataset_order,
        "log_loss",
        maximize=False,
        title="4 estimators: log-loss vs fit+predict time",
        ylabel="log-loss",
        path=output_dir / "pareto_log_loss_ensemble.png",
    )


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
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
        natural_ratio = float(counts.max() / counts.min())
        print(
            f"n={len(y)} p={X.shape[1]} classes={len(counts)} "
            f"counts={counts.tolist()} ratio={natural_ratio:.2f}",
            flush=True,
        )
        if natural_ratio < args.min_ratio:
            print(f"skip: natural ratio {natural_ratio:.2f} < {args.min_ratio}", flush=True)
            continue

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.seed
        )
        # Keep prediction cheap and stable: a stratified test slice, shared
        # by every cap and every training size.
        if args.max_test is not None and len(y_test) > args.max_test:
            X_test, y_test = _stratified_take(X_test, y_test, args.max_test, args.seed)

        for train_size in args.train_sizes:
            try:
                X_train, y_train = _stratified_take(X_train_full, y_train_full, train_size, args.seed)
            except ValueError as exc:
                print(f"skip n_train={train_size}: {exc}", flush=True)
                continue
            train_counts = _class_counts(y_train)
            print(
                f"train={len(y_train)} counts={train_counts.tolist()} "
                f"ratio={train_counts.max() / train_counts.min():.2f} test={len(y_test)}",
                flush=True,
            )
            for ratio in ratios:
                row_key = {
                    "dataset": name,
                    "train_size": int(len(y_train)),
                    "max_imbalance_ratio": _ratio_label(ratio),
                    "n_estimators": int(args.n_estimators),
                    "seed": int(args.seed),
                }
                if _already_ran(results, row_key):
                    print(f"  ratio={row_key['max_imbalance_ratio']}: cached", flush=True)
                    continue
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
                    "n_features": int(X_train.shape[1]),
                    "n_test": int(len(y_test)),
                    "natural_ratio": float(clf.imbalance_ratio_),
                    "n_context": int(clf.context_class_counts_.sum()),
                    "context_ratio": float(clf.context_class_counts_.max() / clf.context_class_counts_.min()),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "fit_predict_seconds": fit_seconds + predict_seconds,
                    "roc_auc": auc,
                    "log_loss": loss,
                }
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                results.to_csv(results_path, index=False)
                _plot(results, output_dir)
                print(
                    f"  ratio={row['max_imbalance_ratio']}: context={row['n_context']} "
                    f"({row['context_ratio']:.2f}x) "
                    f"time={row['fit_predict_seconds']:.2f}s "
                    f"auc={row['roc_auc']:.4f} logloss={row['log_loss']:.4f}",
                    flush=True,
                )
    _plot(results, output_dir)
    print(f"\nWrote {results_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["seismic-bumps"])
    parser.add_argument("--train-sizes", nargs="+", type=int, default=[400])
    parser.add_argument("--ratios", nargs="+", default=["none", "10", "5", "2", "1"])
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-test", type=int, default=800)
    parser.add_argument("--min-ratio", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
