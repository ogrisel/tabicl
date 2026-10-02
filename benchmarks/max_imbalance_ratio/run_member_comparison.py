"""Shared vs per-member majority undersampling, 5-fold CV.

The library draws an independent majority subsample for every ensemble
member and applies the Elkan correction once after ensembling. This script
compares that with the previous behavior: one shared subsample, then the
usual feature/class/normalization ensemble, then the same correction.

``n_estimators`` must be greater than 1. With one member the two strategies
are the same draw.

``None`` is run once; it does not undersample.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as base  # noqa: E402

from tabicl import TabICLClassifier
from tabicl._sklearn.prevalence import correct_class_prior, majority_undersample_indices

warnings.filterwarnings(
    "ignore",
    message="The following categorical columns have a cardinality above 40",
)


def _fit_predict(strategy, ratio, X_train, y_train, X_test, n_estimators, seed):
    common = dict(
        n_estimators=n_estimators,
        device="cpu",
        use_amp=False,
        use_fa3=False,
        random_state=seed,
        verbose=False,
    )
    if ratio is None:
        clf = TabICLClassifier(max_imbalance_ratio=None, **common)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        proba = clf.predict_proba(X_test)
        predict_seconds = time.perf_counter() - t1
        return proba, clf, fit_seconds, predict_seconds, len(y_train), float("nan")

    if strategy == "independent":
        clf = TabICLClassifier(max_imbalance_ratio=ratio, **common)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        proba = clf.predict_proba(X_test)
        predict_seconds = time.perf_counter() - t1
        n_context = int(np.mean([len(idx) for idx in clf.context_indices_])) if clf.context_indices_ else len(y_train)
        return proba, clf, fit_seconds, predict_seconds, n_context, float(clf.imbalance_ratio_)

    y_enc = LabelEncoder().fit_transform(np.asarray(y_train))
    indices, counts, context_counts = majority_undersample_indices(y_enc, ratio, seed)
    clf = TabICLClassifier(max_imbalance_ratio=None, **common)
    t0 = time.perf_counter()
    if indices is None:
        clf.fit(X_train, y_train)
        n_context = len(y_train)
    else:
        clf.fit(X_train.iloc[indices], y_train.iloc[indices])
        n_context = len(indices)
    fit_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = clf.predict_proba(X_test)
    if indices is not None:
        proba = correct_class_prior(proba, context_counts, counts)
    predict_seconds = time.perf_counter() - t1
    natural = float(counts.max() / counts.min())
    return proba, clf, fit_seconds, predict_seconds, n_context, natural


def _plot(summary: pd.DataFrame, output_dir: Path) -> None:
    datasets = [name for name in base.DATASET_ORDER if name in set(summary["dataset"])]
    if not datasets:
        return
    n_cols = 3
    n_rows = int(np.ceil(len(datasets) / n_cols))
    for score, ylabel, maximize, filename in (
        ("roc_auc", "ROC-AUC", True, "pareto_member_roc_auc.png"),
        ("log_loss", "log-loss", False, "pareto_member_log_loss.png"),
    ):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.6 * n_rows), squeeze=False)
        for axis in axes.ravel():
            axis.set_visible(False)
        for axis, dataset in zip(axes.ravel(), datasets):
            axis.set_visible(True)
            for strategy, color in (("shared", "#1f4e79"), ("independent", "#c45c26")):
                group = summary[(summary["dataset"] == dataset) & (summary["strategy"] == strategy)]
                if group.empty:
                    continue
                group = group.sort_values("time_mean")
                axis.errorbar(
                    group["time_mean"],
                    group[f"{score}_mean"],
                    xerr=group["time_std"],
                    yerr=group[f"{score}_std"],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    alpha=0.9,
                    capsize=3,
                    label=strategy,
                )
                for _, row in group.iterrows():
                    axis.annotate(
                        str(row["max_imbalance_ratio"]),
                        (row["time_mean"], row[f"{score}_mean"]),
                        textcoords="offset points",
                        xytext=(4, 3),
                        fontsize=7,
                        color=color,
                    )
            ratio = float(summary.loc[summary["dataset"] == dataset, "natural_ratio_mean"].iloc[0])
            n_rows_used = int(summary.loc[summary["dataset"] == dataset, "n_rows"].iloc[0])
            short = dataset.replace("_and_academic_success", "").replace("_prediction", "").replace(
                "_insurance_policies", ""
            )
            axis.set_title(f"{short}\nratio {ratio:.0f}×, n={n_rows_used}", fontsize=9)
            axis.set_xlabel("fit + predict time (s)")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle(
            f"5-fold mean ± 1 std, n_estimators={int(summary['n_estimators'].iloc[0])}: {ylabel}",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=140)
        plt.close(fig)
        del maximize


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    results_path = output_dir / "results_member_cv.csv"
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    datasets = [
        "seismic-bumps",
        "anneal",
        "MIC",
        "students_dropout_and_academic_success",
        "polish_companies_bankruptcy",
        "taiwanese_bankruptcy_prediction",
        "coil2000_insurance_policies",
        "APSFailure",
        "kddcup09_appetency",
    ]
    ratios = [1.0, 5.0, 20.0, None]
    n_estimators = 4
    n_splits = 5
    max_rows = 12000
    seed = 0

    def done(dataset, fold, ratio_label, strategy) -> bool:
        if results.empty:
            return False
        mask = (
            (results["dataset"] == dataset)
            & (results["fold"].astype(int) == fold)
            & (results["max_imbalance_ratio"].astype(str) == ratio_label)
            & (results["strategy"] == strategy)
            & (results["n_estimators"].astype(int) == n_estimators)
        )
        return bool(mask.any())

    for name in datasets:
        print(f"\n=== {name} ===", flush=True)
        X, y = base._load_task(name)
        counts = base._class_counts(y)
        print(
            f"n={len(y)} p={X.shape[1]} counts={sorted(counts.tolist(), reverse=True)} "
            f"ratio={counts.max() / counts.min():.2f}",
            flush=True,
        )
        if len(y) > max_rows:
            X, y = base._stratified_take(X, y, max_rows, seed)
            counts = base._class_counts(y)
            print(f"capped to n={len(y)} ratio={counts.max() / counts.min():.2f}", flush=True)
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
        for ratio in ratios:
            label = base._ratio_label(ratio)
            strategies = ["independent"] if ratio is None else ["shared", "independent"]
            for strategy in strategies:
                for fold, (train_idx, test_idx) in enumerate(splits):
                    if done(name, fold, label, strategy):
                        print(f"  {strategy} ratio={label} fold={fold}: cached", flush=True)
                        continue
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    proba, clf, fit_s, pred_s, n_context, natural = _fit_predict(
                        strategy, ratio, X_train, y_train, X_test, n_estimators, seed
                    )
                    auc, loss = base._scores(y_test, proba, clf.classes_)
                    row = {
                        "dataset": name,
                        "strategy": strategy,
                        "fold": fold,
                        "max_imbalance_ratio": label,
                        "n_estimators": n_estimators,
                        "n_rows": len(y),
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                        "n_classes": int(len(clf.classes_)),
                        "natural_ratio": natural if np.isfinite(natural) else float(clf.imbalance_ratio_),
                        "n_context": n_context,
                        "fit_seconds": fit_s,
                        "predict_seconds": pred_s,
                        "fit_predict_seconds": fit_s + pred_s,
                        "roc_auc": auc,
                        "log_loss": loss,
                    }
                    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                    results.to_csv(results_path, index=False)
                    print(
                        f"  {strategy} ratio={label} fold={fold}: context={n_context} "
                        f"time={fit_s + pred_s:.2f}s auc={auc:.4f} ll={loss:.4f}",
                        flush=True,
                    )
        _summarize_and_plot(results, output_dir)
    _summarize_and_plot(results, output_dir)
    print(f"\nWrote {results_path}", flush=True)


def _summarize_and_plot(results: pd.DataFrame, output_dir: Path) -> None:
    if results.empty:
        return
    rows = []
    for keys, group in results.groupby(["dataset", "strategy", "max_imbalance_ratio"], sort=False):
        dataset, strategy, ratio = keys
        if len(group) < 2:
            continue
        rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "max_imbalance_ratio": ratio,
                "n_estimators": int(group["n_estimators"].iloc[0]),
                "n_folds": len(group),
                "n_rows": int(group["n_rows"].iloc[0]),
                "natural_ratio_mean": float(group["natural_ratio"].mean()),
                "time_mean": float(group["fit_predict_seconds"].mean()),
                "time_std": float(group["fit_predict_seconds"].std(ddof=1)),
                "roc_auc_mean": float(group["roc_auc"].mean()),
                "roc_auc_std": float(group["roc_auc"].std(ddof=1)),
                "log_loss_mean": float(group["log_loss"].mean()),
                "log_loss_std": float(group["log_loss"].std(ddof=1)),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return
    summary.to_csv(output_dir / "results_member_cv_summary.csv", index=False)
    _plot(summary, output_dir)


if __name__ == "__main__":
    main()
