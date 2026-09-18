#!/usr/bin/env python3
"""Short experiments for whether ``"same"`` feature grouping hurts small-H tables.

Context: https://github.com/soda-inria/tabicl/issues/163

TabICLv2 checkpoints are trained with ``col_feature_group="same"`` and
``col_feature_group_size=3``. The grouping architecture is baked into
``in_linear``, so grouping cannot be disabled at inference without new weights.

These experiments therefore:

1. Count wrap-around pair collisions as a function of the number of columns H.
2. Probe pretrained TabICLv2 on adversarial synthetic tasks with few features,
   with and without extra Gaussian columns that lift H to 7 or 15 (where the
   cyclic Golomb-ruler property holds or is much closer to holding).
3. Repeat (2) on a handful of small-feature classification datasets.
4. Compare TabICLv1 (no feature grouping) on the same tasks as a weak control
   (v1 also differs in prior, architecture, and training).

Run::

    python scripts/investigate_feature_grouping.py
    python scripts/investigate_feature_grouping.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_iris,
    load_wine,
    make_circles,
    make_classification,
    make_moons,
)
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tabicl import TabICLClassifier


GROUP_SIZE = 3
# Smallest H for which {±1, ±2, ±3} are distinct mod H (default group size 3).
MIN_H_NO_PAIR_COLLISION = 7
CKPT_V2 = "tabicl-classifier-v2-20260212.ckpt"
CKPT_V1 = "tabicl-classifier-v1-20250208.ckpt"
RESULTS_PATH = Path(__file__).resolve().parent / "feature_grouping_issue163_results.json"


def same_mode_groups(n_features: int, group_size: int = GROUP_SIZE) -> list[tuple[int, ...]]:
    return [tuple((j + 2**i) % n_features for i in range(group_size)) for j in range(n_features)]


def collision_stats(n_features: int, group_size: int = GROUP_SIZE) -> dict:
    groups = same_mode_groups(n_features, group_size)
    pair_counts: Counter = Counter()
    self_alias_groups = 0
    for group in groups:
        if len(set(group)) < len(group):
            self_alias_groups += 1
        members = sorted(set(group))
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                pair_counts[(a, b)] += 1
    n_repeated_pairs = sum(1 for n in pair_counts.values() if n > 1)
    max_pair_repeats = max(pair_counts.values(), default=0)
    return {
        "H": n_features,
        "n_groups": len(groups),
        "groups": [list(g) for g in groups],
        "n_distinct_pairs_in_groups": len(pair_counts),
        "n_repeated_pairs": n_repeated_pairs,
        "max_pair_repeats": max_pair_repeats,
        "self_alias_groups": self_alias_groups,
        "has_pair_collision": n_repeated_pairs > 0 or self_alias_groups > 0,
    }


def pad_gaussian(X: np.ndarray, target_h: int, rng: np.random.Generator) -> np.ndarray:
    n, h = X.shape
    if h >= target_h:
        return X
    extra = rng.normal(size=(n, target_h - h)).astype(np.float64)
    return np.hstack([X, extra])


def _binary_auc(y_true, y_pred, y_proba) -> float | None:
    if len(np.unique(y_true)) != 2:
        return None
    if y_proba is not None and y_proba.shape[1] == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    return float(roc_auc_score(y_true, y_pred))


def evaluate_clf(clf: TabICLClassifier, X_train, y_train, X_test, y_test) -> dict:
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    elapsed = time.perf_counter() - t0
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": _binary_auc(y_test, y_pred, y_proba),
        "n_features_model": int(getattr(clf, "n_features_in_", X_train.shape[1])),
        "seconds": elapsed,
    }


def make_estimator(checkpoint: str, n_estimators: int, random_state: int) -> TabICLClassifier:
    return TabICLClassifier(
        n_estimators=n_estimators,
        checkpoint_version=checkpoint,
        random_state=random_state,
        device="cpu",
        use_amp=False,
        n_jobs=-1,
        verbose=False,
    )


def run_conditions(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    checkpoints: list[str],
    pad_targets: list[int],
    test_size: float = 0.3,
) -> list[dict]:
    X = np.asarray(X, dtype=np.float64)
    y = LabelEncoder().fit_transform(np.asarray(y).ravel())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    rows = []
    for ckpt in checkpoints:
        # Baseline: original columns.
        clf = make_estimator(ckpt, n_estimators, seed)
        metrics = evaluate_clf(clf, X_train, y_train, X_test, y_test)
        rows.append({"checkpoint": ckpt, "condition": "original", "pad_H": X.shape[1], **metrics})

        if ckpt != CKPT_V2:
            continue
        # Padding is a v2-only probe of wrap-around collisions. Extra columns are
        # independent Gaussian noise (constants would be dropped by UniqueFeatureFilter).
        for target_h in pad_targets:
            if X.shape[1] >= target_h:
                continue
            rng_tr = np.random.default_rng(seed + 10_000 + target_h)
            rng_te = np.random.default_rng(seed + 20_000 + target_h)
            metrics = evaluate_clf(
                make_estimator(ckpt, n_estimators, seed),
                pad_gaussian(X_train, target_h, rng_tr),
                y_train,
                pad_gaussian(X_test, target_h, rng_te),
                y_test,
            )
            rows.append(
                {
                    "checkpoint": ckpt,
                    "condition": f"pad_gaussian_{target_h}",
                    "pad_H": target_h,
                    **metrics,
                }
            )
    return rows


def synthetic_datasets(n_samples: int, seed: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = []

    for h in (2, 3, 4, 5, 6, 7, 8):
        X = rng.normal(size=(n_samples, h))
        y = (X[:, 0] > 0).astype(int)
        out.append((f"needle_h{h}", X, y))

    for h in (2, 3, 4, 7):
        X = rng.normal(size=(n_samples, h))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        out.append((f"xor_h{h}", X, y))

    # All original features are signal; grouping them together is natural.
    X = rng.normal(size=(n_samples, 3))
    y = (X.sum(axis=1) > 0).astype(int)
    out.append(("linear_sum_h3", X, y))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=3,
        n_informative=1,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=seed,
    )
    out.append(("sklearn_cls_h3_inf1", X, y))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=seed,
    )
    out.append(("sklearn_cls_h12_inf3", X, y))
    return out


def real_datasets(quick: bool) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out: list[tuple[str, np.ndarray, np.ndarray]] = []

    X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
    out.append(("moons_h2", X, y))
    X, y = make_circles(n_samples=400, noise=0.1, factor=0.5, random_state=0)
    out.append(("circles_h2", X, y))

    data = load_iris()
    out.append(("iris_h4", data.data, data.target))
    data = load_wine()
    out.append(("wine_h13", data.data, data.target))
    if not quick:
        data = load_breast_cancer()
        out.append(("breast_cancer_h30", data.data, data.target))

    openml_specs = [
        ("haberman_h3", "haberman", 43),
        ("blood_transfusion_h4", "blood-transfusion-service-center", 1464),
        ("banknote_h4", "banknote-authentication", 1462),
        ("phoneme_h5", "phoneme", 1489),
    ]
    if quick:
        openml_specs = openml_specs[:1]

    for name, data_id, openml_id in openml_specs:
        try:
            bunch = fetch_openml(data_id=openml_id, as_frame=False, parser="auto")
            X = np.asarray(bunch.data, dtype=np.float64)
            y = bunch.target
            out.append((name, X, y))
        except Exception as exc:  # noqa: BLE001 — keep the rest of the study running
            print(f"skip {name}: {exc}")
    return out


def _delta_stats(deltas: list[float]) -> dict | None:
    if not deltas:
        return None
    return {
        "n_paired": len(deltas),
        "mean_acc_delta": float(np.mean(deltas)),
        "median_acc_delta": float(np.median(deltas)),
        "n_improved": int(sum(d > 1e-12 for d in deltas)),
        "n_degraded": int(sum(d < -1e-12 for d in deltas)),
    }


def summarize(rows: list[dict]) -> dict:
    """Paired accuracy deltas for v2 padding probes and the v1 vs v2 control."""
    by_key = {(r["task"], r["seed"], r["checkpoint"], r["condition"]): r for r in rows}

    def pad_deltas(condition: str, h_lt_7: bool) -> dict | None:
        deltas = []
        keys = {(r["task"], r["seed"]) for r in rows}
        for task, seed in keys:
            orig = by_key.get((task, seed, CKPT_V2, "original"))
            pad = by_key.get((task, seed, CKPT_V2, condition))
            if orig is None or pad is None:
                continue
            if (orig["H"] < MIN_H_NO_PAIR_COLLISION) != h_lt_7:
                continue
            deltas.append(pad["accuracy"] - orig["accuracy"])
        return _delta_stats(deltas)

    v1_minus_v2 = []
    keys = {(r["task"], r["seed"]) for r in rows}
    for task, seed in keys:
        v2 = by_key.get((task, seed, CKPT_V2, "original"))
        v1 = by_key.get((task, seed, CKPT_V1, "original"))
        if v2 is None or v1 is None:
            continue
        v1_minus_v2.append(v1["accuracy"] - v2["accuracy"])

    return {
        "pad7_minus_original_when_H_lt_7": pad_deltas("pad_gaussian_7", h_lt_7=True),
        "pad15_minus_original_when_H_lt_7": pad_deltas("pad_gaussian_15", h_lt_7=True),
        "pad15_minus_original_when_H_ge_7": pad_deltas("pad_gaussian_15", h_lt_7=False),
        "v1_minus_v2_original_accuracy": _delta_stats(v1_minus_v2),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="Fewer tasks/seeds for a smoke run")
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--skip-v1", action="store_true")
    p.add_argument("--skip-real", action="store_true")
    p.add_argument("--skip-synthetic", action="store_true")
    p.add_argument("--output", type=Path, default=RESULTS_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_estimators = args.n_estimators if args.n_estimators is not None else (2 if args.quick else 4)
    n_samples = args.n_samples if args.n_samples is not None else (200 if args.quick else 400)
    n_seeds = 1 if args.quick else 3
    checkpoints = [CKPT_V2] if args.skip_v1 else [CKPT_V2, CKPT_V1]
    pad_targets = [MIN_H_NO_PAIR_COLLISION] if args.quick else [MIN_H_NO_PAIR_COLLISION, 15]

    collisions = [collision_stats(h) for h in range(1, 16)]
    print("=== pair collisions vs H (group size 3, same mode) ===")
    for row in collisions:
        print(
            f"H={row['H']:2d}  repeated_pairs={row['n_repeated_pairs']:2d}  "
            f"max_repeats={row['max_pair_repeats']}  self_alias_groups={row['self_alias_groups']}"
        )

    eval_rows: list[dict] = []
    errors: list[dict] = []

    def consume(task_name: str, X, y, seed: int, family: str) -> None:
        H = int(np.asarray(X).shape[1])
        print(f"\n--- {family}/{task_name} seed={seed} H={H} n={len(y)} ---")
        try:
            rows = run_conditions(
                X,
                y,
                seed=seed,
                n_estimators=n_estimators,
                checkpoints=checkpoints,
                pad_targets=pad_targets,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            errors.append({"task": task_name, "seed": seed, "error": str(exc)})
            return
        for row in rows:
            rec = {"family": family, "task": task_name, "seed": seed, "H": H, **row}
            eval_rows.append(rec)
            auc = rec["roc_auc"]
            auc_s = f"{auc:.3f}" if auc is not None else "n/a"
            print(
                f"  {rec['checkpoint']} {rec['condition']:18s} "
                f"acc={rec['accuracy']:.3f} auc={auc_s} {rec['seconds']:.1f}s"
            )

    if not args.skip_synthetic:
        for seed in range(n_seeds):
            for name, X, y in synthetic_datasets(n_samples, seed=seed):
                consume(name, X, y, seed=seed, family="synthetic")

    if not args.skip_real:
        for name, X, y in real_datasets(quick=args.quick):
            consume(name, X, y, seed=0, family="real")

    summary = summarize(eval_rows)
    payload = {
        "config": {
            "n_estimators": n_estimators,
            "n_samples": n_samples,
            "n_seeds": n_seeds,
            "checkpoints": checkpoints,
            "pad_targets": pad_targets,
            "quick": args.quick,
            "min_H_no_pair_collision": MIN_H_NO_PAIR_COLLISION,
            "note": (
                "v2 grouping cannot be disabled at inference; pad_gaussian_* lifts H "
                "so wrap-around pair collisions disappear (H>=7) or become rarer."
            ),
        },
        "collisions": [{k: v for k, v in row.items() if k != "groups"} for row in collisions],
        "evaluations": eval_rows,
        "summary": summary,
        "errors": errors,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
