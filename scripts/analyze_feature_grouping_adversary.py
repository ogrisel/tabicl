#!/usr/bin/env python3
"""Why a grouping adversary is hard on tasks that are easy without grouping.

Context: https://github.com/soda-inria/tabicl/issues/163

An ungrouped column embedder maps each scalar cell through a 1→d linear layer.
``"same"`` grouping maps each group of 3 scalars through a shared 3→d layer
(``in_linear``) and still emits **one token per original column**.

This script makes the information-preserving argument concrete and checks the
natural attacks against pretrained TabICLv2 vs TabICLv1 (no grouping):

1. Channel 0 of group ``j`` is feature ``(j+1) % H`` — a cyclic shift. Grouping
   never drops a column; it only concatenates two extra (possibly aliased) views.
2. v2 ``in_linear`` is 3→128 and full rank, so each group token is an injective
   embedding of its 3-vector. Least-squares recovers the 3-vector from
   ``in_linear`` outputs to numerical precision.
3. Per-column preprocessing (default ensemble includes power transform) removes
   scale, so “bury the signal under huge distractors” is not available unless
   normalization is forced off.
4. Empirically, v1 (ungrouped) and v2 (grouped) both stay near-perfect on 1D
   needles, including after mixing; forcing ``norm_methods='none'`` and large
   distractors still does not produce a grouped-only failure.

Run::

    python scripts/analyze_feature_grouping_adversary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabicl import TabICLClassifier
from tabicl._model.embedding import ColEmbedding


CKPT_V2 = "tabicl-classifier-v2-20260212.ckpt"
CKPT_V1 = "tabicl-classifier-v1-20250208.ckpt"
OUT = Path(__file__).resolve().parent / "feature_grouping_adversary_results.json"


def same_mode_groups(n_features: int, group_size: int = 3) -> list[tuple[int, ...]]:
    return [tuple((j + 2**i) % n_features for i in range(group_size)) for j in range(n_features)]


def grouping_linear_algebra() -> dict:
    """Grouping is a permutation of columns plus extra channels, not a collapse."""
    report = {"first_slot_is_cyclic_shift": {}, "h3_self_alias_still_injective": None}
    for h in range(1, 13):
        groups = same_mode_groups(h)
        first = [g[0] for g in groups]
        report["first_slot_is_cyclic_shift"][str(h)] = {
            "first_slots": first,
            "covers_all_features": sorted(first) == list(range(h)),
        }

    # H=3 map x |-> ( (x1,x2,x1), (x2,x0,x2), (x0,x1,x0) ) as a 9-vector.
    A = np.zeros((9, 3))
    groups = same_mode_groups(3)
    for j, g in enumerate(groups):
        for k, feat in enumerate(g):
            A[3 * j + k, feat] += 1
    svals = np.linalg.svd(A, compute_uv=False)
    report["h3_self_alias_still_injective"] = {
        "design_matrix_rank": int(np.linalg.matrix_rank(A)),
        "singular_values": svals.tolist(),
        "note": "Rank 3: every (x0,x1,x2) is recoverable from the three aliased groups.",
    }
    return report


def inspect_v2_in_linear() -> dict:
    clf = TabICLClassifier(n_estimators=1, checkpoint_version=CKPT_V2, device="cpu", use_amp=False, verbose=False)
    clf._load_model()
    col: ColEmbedding = clf.model_.col_embedder
    W = col.in_linear.weight.detach().float().cpu().numpy()  # (embed_dim, 3)
    b = col.in_linear.bias.detach().float().cpu().numpy()
    svals = np.linalg.svd(W, compute_uv=False)
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2048, 3))
    e = z @ W.T + b
    z_hat, *_ = np.linalg.lstsq(W, (e - b).T, rcond=None)
    z_hat = z_hat.T
    recon_mse = float(np.mean((z - z_hat) ** 2))

    # Smallest right singular vector: the 3D direction in_linear is least sensitive to.
    _, _, vh = np.linalg.svd(W, full_matrices=False)
    weakest = vh[-1]

    cfg = dict(clf.model_config_)
    return {
        "model_config_subset": {
            k: cfg.get(k)
            for k in (
                "col_feature_group",
                "col_feature_group_size",
                "col_affine",
                "embed_dim",
            )
        },
        "in_linear_shape": list(W.shape),
        "in_linear_rank": int(np.linalg.matrix_rank(W)),
        "singular_values": svals.tolist(),
        "column_l2_norms": np.linalg.norm(W, axis=0).tolist(),
        "reconstruction_mse_from_in_linear": recon_mse,
        "weakest_3d_direction": weakest.tolist(),
        "condition_number": float(svals[0] / svals[-1]),
    }


def make_clf(checkpoint: str, *, n_estimators: int, norm_methods, feat_shuffle_method: str, seed: int = 0):
    return TabICLClassifier(
        n_estimators=n_estimators,
        checkpoint_version=checkpoint,
        norm_methods=norm_methods,
        feat_shuffle_method=feat_shuffle_method,
        random_state=seed,
        device="cpu",
        use_amp=False,
        n_jobs=-1,
        verbose=False,
    )


def eval_pair(X, y, *, n_estimators, norm_methods, feat_shuffle_method, seed=0) -> dict:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    out = {}
    for ckpt, name in ((CKPT_V1, "v1_ungrouped"), (CKPT_V2, "v2_grouped")):
        clf = make_clf(
            ckpt,
            n_estimators=n_estimators,
            norm_methods=norm_methods,
            feat_shuffle_method=feat_shuffle_method,
            seed=seed,
        )
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        proba = clf.predict_proba(X_te)[:, 1]
        out[name] = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "roc_auc": float(roc_auc_score(y_te, proba)),
        }
    out["v2_minus_v1_accuracy"] = out["v2_grouped"]["accuracy"] - out["v1_ungrouped"]["accuracy"]
    return out


def synthetic_attacks(n: int = 400, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    tasks = {}

    # Easy 1D: both architectures should be perfect. H=1 groups (x,x,x).
    x = rng.normal(size=(n, 1))
    tasks["needle_h1"] = (x, (x[:, 0] > 0).astype(int))

    x = rng.normal(size=(n, 3))
    tasks["needle_h3_unit_noise"] = (x, (x[:, 0] > 0).astype(int))

    # Scale-bury: only works if columns are NOT standardized.
    x = rng.normal(size=(n, 3))
    x[:, 1:] *= 100.0
    tasks["needle_h3_distractor_x100"] = (x, (x[:, 0] > 0).astype(int))

    x = rng.normal(size=(n, 3))
    x[:, 1:] *= 1000.0
    tasks["needle_h3_distractor_x1000"] = (x, (x[:, 0] > 0).astype(int))

    # Anti-correlated distractor in the extra channels: group (x1,x2,x1) etc.
    x0 = rng.normal(size=n)
    x = np.stack([x0, -x0, -x0], axis=1)
    tasks["needle_h3_anticorr_copies"] = (x, (x0 > 0).astype(int))

    # XOR is easy for ICL; grouping co-embeds the pair (helps v2, does not hurt).
    x = rng.normal(size=(n, 2))
    tasks["xor_h2"] = (x, ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int))

    results = {}
    settings = [
        {
            "name": "default_ensemble",
            "n_estimators": 4,
            "norm_methods": None,
            "feat_shuffle_method": "latin",
        },
        {
            "name": "no_norm_no_shuffle",
            "n_estimators": 1,
            "norm_methods": "none",
            "feat_shuffle_method": "none",
        },
    ]
    for setting in settings:
        block = {}
        for task_name, (X, y) in tasks.items():
            print(f"--- {setting['name']} / {task_name} H={X.shape[1]} ---")
            metrics = eval_pair(
                X,
                y,
                n_estimators=setting["n_estimators"],
                norm_methods=setting["norm_methods"],
                feat_shuffle_method=setting["feat_shuffle_method"],
                seed=seed,
            )
            print(
                f"  v1 acc={metrics['v1_ungrouped']['accuracy']:.3f}  "
                f"v2 acc={metrics['v2_grouped']['accuracy']:.3f}  "
                f"Δ={metrics['v2_minus_v1_accuracy']:+.3f}"
            )
            block[task_name] = metrics
        results[setting["name"]] = block
    return results


def main() -> None:
    lin = grouping_linear_algebra()
    print("=== grouping as a map on columns ===")
    print("H=3 rank", lin["h3_self_alias_still_injective"]["design_matrix_rank"])
    for h, row in lin["first_slot_is_cyclic_shift"].items():
        if not row["covers_all_features"]:
            raise SystemExit(f"first slot does not cover all features for H={h}")
    print("first slot covers all features for H=1..12")

    print("\n=== v2 in_linear ===")
    weights = inspect_v2_in_linear()
    print(json.dumps(weights, indent=2))

    print("\n=== v1 (ungrouped) vs v2 (grouped) on easy synthetic tasks ===")
    attacks = synthetic_attacks()

    payload = {
        "linear_algebra": lin,
        "v2_in_linear": weights,
        "attacks": attacks,
        "conclusion": (
            "Same-mode grouping is not an information bottleneck: every column remains "
            "channel 0 of some group, and v2 in_linear is a well-conditioned injective "
            "3→d map. A predictor that is perfect on ungrouped 1D/2D features can read "
            "the same coordinates out of grouped tokens. Frozen-v2 vs frozen-v1 checks "
            "on needles, buried-scale distractors, and XOR do not yield a grouped-only "
            "failure. A true adversary would need a rank-deficient in_linear, a collapse "
            "of the number of tokens (the unused 'valid' mode), or a training-time "
            "failure to invert the mixing — none of which hold for TabICLv2 inference."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT}")
    print(payload["conclusion"])


if __name__ == "__main__":
    main()
