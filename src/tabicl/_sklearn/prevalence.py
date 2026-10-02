"""Class-prior correction and majority-class undersampling.

Undersampling majority classes changes the label prior seen by TabICL at
inference time. The predictive probabilities of a classifier fit under that
shifted prior are mapped back to the original prior with the multiclass
extension of Elkan's correction.

Binary case
-----------
Theorem 2 of Elkan (IJCAI 2001, https://cseweb.ucsd.edu/~elkan/rescale.pdf)
rescales a positive-class probability ``p`` estimated under an observed
prevalence ``b`` to the probability under a target prevalence ``b'``:

.. math::

    p' = \\frac{b' (p - p b)}{b - p b + b' p - b' b}.

This is identical to the odds form

.. math::

    \\frac{p'}{1 - p'} = \\frac{p}{1 - p} \\cdot \\frac{b' / (1 - b')}{b / (1 - b)}.

Multiclass case
---------------
The same identity follows from Bayes' rule whenever only the class prior
changes and the class-conditional feature distributions :math:`P(x \\mid y)`
stay fixed (prior shift, or label-dependent sampling independent of ``x``).
For classes :math:`k = 1 \\ldots K` and source / target priors :math:`\\pi`
and :math:`\\pi'`:

.. math::

    P'(y = k \\mid x)
        = \\frac{P(y = k \\mid x) \\, \\pi'_k / \\pi_k}
               {\\sum_j P(y = j \\mid x) \\, \\pi'_j / \\pi_j}.

For :math:`K = 2` this reduces exactly to Elkan's formula. It is the
closed-form prior adjustment of Saerens, Latinne, and Decaestecker
(Neural Computation, 2002). Random undersampling of majority classes is
independent of the features given the label, so the assumption holds for
the context TabICL attends to.

The map is strictly increasing in the positive probability when :math:`K = 2`,
so binary ROC-AUC is unchanged by the correction itself. For :math:`K > 2`
the per-example normalizer depends on ``x``, so one-vs-rest ranking of a
single class can change. The corrected probabilities are still the ones
calibrated to :math:`\\pi'`.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state


def correct_class_prior(
    proba: np.ndarray,
    source_prior: np.ndarray,
    target_prior: np.ndarray,
) -> np.ndarray:
    """Map probabilities from ``source_prior`` to ``target_prior``.

    Parameters
    ----------
    proba : ndarray of shape (n_samples, n_classes)
        Class probabilities estimated under ``source_prior``. Rows should be
        non-negative. Zero entries stay zero.

    source_prior : array-like of shape (n_classes,)
        Class prior the probabilities are calibrated to (counts or
        frequencies). Every entry must be strictly positive. A shared
        positive scale cancels after normalization.

    target_prior : array-like of shape (n_classes,)
        Class prior the returned probabilities should be calibrated to.

    Returns
    -------
    corrected : ndarray of shape (n_samples, n_classes)
        Probabilities under ``target_prior``. Each row sums to one.

    Raises
    ------
    ValueError
        If shapes do not match, a prior is not strictly positive, or a row
        of ``proba`` is entirely zero.
    """
    proba = np.asarray(proba, dtype=np.float64)
    source_prior = np.asarray(source_prior, dtype=np.float64).reshape(-1)
    target_prior = np.asarray(target_prior, dtype=np.float64).reshape(-1)

    if proba.ndim != 2:
        raise ValueError(f"proba must have shape (n_samples, n_classes), got {proba.shape}.")
    n_classes = proba.shape[1]
    if source_prior.shape != (n_classes,) or target_prior.shape != (n_classes,):
        raise ValueError(
            "source_prior and target_prior must have shape "
            f"({n_classes},), got {source_prior.shape} and {target_prior.shape}."
        )
    if np.any(proba < 0) or not np.all(np.isfinite(proba)):
        raise ValueError("proba must be finite and non-negative.")
    if np.any(source_prior <= 0) or np.any(target_prior <= 0) or not np.all(np.isfinite(source_prior)):
        raise ValueError("Class priors must be finite and strictly positive.")
    if not np.all(np.isfinite(target_prior)):
        raise ValueError("Class priors must be finite and strictly positive.")

    log_weight = np.log(target_prior) - np.log(source_prior)
    with np.errstate(divide="ignore"):
        log_corrected = np.log(proba) + log_weight

    row_max = np.max(log_corrected, axis=1, keepdims=True)
    # An all-zero probability row has max -inf. Leave it for the check below.
    stable_max = np.where(np.isfinite(row_max), row_max, 0.0)
    corrected = np.exp(log_corrected - stable_max)
    denom = corrected.sum(axis=1, keepdims=True)
    if np.any(denom <= 0) or not np.all(np.isfinite(denom)):
        raise ValueError("Cannot correct a sample whose class probabilities are all zero.")
    corrected /= denom
    return corrected


def _validate_max_imbalance_ratio(max_imbalance_ratio: float | None) -> float | None:
    """Return the ratio cap, or ``None`` when undersampling is disabled.

    ``None`` and positive infinity disable the cap. Any other non-finite
    value, any value below 1, and non-numeric inputs are rejected.
    """
    if max_imbalance_ratio is None:
        return None
    if isinstance(max_imbalance_ratio, bool) or not isinstance(
        max_imbalance_ratio, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            "max_imbalance_ratio must be a float >= 1, inf, or None; "
            f"got {max_imbalance_ratio!r}."
        )
    ratio_limit = float(max_imbalance_ratio)
    if np.isinf(ratio_limit) and ratio_limit > 0:
        return None
    if not np.isfinite(ratio_limit) or ratio_limit < 1.0:
        raise ValueError(f"max_imbalance_ratio must be >= 1, inf, or None; got {max_imbalance_ratio!r}.")
    return ratio_limit


def majority_undersample_indices(
    y: np.ndarray,
    max_imbalance_ratio: float | None,
    random_state,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Indices of a context whose class counts respect ``max_imbalance_ratio``.

    Majority classes (every class with more than
    ``floor(max_imbalance_ratio * n_minority)`` rows) are subsampled without
    replacement. The rarest class is kept in full, and so is every class
    already under the cap. If the natural ratio
    ``max(count) / min(count)`` is already at most ``max_imbalance_ratio``,
    no rows are dropped.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Integer class labels encoded as ``0 .. n_classes - 1``, with every
        class present at least once.

    max_imbalance_ratio : float or None
        Largest allowed majority-to-minority count ratio. ``None`` and
        ``inf`` disable undersampling. Values below 1 are rejected because
        majority undersampling cannot make a class rarer than the minority.

    random_state : int, RandomState, or None
        Seed for which majority-class rows are kept.

    Returns
    -------
    indices : ndarray of shape (n_context,) or None
        Positions into ``y`` to keep, sorted into the original row order.
        ``None`` when undersampling is a no-op. Sorting is for a stable
        index set. Column embedding and in-context attention are
        permutation-equivariant over rows when each row stays tied to its
        label, so a shuffle of the same rows changes predictions only at
        floating-point noise. Rotary positions are applied across features
        within a row, not across training rows.

    class_counts : ndarray of shape (n_classes,)
        Counts in ``y``.

    context_counts : ndarray of shape (n_classes,)
        Counts after undersampling. Equal to ``class_counts`` when no rows
        are dropped.

    Raises
    ------
    ValueError
        If ``max_imbalance_ratio`` is below 1, or ``y`` does not contain
        every class id in ``0 .. n_classes - 1``.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y.shape}.")
    if y.size == 0:
        raise ValueError("y must contain at least one sample.")
    if y.min() < 0:
        raise ValueError("y must contain non-negative integer class ids.")

    n_classes = int(y.max()) + 1
    class_counts = np.bincount(y, minlength=n_classes).astype(np.int64, copy=False)
    if np.any(class_counts == 0):
        raise ValueError("y is missing a class id between 0 and max(y). Encode labels first.")

    ratio_limit = _validate_max_imbalance_ratio(max_imbalance_ratio)
    ratio = float(class_counts.max() / class_counts.min())
    if ratio_limit is None or ratio <= ratio_limit:
        return None, class_counts, class_counts.copy()

    n_minority = int(class_counts.min())
    cap = int(np.floor(ratio_limit * float(n_minority) + 1e-8))
    cap = max(cap, 1)

    rng = check_random_state(random_state)
    kept = []
    for class_id, count in enumerate(class_counts):
        class_rows = np.flatnonzero(y == class_id)
        if count > cap:
            class_rows = rng.choice(class_rows, size=cap, replace=False)
        kept.append(np.asarray(class_rows, dtype=np.int64).reshape(-1))

    indices = np.concatenate(kept)
    # Sort for a stable index set. Row order is not a model input: attention
    # over rows is permutation-equivariant when features stay tied to labels.
    indices.sort()
    # Counts are taken from the selected rows so they cannot drift from the
    # index set if the per-class draw is ever changed.
    context_counts = np.bincount(y[indices], minlength=n_classes).astype(np.int64, copy=False)
    return indices, class_counts, context_counts
