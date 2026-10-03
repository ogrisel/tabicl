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

from numbers import Integral, Real

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


def _validate_subsample(subsample) -> None:
    """Reject a ``subsample`` value that is not ``None``, a positive int, or a fraction."""
    if subsample is None:
        return
    if isinstance(subsample, bool) or not isinstance(subsample, Real):
        raise ValueError(
            "subsample must be None, a positive int, or a float in (0, 1]; "
            f"got {subsample!r}."
        )
    if isinstance(subsample, Integral):
        if int(subsample) < 1:
            raise ValueError(f"subsample must be at least 1; got {subsample!r}.")
        return
    value = float(subsample)
    if not np.isfinite(value) or not (0.0 < value <= 1.0):
        raise ValueError(f"float subsample must be in (0, 1]; got {subsample!r}.")


def _subsample_count(subsample, n_samples: int) -> int | None:
    """Row budget, or ``None`` when ``subsample`` does not limit the context.

    ``None`` keeps every row. An integer is a row count and cannot exceed
    ``n_samples``. A float is the fraction of ``n_samples`` used by
    ``BaggingClassifier(max_samples=...)``, truncated with ``int`` and raised
    to at least 1.
    """
    _validate_subsample(subsample)
    if subsample is None:
        return None
    if isinstance(subsample, Integral):
        count = int(subsample)
        if count > n_samples:
            raise ValueError(f"subsample={count} is greater than n_samples={n_samples}.")
        return count
    return max(int(float(subsample) * n_samples), 1)


def _largest_remainder(upper: np.ndarray, n_target: int) -> np.ndarray:
    """Spread ``n_target`` rows across classes in proportion to ``upper``."""
    upper = np.asarray(upper, dtype=np.int64)
    n_classes = int(upper.shape[0])
    if n_target < n_classes:
        raise ValueError(
            f"subsample keeps {n_target} rows but there are {n_classes} classes. "
            "Each class needs at least one row."
        )
    if n_target >= int(upper.sum()):
        return upper.copy()
    weights = upper.astype(np.float64)
    raw = weights / weights.sum() * n_target
    alloc = np.minimum(np.maximum(np.floor(raw).astype(np.int64), 1), upper)
    while int(alloc.sum()) > n_target:
        surplus = alloc.astype(np.float64) - raw
        candidates = np.flatnonzero(alloc > 1)
        if len(candidates) == 0:
            break
        idx = int(candidates[np.argmax(surplus[candidates])])
        alloc[idx] -= 1
    leftover = n_target - int(alloc.sum())
    fractional = raw - np.floor(raw)
    for idx in np.argsort(-fractional):
        if leftover <= 0:
            break
        if alloc[idx] >= upper[idx]:
            continue
        alloc[idx] += 1
        leftover -= 1
    if leftover > 0:
        for idx in np.argsort(-(upper - alloc)):
            room = int(upper[idx] - alloc[idx])
            take = min(room, leftover)
            alloc[idx] += take
            leftover -= take
            if leftover <= 0:
                break
    return alloc


def _repair_ratio(alloc: np.ndarray, ratio_limit: float, upper: np.ndarray) -> np.ndarray:
    """Move or drop majority rows until ``max(alloc) / min(alloc)`` is within the cap."""
    alloc = np.asarray(alloc, dtype=np.int64).copy()
    upper = np.asarray(upper, dtype=np.int64)
    for _ in range(int(alloc.sum()) + 1):
        minority = int(alloc.min())
        cap = max(int(np.floor(ratio_limit * float(minority) + 1e-8)), 1)
        if int(alloc.max()) <= cap:
            return alloc
        largest = int(np.argmax(alloc))
        smallest = int(np.argmin(alloc))
        alloc[largest] -= 1
        if alloc[smallest] < upper[smallest]:
            alloc[smallest] += 1
    return alloc


def _class_quotas(class_counts: np.ndarray, ratio_limit: float | None, n_target: int | None) -> np.ndarray:
    """Per-class row budget shared by every ensemble member."""
    if ratio_limit is None:
        upper = np.asarray(class_counts, dtype=np.int64).copy()
    else:
        cap = max(int(np.floor(float(ratio_limit) * float(class_counts.min()) + 1e-8)), 1)
        upper = np.minimum(np.asarray(class_counts, dtype=np.int64), cap)
    if n_target is None or n_target >= int(upper.sum()):
        return upper
    alloc = _largest_remainder(upper, n_target)
    if ratio_limit is None:
        return alloc
    return _repair_ratio(alloc, ratio_limit, upper)


def _draw_quotas(class_rows: list[np.ndarray], quotas: np.ndarray, n_sets: int, rng) -> list[np.ndarray]:
    """Independent without-replacement draws that all realize ``quotas``."""
    index_sets = []
    for _ in range(n_sets):
        kept = []
        for rows, quota in zip(class_rows, quotas):
            quota = int(quota)
            chosen = rows if quota == len(rows) else rng.choice(rows, size=quota, replace=False)
            kept.append(np.asarray(chosen, dtype=np.int64).reshape(-1))
        indices = np.concatenate(kept)
        indices.sort()
        index_sets.append(indices)
    return index_sets


def _uniform_index_sets(y: np.ndarray, class_rows: list[np.ndarray], n_target: int, n_sets: int, rng) -> list[np.ndarray]:
    """Uniform draws of ``n_target`` rows that still contain every class."""
    n_classes = len(class_rows)
    if n_target < n_classes:
        raise ValueError(
            f"subsample keeps {n_target} rows but there are {n_classes} classes. "
            "Each class needs at least one row."
        )
    n_samples = int(y.shape[0])
    index_sets = []
    for _ in range(n_sets):
        chosen = None
        for _attempt in range(10000):
            candidate = rng.choice(n_samples, size=n_target, replace=False)
            if len(np.unique(y[candidate])) == n_classes:
                chosen = candidate
                break
        if chosen is None:
            chosen = rng.choice(n_samples, size=n_target, replace=False)
            present = set(np.unique(y[chosen]).tolist())
            for class_id, rows in enumerate(class_rows):
                if class_id in present:
                    continue
                member_counts = np.bincount(y[chosen], minlength=n_classes)
                donors = np.flatnonzero(member_counts[y[chosen]] > 1)
                if len(donors) == 0:
                    donors = np.flatnonzero(y[chosen] != class_id)
                chosen[int(donors[0])] = int(rng.choice(rows))
                present.add(class_id)
        chosen = np.asarray(chosen, dtype=np.int64)
        chosen.sort()
        index_sets.append(chosen)
    return index_sets


def context_index_sets(
    y: np.ndarray,
    subsample,
    max_imbalance_ratio: float | None,
    n_sets: int,
    random_state,
) -> tuple[list[np.ndarray] | None, np.ndarray, np.ndarray]:
    """Per-member context indices for ``subsample`` and ``max_imbalance_ratio``.

    ``subsample`` is the fixed row budget used by ``BaggingClassifier``'s
    ``max_samples``: ``None`` does not impose one, an int is a row count, and
    a float is a fraction of ``len(y)``. Draws are without replacement.

    When ``max_imbalance_ratio`` is ``None``, a finite budget is drawn
    uniformly and the Elkan correction is not applied. Members then do not
    share class counts, so the returned context counts equal the original
    class counts and the caller leaves probabilities unchanged.

    When ``max_imbalance_ratio`` is set, the same budget is drawn classwise.
    Majority classes are capped, the remaining rows are spread in proportion
    to those caps, and every member uses that same count vector. The Elkan
    correction therefore has one source prior. A budget larger than the
    capped context does not add majority rows back.
    """
    if n_sets < 1:
        raise ValueError(f"n_sets must be at least 1, got {n_sets}.")
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

    n_target = _subsample_count(subsample, int(y.shape[0]))
    ratio_limit = _validate_max_imbalance_ratio(max_imbalance_ratio)
    if n_target is None or n_target >= int(y.shape[0]):
        if ratio_limit is None:
            return None, class_counts, class_counts.copy()
        return majority_undersample_index_sets(y, ratio_limit, n_sets, random_state)

    class_rows = [np.flatnonzero(y == class_id) for class_id in range(n_classes)]
    rng = check_random_state(random_state)
    if ratio_limit is None:
        index_sets = _uniform_index_sets(y, class_rows, n_target, n_sets, rng)
        # Members do not share a class prior, so the post-hoc map is skipped.
        return index_sets, class_counts, class_counts.copy()

    quotas = _class_quotas(class_counts, ratio_limit, n_target)
    if np.array_equal(quotas, class_counts):
        return None, class_counts, class_counts.copy()
    index_sets = _draw_quotas(class_rows, quotas, n_sets, rng)
    return index_sets, class_counts, quotas.astype(np.int64, copy=False)


def majority_undersample_index_sets(
    y: np.ndarray,
    max_imbalance_ratio: float | None,
    n_sets: int,
    random_state,
) -> tuple[list[np.ndarray] | None, np.ndarray, np.ndarray]:
    """Independent majority-class subsamples that share the same class counts.

    Each set keeps every class that is already under the cap, including the
    rarest class, and draws its own without-replacement subset of each larger
    class. The draws are sequential on one RNG, so the sets differ when the
    majority class is larger than the cap. Every set has the same per-class
    counts, which is the source prior used by the single post-hoc correction.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Integer class labels encoded as ``0 .. n_classes - 1``.

    max_imbalance_ratio : float or None
        Largest allowed majority-to-minority count ratio. ``None`` and
        ``inf`` disable undersampling.

    n_sets : int
        Number of independent index sets. Must be at least 1.

    random_state : int, RandomState, or None
        Seed for which majority-class rows are kept.

    Returns
    -------
    index_sets : list of ndarray, each of shape (n_context,), or None
        ``None`` when undersampling is a no-op. Otherwise one sorted index
        array per set. Sorting is only for a stable index set: attention
        over rows is permutation-equivariant when each row stays tied to
        its label.

    class_counts : ndarray of shape (n_classes,)
        Counts in ``y``.

    context_counts : ndarray of shape (n_classes,)
        Counts inside every returned set. Equal to ``class_counts`` when no
        rows are dropped.
    """
    if n_sets < 1:
        raise ValueError(f"n_sets must be at least 1, got {n_sets}.")

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
    class_rows = [np.flatnonzero(y == class_id) for class_id in range(n_classes)]

    rng = check_random_state(random_state)
    index_sets = []
    context_counts = None
    for _ in range(n_sets):
        kept = []
        for rows, count in zip(class_rows, class_counts):
            chosen = rows
            if count > cap:
                chosen = rng.choice(rows, size=cap, replace=False)
            kept.append(np.asarray(chosen, dtype=np.int64).reshape(-1))
        indices = np.concatenate(kept)
        indices.sort()
        counts = np.bincount(y[indices], minlength=n_classes).astype(np.int64, copy=False)
        if context_counts is None:
            context_counts = counts
        elif not np.array_equal(counts, context_counts):
            raise RuntimeError("Independent undersamples produced different class counts.")
        index_sets.append(indices)
    return index_sets, class_counts, context_counts


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
    index_sets, class_counts, context_counts = majority_undersample_index_sets(
        y, max_imbalance_ratio, n_sets=1, random_state=random_state
    )
    if index_sets is None:
        return None, class_counts, context_counts
    return index_sets[0], class_counts, context_counts
