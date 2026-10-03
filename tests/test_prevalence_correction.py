"""Tests for majority undersampling and the multiclass Elkan correction.

The binary formula is Theorem 2 of Elkan (IJCAI 2001). The multiclass map is
the prior-shift adjustment that reduces to that formula when there are two
classes; these tests check the reduction, Bayes recovery under a known
class-conditional model, and that TabICLClassifier applies the map to the
undersampled context.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tabicl._sklearn.prevalence import (
    context_index_sets,
    correct_class_prior,
    majority_undersample_index_sets,
    majority_undersample_indices,
)


def classical_elkan(p, observed_prevalence, target_prevalence):
    """Theorem 2 of Elkan (2001), implemented independently of the library."""
    p = np.asarray(p, dtype=np.float64)
    b = float(observed_prevalence)
    b_prime = float(target_prevalence)
    numerator = b_prime * (p - p * b)
    denominator = b - p * b + b_prime * p - b_prime * b
    return numerator / denominator


def _gaussian_posterior(x, means, variances, prior):
    """Exact posterior under diagonal-covariance class-conditional Gaussians."""
    x = np.asarray(x, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    log_joint = []
    for mean, variance, pi in zip(means, variances, prior):
        mean = np.asarray(mean, dtype=np.float64)
        variance = np.asarray(variance, dtype=np.float64)
        quad = np.sum((x - mean) ** 2 / variance, axis=-1)
        log_det = np.sum(np.log(2.0 * np.pi * variance))
        log_joint.append(-0.5 * (quad + log_det) + np.log(pi))
    log_joint = np.stack(log_joint, axis=-1)
    log_joint -= log_joint.max(axis=-1, keepdims=True)
    joint = np.exp(log_joint)
    return joint / joint.sum(axis=-1, keepdims=True)


def test_binary_correction_matches_elkan_formula():
    observed = np.array([0.2, 0.8])
    target = np.array([0.9, 0.1])
    positive = np.linspace(0.0, 1.0, 21)
    proba = np.column_stack([1.0 - positive, positive])

    corrected = correct_class_prior(proba, source_prior=observed, target_prior=target)
    expected = classical_elkan(positive, observed_prevalence=0.8, target_prevalence=0.1)

    assert_allclose(corrected[:, 1], expected, atol=1e-12)
    assert_allclose(corrected.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(corrected >= 0.0)
    assert np.all(corrected <= 1.0)


def test_binary_correction_is_strictly_increasing():
    """A monotone map leaves binary ROC-AUC unchanged."""
    positive = np.linspace(0.0, 1.0, 51)
    proba = np.column_stack([1.0 - positive, positive])
    corrected = correct_class_prior(proba, source_prior=[0.5, 0.5], target_prior=[0.95, 0.05])
    assert np.all(np.diff(corrected[:, 1]) > 0.0)


def test_correction_is_identity_when_priors_match():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(4), size=30)
    prior = np.array([0.7, 0.15, 0.1, 0.05])
    corrected = correct_class_prior(proba, prior, prior * 3.0)  # shared scale cancels
    assert_allclose(corrected, proba, atol=1e-12)


def test_multiclass_correction_recovers_bayes_posterior():
    """Undersampling labels (not features) is a prior shift, so the map is exact."""
    rng = np.random.default_rng(1)
    means = [np.array([-1.5, 0.2]), np.array([0.4, 1.2]), np.array([1.3, -0.8])]
    variances = [np.array([0.6, 1.1]), np.array([0.8, 0.5]), np.array([1.4, 0.7])]
    # Original population prior, and the prior after capping the majority class.
    target_prior = np.array([0.85, 0.10, 0.05])
    source_prior = np.array([0.25, 0.50, 0.25])
    x = rng.normal(size=(200, 2))

    source_posterior = _gaussian_posterior(x, means, variances, source_prior)
    target_posterior = _gaussian_posterior(x, means, variances, target_prior)
    corrected = correct_class_prior(source_posterior, source_prior, target_prior)

    assert_allclose(corrected, target_posterior, atol=1e-10)
    assert_allclose(corrected.sum(axis=1), 1.0, atol=1e-12)


def test_multiclass_correction_can_change_per_class_ranking():
    """Unlike the binary case, the normalizer depends on x, so OVR ranks can move.

    Class 1 is more probable for sample A than sample B before the correction
    and less probable after it. The corrected values are still the right
    probabilities under the target prior; one-vs-rest AUC is not invariant.
    """
    proba = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.05, 0.10, 0.85],
        ]
    )
    corrected = correct_class_prior(proba, source_prior=[1.0, 1.0, 1.0], target_prior=[10.0, 1.0, 1.0])
    assert proba[0, 1] > proba[1, 1]
    assert corrected[0, 1] < corrected[1, 1]


def test_correction_rejects_bad_inputs():
    proba = np.array([[0.2, 0.8]])
    with pytest.raises(ValueError):
        correct_class_prior(proba, source_prior=[0.0, 1.0], target_prior=[0.5, 0.5])
    with pytest.raises(ValueError):
        correct_class_prior(proba, source_prior=[0.5, 0.5], target_prior=[0.5])
    with pytest.raises(ValueError):
        correct_class_prior(np.array([[0.0, 0.0]]), source_prior=[0.5, 0.5], target_prior=[0.2, 0.8])


def test_undersample_is_noop_when_ratio_is_already_small():
    y = np.array([0] * 10 + [1] * 4 + [2] * 3)  # ratio 10/3 < 5
    indices, counts, context_counts = majority_undersample_indices(y, max_imbalance_ratio=5, random_state=0)
    assert indices is None
    assert_allclose(counts, [10, 4, 3])
    assert_allclose(context_counts, counts)

    indices_inf, _, context_inf = majority_undersample_indices(y, max_imbalance_ratio=np.inf, random_state=0)
    assert indices_inf is None
    assert_allclose(context_inf, counts)

    indices_none, _, _ = majority_undersample_indices(y, max_imbalance_ratio=None, random_state=0)
    assert indices_none is None


def test_undersample_caps_every_majority_class_and_keeps_the_minority():
    y = np.array([0] * 100 + [1] * 40 + [2] * 8)
    indices, counts, context_counts = majority_undersample_indices(y, max_imbalance_ratio=5, random_state=0)
    # cap = floor(5 * 8) = 40. Class 1 is already on the cap and is kept.
    assert counts.tolist() == [100, 40, 8]
    assert context_counts.tolist() == [40, 40, 8]
    assert indices is not None
    assert len(indices) == 88
    assert np.all(np.diff(indices) > 0)
    assert np.bincount(y[indices], minlength=3).tolist() == context_counts.tolist()
    minority_rows = set(np.flatnonzero(y == 2).tolist())
    assert minority_rows.issubset(set(indices.tolist()))
    assert context_counts.max() / context_counts.min() <= 5

    again, _, _ = majority_undersample_indices(y, max_imbalance_ratio=5, random_state=0)
    assert np.array_equal(indices, again)
    other, _, _ = majority_undersample_indices(y, max_imbalance_ratio=5, random_state=1)
    assert not np.array_equal(indices, other)


def test_independent_index_sets_share_counts_and_differ():
    y = np.array([0] * 100 + [1] * 8)
    sets, _counts, context_counts = majority_undersample_index_sets(
        y, max_imbalance_ratio=5, n_sets=3, random_state=0
    )
    assert context_counts.tolist() == [40, 8]
    assert len(sets) == 3
    for indices in sets:
        assert np.bincount(y[indices], minlength=2).tolist() == [40, 8]
        assert np.all(np.diff(indices) > 0)
    assert not np.array_equal(sets[0], sets[1])
    # The single-set helper is the first draw of the same seed.
    one, _, _ = majority_undersample_indices(y, max_imbalance_ratio=5, random_state=0)
    assert np.array_equal(one, sets[0])


def test_uniform_subsample_matches_the_requested_size_and_skips_elkan_counts():
    y = np.array([0] * 100 + [1] * 8)
    sets, counts, context_counts = context_index_sets(
        y, subsample=24, max_imbalance_ratio=None, n_sets=3, random_state=0
    )
    assert counts.tolist() == [100, 8]
    # Uniform members do not share a prior, so the correction is left off.
    assert context_counts.tolist() == [100, 8]
    assert len(sets) == 3
    for indices in sets:
        assert len(indices) == 24
        assert np.all(np.diff(indices) > 0)
        assert set(np.unique(y[indices]).tolist()) == {0, 1}
    assert not np.array_equal(sets[0], sets[1])


def test_float_subsample_is_a_fraction_of_the_rows():
    y = np.array([0] * 40 + [1] * 10)
    sets, _, _ = context_index_sets(y, subsample=0.5, max_imbalance_ratio=None, n_sets=1, random_state=0)
    assert len(sets[0]) == 25


def test_classwise_subsample_respects_the_ratio_and_the_budget():
    y = np.array([0] * 100 + [1] * 8)
    sets, counts, context_counts = context_index_sets(
        y, subsample=20, max_imbalance_ratio=5, n_sets=2, random_state=0
    )
    assert counts.tolist() == [100, 8]
    assert context_counts.sum() <= 20
    assert context_counts.min() >= 1
    assert context_counts.max() / context_counts.min() <= 5
    assert len(sets) == 2
    for indices in sets:
        assert np.bincount(y[indices], minlength=2).tolist() == context_counts.tolist()
        assert np.all(np.diff(indices) > 0)
    assert not np.array_equal(sets[0], sets[1])


def test_subsample_does_not_put_majority_rows_back_past_the_ratio():
    y = np.array([0] * 100 + [1] * 8)
    sets, _, context_counts = context_index_sets(
        y, subsample=80, max_imbalance_ratio=5, n_sets=1, random_state=0
    )
    # cap = floor(5 * 8) = 40, so the context is 48 rows, not 80.
    assert context_counts.tolist() == [40, 8]
    assert len(sets[0]) == 48


def test_subsample_none_matches_the_ratio_only_draw():
    y = np.array([0] * 100 + [1] * 8)
    capped, _, capped_counts = context_index_sets(
        y, subsample=None, max_imbalance_ratio=5, n_sets=1, random_state=0
    )
    legacy, _, legacy_counts = majority_undersample_indices(y, max_imbalance_ratio=5, random_state=0)
    assert np.array_equal(capped[0], legacy)
    assert np.array_equal(capped_counts, legacy_counts)

    untouched, _, untouched_counts = context_index_sets(
        y, subsample=None, max_imbalance_ratio=None, n_sets=1, random_state=0
    )
    assert untouched is None
    assert untouched_counts.tolist() == [100, 8]


def test_subsample_rejects_bad_values():
    y = np.array([0] * 10 + [1] * 4)
    with pytest.raises(ValueError):
        context_index_sets(y, subsample=0, max_imbalance_ratio=None, n_sets=1, random_state=0)
    with pytest.raises(ValueError):
        context_index_sets(y, subsample=1.5, max_imbalance_ratio=None, n_sets=1, random_state=0)
    with pytest.raises(ValueError):
        context_index_sets(y, subsample=True, max_imbalance_ratio=None, n_sets=1, random_state=0)
    with pytest.raises(ValueError):
        context_index_sets(y, subsample=50, max_imbalance_ratio=None, n_sets=1, random_state=0)
    with pytest.raises(ValueError):
        context_index_sets(y, subsample=1, max_imbalance_ratio=5, n_sets=1, random_state=0)


def test_undersample_rejects_ratio_below_one():
    y = np.array([0, 0, 1])
    with pytest.raises(ValueError):
        majority_undersample_indices(y, max_imbalance_ratio=0.5, random_state=0)
    with pytest.raises(ValueError):
        majority_undersample_indices(y, max_imbalance_ratio=-np.inf, random_state=0)


def _classifier_kwargs():
    return dict(
        n_estimators=1,
        norm_methods=["none"],
        device="cpu",
        use_amp=False,
        use_fa3=False,
        random_state=0,
    )


@pytest.mark.parametrize("average_logits", [True, False])
def test_classifier_applies_multiclass_correction_to_the_undersampled_context(average_logits):
    from tabicl import TabICLClassifier

    rng = np.random.RandomState(0)
    counts = (80, 24, 8)  # ratio 10 > 5; cap at 5 * 8 = 40
    y_train = np.concatenate([np.full(count, class_id) for class_id, count in enumerate(counts)])
    x_train = rng.normal(size=(len(y_train), 4))
    x_test = rng.normal(size=(12, 4))

    corrected = TabICLClassifier(max_imbalance_ratio=5, average_logits=average_logits, **_classifier_kwargs())
    corrected.fit(x_train, y_train)
    assert corrected.class_counts_.tolist() == [80, 24, 8]
    assert corrected.context_class_counts_.tolist() == [40, 24, 8]
    assert len(corrected.context_indices_) == 1
    indices = corrected.context_indices_[0]
    assert len(indices) == 72
    assert np.bincount(y_train[indices], minlength=3).tolist() == [40, 24, 8]
    assert np.all(np.diff(indices) > 0)

    raw = TabICLClassifier(max_imbalance_ratio=None, average_logits=average_logits, **_classifier_kwargs())
    raw.fit(x_train[indices], y_train[indices])
    expected = correct_class_prior(
        raw.predict_proba(x_test),
        source_prior=corrected.context_class_counts_,
        target_prior=corrected.class_counts_,
    )
    got = corrected.predict_proba(x_test)
    assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_allclose(got.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(got >= 0.0)


@pytest.mark.parametrize("average_logits", [True, False])
def test_correction_is_applied_once_after_independent_members(average_logits):
    """Each member has its own majority subsample; Elkan runs after the ensemble."""
    from tabicl import TabICLClassifier

    rng = np.random.RandomState(0)
    y_train = np.concatenate([np.full(80, 0), np.full(24, 1), np.full(8, 2)])
    x_train = rng.normal(size=(len(y_train), 4))
    x_test = rng.normal(size=(12, 4))
    common = dict(
        norm_methods=["none", "power"],
        feat_shuffle_method="none",
        class_shuffle_method="none",
        device="cpu",
        use_amp=False,
        use_fa3=False,
        random_state=0,
        average_logits=average_logits,
    )
    parent = TabICLClassifier(n_estimators=2, max_imbalance_ratio=5, **common).fit(x_train, y_train)
    member_indices = [
        idx for indices in parent.ensemble_generator_.row_indices_.values() for idx in indices
    ]
    assert len(member_indices) == 2
    assert not np.array_equal(member_indices[0], member_indices[1])

    member_probas = []
    for method, indices in parent.ensemble_generator_.row_indices_.items():
        for idx in indices:
            child = TabICLClassifier(
                n_estimators=1, norm_methods=[method], max_imbalance_ratio=None, **{
                    k: v for k, v in common.items() if k != "norm_methods"
                }
            ).fit(x_train[idx], y_train[idx])
            member_probas.append(child.predict_proba(x_test))
    stacked = np.stack(member_probas, axis=0)
    if average_logits:
        # Child probabilities are softmax(logits / temperature). Averaging the
        # recovered logits matches averaging the logits, then softmax.
        recovered = np.log(np.clip(stacked, 1e-12, None))
        ensembled = recovered.mean(axis=0)
        ensembled -= ensembled.max(axis=1, keepdims=True)
        ensembled = np.exp(ensembled)
        ensembled /= ensembled.sum(axis=1, keepdims=True)
    else:
        ensembled = stacked.mean(axis=0)
        ensembled /= ensembled.sum(axis=1, keepdims=True)
    expected = correct_class_prior(
        ensembled, parent.context_class_counts_, parent.class_counts_
    )
    got = parent.predict_proba(x_test)
    assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert_allclose(got.sum(axis=1), 1.0, atol=1e-5)


def test_classifier_imbalance_ratio_is_a_noop_below_the_cap():
    from sklearn.datasets import make_classification

    from tabicl import TabICLClassifier

    x, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        weights=[0.6, 0.4],
        random_state=0,
    )
    capped = TabICLClassifier(max_imbalance_ratio=5, **_classifier_kwargs())
    disabled = TabICLClassifier(max_imbalance_ratio=None, **_classifier_kwargs())
    capped.fit(x[:60], y[:60])
    disabled.fit(x[:60], y[:60])

    assert capped.imbalance_ratio_ < 5
    assert capped.context_indices_ is None
    assert np.array_equal(capped.class_counts_, capped.context_class_counts_)
    assert_allclose(capped.predict_proba(x[60:]), disabled.predict_proba(x[60:]), rtol=1e-6, atol=1e-6)


def test_classifier_subsample_is_uniform_or_classwise():
    from tabicl import TabICLClassifier

    rng = np.random.RandomState(0)
    y_train = np.concatenate([np.full(80, 0), np.full(8, 1)])
    x_train = rng.normal(size=(len(y_train), 4))
    x_test = rng.normal(size=(6, 4))
    uniform = TabICLClassifier(subsample=20, max_imbalance_ratio=None, **_classifier_kwargs()).fit(x_train, y_train)
    # Ratio 1 caps the majority at the 8 minority rows, so the budget of 20
    # cannot add those majority rows back.
    classwise = TabICLClassifier(subsample=20, max_imbalance_ratio=1, **_classifier_kwargs()).fit(x_train, y_train)

    assert len(uniform.context_indices_[0]) == 20
    assert np.array_equal(uniform.context_class_counts_, uniform.class_counts_)
    assert set(np.unique(y_train[uniform.context_indices_[0]]).tolist()) == {0, 1}

    assert classwise.context_class_counts_.tolist() == [8, 8]
    assert len(classwise.context_indices_[0]) == 16
    assert np.bincount(y_train[classwise.context_indices_[0]], minlength=2).tolist() == [8, 8]

    for model in (uniform, classwise):
        proba = model.predict_proba(x_test)
        assert proba.shape == (len(x_test), 2)
        assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
