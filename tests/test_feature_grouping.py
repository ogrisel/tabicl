"""Combinatorial properties of ColEmbedding.feature_grouping in ``"same"`` mode.

These tests document the wrap-around collisions discussed in
https://github.com/soda-inria/tabicl/issues/163.

In ``"same"`` mode, group ``j`` contains features ``(j + 2**i) % H`` for
``i = 0, ..., feature_group_size - 1``. Relative to the first member this is
the translated Golomb ruler ``{0, 1, 3, ..., 2**(k-1) - 1}``. On the line each
unordered pair appears in at most one group, but wrapping modulo ``H`` only
preserves that property when the nonzero differences are distinct modulo ``H``.
With the default group size of 3 this fails whenever ``H < 7``: ``H in {1, 2, 3}``
self-aliases slots inside a group, while ``H in {4, 5, 6}`` repeats unordered
pairs across groups.
"""

from collections import Counter

import pytest
import torch

from tabicl._model.embedding import ColEmbedding


def _tiny_col_embedding(feature_group="same", feature_group_size=3) -> ColEmbedding:
    return ColEmbedding(
        embed_dim=8,
        num_blocks=1,
        nhead=2,
        dim_feedforward=16,
        num_inds=4,
        feature_group=feature_group,
        feature_group_size=feature_group_size,
        affine=False,
        target_aware=False,
    )


def same_mode_groups(n_features: int, group_size: int = 3) -> list[tuple[int, ...]]:
    """Return the feature index tuples used by ``"same"`` grouping."""
    return [tuple((j + 2**i) % n_features for i in range(group_size)) for j in range(n_features)]


def unordered_pair_counts(groups: list[tuple[int, ...]]) -> Counter:
    counts: Counter = Counter()
    for group in groups:
        members = sorted(set(group))
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                counts[(a, b)] += 1
    return counts


@pytest.mark.parametrize("n_features", [1, 2, 3, 4, 5, 6, 7, 8, 15, 16])
def test_same_mode_matches_col_embedding(n_features):
    embedder = _tiny_col_embedding()
    # Distinct integer feature ids so grouping can be read back from values.
    X = torch.arange(n_features, dtype=torch.float32).view(1, 1, n_features)
    grouped = embedder.feature_grouping(X)
    assert grouped.shape == (1, 1, n_features, 3)
    expected = same_mode_groups(n_features)
    got = [tuple(int(v) for v in grouped[0, 0, j].tolist()) for j in range(n_features)]
    assert got == expected


def test_each_pair_appears_once_when_h_equals_7():
    """H=7 is a Singer difference set / Fano plane: each pair in exactly one group."""
    counts = unordered_pair_counts(same_mode_groups(7))
    assert set(counts.values()) == {1}
    assert len(counts) == 21  # C(7, 2)


@pytest.mark.parametrize("n_features", [4, 5, 6])
def test_wraparound_repeats_pairs_when_4_le_h_lt_7(n_features):
    counts = unordered_pair_counts(same_mode_groups(n_features))
    repeated = {pair: n for pair, n in counts.items() if n > 1}
    assert repeated, f"expected wrap-around pair collisions for H={n_features}"


def test_h_equals_3_self_aliases_instead_of_using_all_features():
    """4 ≡ 1 (mod 3), so each group is a duplicated pair rather than all three columns."""
    groups = same_mode_groups(3)
    assert groups == [(1, 2, 1), (2, 0, 2), (0, 1, 0)]
    assert all(len(set(g)) == 2 for g in groups)
    # Every unordered pair still appears once; the modular failure is self-aliasing.
    assert set(unordered_pair_counts(groups).values()) == {1}


@pytest.mark.parametrize("n_features", [7, 8, 9, 15, 16])
def test_no_pair_collisions_or_self_alias_when_h_at_least_7(n_features):
    groups = same_mode_groups(n_features)
    assert all(len(set(g)) == 3 for g in groups)
    assert max(unordered_pair_counts(groups).values()) == 1


def test_self_repeats_when_h_smaller_than_offsets():
    """For H=1 and H=2 some group slots alias the same feature."""
    assert same_mode_groups(1) == [(0, 0, 0)]
    # offsets 1, 2, 4 modulo 2 -> (1, 0, 0) for group 0
    assert same_mode_groups(2)[0] == (1, 0, 0)
