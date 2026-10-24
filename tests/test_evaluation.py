import numpy as np
import pandas as pd

from inf8239_u03.evaluation import (
    coverage_of_lists,
    head_share,
    intra_list_diversity,
    ndcg_single,
    pareto_flags,
    summarize_ranks,
    target_rank,
    top_k,
)


def test_target_rank_skips_seen_items():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert target_rank(scores, target=2, seen=np.array([0])) == 2


def test_target_rank_breaks_ties_by_lowest_index():
    scores = np.array([1.0, 1.0, 1.0])
    assert target_rank(scores, target=2, seen=np.array([], dtype=int)) == 3
    assert target_rank(scores, target=0, seen=np.array([], dtype=int)) == 1


def test_top_k_excludes_seen_and_is_stable():
    scores = np.array([0.5, 0.9, 0.9, 0.1])
    assert top_k(scores, seen=np.array([1]), k=2).tolist() == [2, 0]


def test_ndcg_single_values():
    assert ndcg_single(1) == 1.0
    assert abs(ndcg_single(3) - 0.5) < 1e-9
    assert ndcg_single(11, k=10) == 0.0
    assert ndcg_single(None) == 0.0


def test_summarize_ranks():
    result = summarize_ranks([1, 20, None, 5], k=10)
    assert result["users"] == 4
    assert result["hit_rate@10"] == 0.5
    assert 0 < result["ndcg@10"] < 1


def test_coverage_and_head_share():
    lists = [[1, 2], [2, 3]]
    assert coverage_of_lists(lists, catalog_size=6) == 0.5
    assert head_share(lists, head_items={2}) == 0.5


def test_diversity_is_zero_for_identical_genres_and_one_for_disjoint():
    genre_sets = {1: frozenset({"A"}), 2: frozenset({"A"}), 3: frozenset({"B"})}
    assert intra_list_diversity([[1, 2]], genre_sets) == 0.0
    assert intra_list_diversity([[1, 3]], genre_sets) == 1.0


def test_pareto_flags_mark_dominated_rows():
    df = pd.DataFrame({"hit_rate@10": [0.10, 0.10, 0.05], "train_seconds": [2.0, 1.0, 3.0]})
    assert pareto_flags(df) == [False, True, False]
