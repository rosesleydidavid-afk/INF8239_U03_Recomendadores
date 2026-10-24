import numpy as np

from inf8239_u03.recommenders import MatrixFactorization
from inf8239_u03.scorers import (
    build_genre_matrix,
    content_profile_scores,
    hybrid_rerank,
    mf_scores,
    popularity_scores,
)


def test_weighted_popularity_shrinks_items_with_little_evidence(sample_ratings):
    items = np.array([1, 2, 3, 4, 5])
    count, weighted, m = popularity_scores(sample_ratings, items)
    assert count.tolist() == [2, 2, 2, 2, 1]
    global_mean = sample_ratings["rating"].mean()
    item5_mean = sample_ratings.loc[sample_ratings["movieId"].eq(5), "rating"].iloc[0]
    assert abs(weighted[4] - global_mean) < abs(item5_mean - global_mean)
    assert m > 0


def test_mf_scores_match_scalar_prediction(sample_ratings):
    model = MatrixFactorization(factors=3, seed=1).fit(sample_ratings, epochs=3)
    vector = mf_scores(model, 1)
    for position, item in enumerate(model.items):
        assert abs(vector[position] - model.predict(1, item)) < 1e-9


def test_content_profile_prefers_items_with_liked_genres(sample_movies):
    matrix, row_of = build_genre_matrix(sample_movies)
    rows = [row_of[3], row_of[4]]  # el usuario valoró alto dos dramas
    universe = [row_of[item] for item in [1, 2, 3, 4, 5]]
    scores = content_profile_scores(rows, np.array([5.0, 5.0]), matrix, universe)
    assert scores[3] > scores[0]  # Drama|Romance supera a Animation|Comedy


def test_hybrid_rerank_alpha_extremes_and_seen_exclusion():
    mf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    content = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    seen = np.array([0])
    only_mf = hybrid_rerank(mf, content, seen, alpha=1.0, pool=4)
    only_content = hybrid_rerank(mf, content, seen, alpha=0.0, pool=4)
    assert only_mf.tolist() == [1, 2, 3, 4]
    assert only_content.tolist() == [4, 3, 2, 1]
    assert 0 not in only_mf and 0 not in only_content
