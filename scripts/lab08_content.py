from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from inf8239_u03.config import MOVIELENS_DIR, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.recommenders import ContentRecommender, weighted_popularity

DEFAULT_TITLES = ["Toy Story (1995)", "Matrix, The (1999)", "Titanic (1997)"]
QUANTILE = 0.80

def genre_set(text: str) -> frozenset:
    return frozenset(text.split("|"))


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--titles", nargs="+", default=DEFAULT_TITLES)
    args = parser.parse_args()

    ratings, movies = load_movielens(MOVIELENS_DIR)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    # 1. Baseline de popularidad suavizada
    popular = weighted_popularity(ratings, movies, QUANTILE)
    columns = ["title", "count", "mean", "weighted_score"]
    popular.head(10)[columns].to_csv(reports / "popular_top10.csv")
    counts = ratings.groupby("movieId")["rating"].count()
    popularity_summary = {
        "global_mean": round(float(ratings["rating"].mean()), 4),
        "smoothing_constant_m": float(counts.quantile(QUANTILE)),
        "eligible_movies": int(len(popular)),
        "catalog_movies": int(len(movies)),
        "catalog_share": round(len(popular) / len(movies), 4),
    }
    print("Popularidad (Top 10):\n", popular.head(10)[columns])
    print("\nResumen de popularidad:", popularity_summary)

    # 2. Recomendación por contenido: una lista por cada película consultada
    recommender = ContentRecommender().fit(movies)
    titles_available = set(recommender.movies["title"])
    frames, summaries = [], []
    for title in args.titles:
        if title not in titles_available:
            stem = title.split(" (")[0]
            close = recommender.movies.loc[
                recommender.movies["title"].str.contains(stem, case=False, regex=False), "title"
            ].head(5).tolist()
            print(f"\nNo se encontró '{title}'. Títulos parecidos: {close}")
            continue
        top = recommender.recommend(title, 10)
        index = int(recommender.movies.index[recommender.movies["title"] == title][0])
        scores = cosine_similarity(recommender.matrix[index], recommender.matrix).ravel()
        others = np.delete(scores, index)
        query_genres = recommender.movies.loc[index, "genres"]
        summaries.append({
            "query": title,
            "query_genres": query_genres,
            "top_score": round(float(top["content_score"].iloc[0]), 4),
            "movies_tied_at_top_score": int((others >= others.max() - 1e-9).sum()),
            "top10_with_identical_genres": int((top["genres"].map(genre_set) == genre_set(query_genres)).sum()),
            "top10_distinct_genre_sets": int(top["genres"].map(genre_set).nunique()),
        })
        frames.append(top.assign(query=title))
        print(f"\nSimilares a {title}  [géneros: {query_genres}]\n", top)
        print("Resumen:", summaries[-1])

    if frames:
        pd.concat(frames, ignore_index=True)[["query", "movieId", "title", "genres", "content_score"]].to_csv(
            reports / "content_recommendations.csv", index=False)
    (reports / "lab08_summary.json").write_text(
        json.dumps({"popularity": popularity_summary, "content_queries": summaries}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("\nArchivos: reports/popular_top10.csv, reports/content_recommendations.csv, reports/lab08_summary.json")


if __name__ == "__main__":
    main()
