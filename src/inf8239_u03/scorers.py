from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NO_GENRES = "(no genres listed)"


def popularity_scores(train: pd.DataFrame, items: np.ndarray, quantile: float = 0.80):
    """Puntuaciones de popularidad calculadas solo con entrenamiento.

    Devuelve (cantidad de valoraciones, media suavizada, constante de suavizado m).
    """
    stats = train.groupby("movieId")["rating"].agg(["count", "mean"]).reindex(items)
    count = stats["count"].fillna(0).to_numpy(dtype=float)
    global_mean = float(train["rating"].mean())
    mean = stats["mean"].fillna(global_mean).to_numpy(dtype=float)
    positive = count[count > 0]
    m = float(np.quantile(positive, quantile)) if len(positive) else 1.0
    weighted = (count * mean + m * global_mean) / (count + m)
    return count, weighted, m


def build_genre_matrix(movies: pd.DataFrame):
    """TF-IDF de los géneros (metadatos del catálogo, no de interacciones) y fila de cada película."""
    table = movies.drop_duplicates("movieId").reset_index(drop=True)
    text = table["genres"].fillna(NO_GENRES).str.replace("|", " ", regex=False)
    matrix = TfidfVectorizer().fit_transform(text)
    row_of = {int(movie_id): index for index, movie_id in enumerate(table["movieId"])}
    return matrix, row_of


def genre_sets_of(movies: pd.DataFrame) -> dict[int, frozenset]:
    table = movies.drop_duplicates("movieId")
    return {
        int(movie_id): frozenset(str(genres).split("|"))
        for movie_id, genres in zip(table["movieId"], table["genres"].fillna(NO_GENRES))
    }


def content_profile_scores(history_rows: list[int], ratings: np.ndarray, matrix, universe_rows: list[int]) -> np.ndarray:
    """Similitud coseno entre el perfil de géneros del usuario y cada ítem del universo.

    El perfil promedia los géneros de lo valorado, con más peso a lo mejor valorado.
    """
    weights = np.clip(np.asarray(ratings, dtype=float) - 2.5, 0.1, None)
    profile = np.asarray(matrix[history_rows].multiply(weights[:, None]).sum(axis=0)) / weights.sum()
    return cosine_similarity(profile, matrix[universe_rows]).ravel()


def mf_scores(model, user_id: int) -> np.ndarray:
    """Puntuación de factorización para todos los ítems del modelo, en un solo producto matricial."""
    return model.global_mean + model.item_factors @ model.user_factors[model.user_index[user_id]]


def _min_max(values: np.ndarray) -> np.ndarray:
    return (values - values.min()) / max(values.max() - values.min(), 1e-9)


def hybrid_rerank(mf_vector: np.ndarray, content_vector: np.ndarray, seen: np.ndarray,
                  alpha: float, pool: int = 100) -> np.ndarray:
    """Candidatos: Top-`pool` de la factorización (sin vistos); orden final: mezcla normalizada.

    alpha es el peso colaborativo; (1 - alpha) es el peso del contenido.
    """
    masked = np.asarray(mf_vector, dtype=float).copy()
    masked[np.asarray(seen, dtype=int)] = -np.inf
    candidates = np.argsort(-masked, kind="stable")[:pool]
    blended = alpha * _min_max(mf_vector[candidates]) + (1 - alpha) * _min_max(content_vector[candidates])
    return candidates[np.argsort(-blended, kind="stable")]
