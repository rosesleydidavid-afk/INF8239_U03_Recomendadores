from __future__ import annotations

import numpy as np
import pandas as pd


def _mask_seen(scores: np.ndarray, seen: np.ndarray) -> np.ndarray:
    masked = np.asarray(scores, dtype=float).copy()
    masked[np.asarray(seen, dtype=int)] = -np.inf
    return masked


def target_rank(scores: np.ndarray, target: int, seen: np.ndarray) -> int:
    """Posición (1 = primero) del ítem objetivo entre los candidatos no vistos.

    Regla de desempate documentada: ante igual puntuación gana el ítem de menor índice.
    """
    masked = _mask_seen(scores, seen)
    value = masked[target]
    higher = int((masked > value).sum())
    tied_before = int(((masked == value) & (np.arange(len(masked)) < target)).sum())
    return 1 + higher + tied_before


def top_k(scores: np.ndarray, seen: np.ndarray, k: int = 10) -> np.ndarray:
    """Índices de los k mejores candidatos no vistos (desempate por menor índice)."""
    masked = _mask_seen(scores, seen)
    return np.argsort(-masked, kind="stable")[:k]


def ndcg_single(rank: int | None, k: int = 10) -> float:
    """NDCG@k cuando hay un único ítem relevante: 1/log2(posición+1) si cae en el Top-k."""
    if rank is None or rank > k:
        return 0.0
    return float(1.0 / np.log2(rank + 1))


def summarize_ranks(ranks: list[int | None], k: int = 10) -> dict:
    n = len(ranks)
    if n == 0:
        return {"users": 0, f"hit_rate@{k}": 0.0, f"ndcg@{k}": 0.0, "mrr": 0.0}
    hits = [rank is not None and rank <= k for rank in ranks]
    reciprocal = [0.0 if rank is None else 1.0 / rank for rank in ranks]
    return {
        "users": n,
        f"hit_rate@{k}": float(np.mean(hits)),
        f"ndcg@{k}": float(np.mean([ndcg_single(rank, k) for rank in ranks])),
        "mrr": float(np.mean(reciprocal)),
    }


def coverage_of_lists(lists: list[list[int]], catalog_size: int) -> float:
    recommended = {item for values in lists for item in values}
    return len(recommended) / catalog_size if catalog_size else 0.0


def head_share(lists: list[list[int]], head_items: set[int]) -> float:
    """Proporción de posiciones recomendadas ocupadas por ítems de la 'cabeza' del catálogo."""
    total = sum(len(values) for values in lists)
    if total == 0:
        return 0.0
    return sum(item in head_items for values in lists for item in values) / total


def intra_list_diversity(lists: list[list[int]], genre_sets: dict[int, frozenset]) -> float:
    """Distancia de Jaccard promedio entre los géneros de los ítems de una misma lista."""
    values = []
    for items in lists:
        distances = []
        for first in range(len(items)):
            for second in range(first + 1, len(items)):
                a, b = genre_sets[items[first]], genre_sets[items[second]]
                union = len(a | b)
                distances.append(0.0 if union == 0 else 1.0 - len(a & b) / union)
        if distances:
            values.append(float(np.mean(distances)))
    return float(np.mean(values)) if values else 0.0


def pareto_flags(df: pd.DataFrame, score: str = "hit_rate@10", cost: str = "train_seconds") -> list[bool]:
    """Una fila está en la frontera si ninguna otra iguala o mejora ambas dimensiones y mejora una."""
    flags = []
    for _, row in df.iterrows():
        dominated = (
            (df[score] >= row[score]) & (df[cost] <= row[cost])
            & ((df[score] > row[score]) | (df[cost] < row[cost]))
        ).any()
        flags.append(not bool(dominated))
    return flags
