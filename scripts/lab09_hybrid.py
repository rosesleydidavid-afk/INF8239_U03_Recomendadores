from __future__ import annotations

import argparse
import json
import platform
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from inf8239_u03.config import MOVIELENS_DIR, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.evaluation import (
    coverage_of_lists,
    head_share,
    intra_list_diversity,
    pareto_flags,
    summarize_ranks,
    target_rank,
    top_k,
)
from inf8239_u03.recommenders import MatrixFactorization, temporal_leave_one_out
from inf8239_u03.scorers import (
    build_genre_matrix,
    content_profile_scores,
    genre_sets_of,
    hybrid_rerank,
    mf_scores,
    popularity_scores,
)

K = 10
POOL = 100
ONBOARDING = 5
EMPTY = np.array([], dtype=int)

parser = argparse.ArgumentParser()
parser.add_argument("--factors", type=int, nargs="+", default=[10, 20, 40])
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--alphas", "--alpha", dest="alphas", type=float, nargs="+", default=[0.25, 0.75])
parser.add_argument("--max-users", type=int, default=0, help="usa solo N usuarios (depuración rápida)")
parser.add_argument("--quick", action="store_true", help="prueba rápida: 1 configuración, 1 semilla, 2 épocas, 150 usuarios")
args = parser.parse_args()
if args.quick:
    args.factors, args.seeds, args.epochs, args.max_users = [10], [42], 2, 150

reports = ROOT / "reports"
reports.mkdir(exist_ok=True)

# 1. Datos y corte temporal: la última valoración de cada usuario queda reservada para evaluar.
ratings, movies = load_movielens(MOVIELENS_DIR)
if args.max_users:
    ratings = ratings[ratings["userId"].isin(sorted(ratings["userId"].unique())[: args.max_users])]
train, test = temporal_leave_one_out(ratings)
last_timestamp_ties = ratings.groupby("userId")["timestamp"].agg(lambda values: int((values == values.max()).sum()))
items = np.array(sorted(train["movieId"].unique()))
item_pos = {int(movie): index for index, movie in enumerate(items)}
n_items = len(items)
evaluable = test[test["movieId"].isin(item_pos)].reset_index(drop=True)
users = [int(user) for user in evaluable["userId"]]
target = {int(row.userId): item_pos[int(row.movieId)] for row in evaluable.itertuples()}
history = {int(user): group for user, group in train.groupby("userId")}
seen_idx = {user: np.array([item_pos[int(m)] for m in group["movieId"]]) for user, group in history.items()}
audit = {
    "users": int(ratings["userId"].nunique()),
    "evaluable_users": len(users),
    "users_with_cold_item_target": int(len(test) - len(evaluable)),
    "users_with_tied_last_timestamp": int((last_timestamp_ties > 1).sum()),
    "train_ratings": len(train),
    "candidate_universe_items": int(n_items),
    "k": K,
    "hybrid_candidate_pool": POOL,
    "tie_break_rule": "a igual puntuación gana el ítem de menor índice",
}

# 2. Señales sin aprendizaje: popularidad y contenido (solo con entrenamiento).
start = perf_counter()
count, weighted, smoothing_m = popularity_scores(train, items)
popularity_seconds = perf_counter() - start
start = perf_counter()
genre_matrix, row_of = build_genre_matrix(movies)
content_seconds = perf_counter() - start
universe_rows = [row_of[int(movie)] for movie in items]
genre_sets = genre_sets_of(movies)
head_items = {int(items[i]) for i in np.where(count >= np.quantile(count, 0.90))[0]}
content_vector = {}
for user in users:
    group = history[user]
    content_vector[user] = content_profile_scores(
        [row_of[int(m)] for m in group["movieId"]], group["rating"].to_numpy(), genre_matrix, universe_rows)
audit["popularity_smoothing_m"] = smoothing_m
audit["rmse_global_mean"] = float(root_mean_squared_error(
    evaluable["rating"], np.full(len(evaluable), train["rating"].mean())))
audit["rmse_smoothed_item_mean"] = float(root_mean_squared_error(
    evaluable["rating"], [weighted[item_pos[int(m)]] for m in evaluable["movieId"]]))


def evaluate(recommend) -> dict:
    ranks, lists = [], []
    start = perf_counter()
    for user in users:
        top, rank = recommend(user)
        ranks.append(rank)
        lists.append([int(items[i]) for i in top])
    elapsed = perf_counter() - start
    result = summarize_ranks(ranks, K)
    result.update({
        f"coverage@{K}": coverage_of_lists(lists, n_items),
        f"diversity@{K}": intra_list_diversity(lists, genre_sets),
        f"head_share@{K}": head_share(lists, head_items),
        "infer_ms_per_user": 1000 * elapsed / len(users),
    })
    return result


def from_scores(score_function):
    def recommend(user):
        scores = score_function(user)
        return top_k(scores, seen_idx[user], K), target_rank(scores, target[user], seen_idx[user])
    return recommend


def hybrid(model, alpha):
    def recommend(user):
        ranked = hybrid_rerank(mf_scores(model, user), content_vector[user], seen_idx[user], alpha, POOL)
        position = np.where(ranked == target[user])[0]
        return ranked[:K], (int(position[0]) + 1 if len(position) else None)
    return recommend


rows = []


def add(method, factors, alpha, seed, result, train_seconds, memory_kb=np.nan, rmse=np.nan):
    rows.append({"method": method, "factors": factors, "alpha": alpha, "seed": seed, **result,
                 "train_seconds": train_seconds, "factor_memory_kb": memory_kb, "rmse": rmse})


# 3. Comparación: popularidad, contenido, azar, factorización e híbrido, con el mismo corte y universo.
add("popularity_count", np.nan, np.nan, np.nan, evaluate(from_scores(lambda user: count)), popularity_seconds)
add("popularity_weighted", np.nan, np.nan, np.nan, evaluate(from_scores(lambda user: weighted)), popularity_seconds)
add("content_profile", np.nan, np.nan, np.nan,
    evaluate(from_scores(lambda user: content_vector[user])), content_seconds)
for seed in args.seeds:
    rng = np.random.default_rng(seed)
    add("random", np.nan, np.nan, seed, evaluate(from_scores(lambda user, rng=rng: rng.random(n_items))), 0.0)

profile_factors = 20 if 20 in args.factors else args.factors[0]
profile_model = None
for factors in args.factors:
    for seed in args.seeds:
        start = perf_counter()
        model = MatrixFactorization(factors=factors, seed=seed).fit(train, epochs=args.epochs)
        fit_seconds = perf_counter() - start
        assert [int(m) for m in model.items] == items.tolist(), "El universo del modelo no coincide con el del experimento"
        memory_kb = (model.user_factors.nbytes + model.item_factors.nbytes) / 1024
        predictions = [model.predict(int(r.userId), int(r.movieId)) for r in evaluable.itertuples()]
        rmse = float(root_mean_squared_error(evaluable["rating"], predictions))
        add("mf", factors, np.nan, seed, evaluate(from_scores(lambda user, model=model: mf_scores(model, user))),
            fit_seconds, memory_kb, rmse)
        for alpha in args.alphas:
            add("hybrid", factors, alpha, seed, evaluate(hybrid(model, alpha)),
                fit_seconds + content_seconds, memory_kb, rmse)
        if profile_model is None and factors == profile_factors:
            profile_model = model
        print(f"factores={factors} semilla={seed} ajuste={fit_seconds:.1f}s rmse={rmse:.3f}", flush=True)

runs = pd.DataFrame(rows)
runs.to_csv(reports / "lab09_runs.csv", index=False)
metric_columns = [f"hit_rate@{K}", f"ndcg@{K}", "mrr", f"coverage@{K}", f"diversity@{K}", f"head_share@{K}",
                  "infer_ms_per_user", "train_seconds", "factor_memory_kb", "rmse"]
keys = ["method", "factors", "alpha"]
summary = runs.groupby(keys, dropna=False)[metric_columns].mean().reset_index()
summary[f"hit_rate@{K}_std"] = runs.groupby(keys, dropna=False)[f"hit_rate@{K}"].std(ddof=0).to_numpy()
summary["runs"] = runs.groupby(keys, dropna=False).size().to_numpy()
# El azar es un piso de referencia, no un candidato: se excluye de la frontera de Pareto.
candidates = summary["method"].ne("random")
summary["is_pareto"] = False
summary.loc[candidates, "is_pareto"] = pareto_flags(summary[candidates], f"hit_rate@{K}", "train_seconds")
summary = summary.sort_values(f"hit_rate@{K}", ascending=False).reset_index(drop=True)
summary.to_csv(reports / "lab09_summary.csv", index=False)

figure, axis = plt.subplots(figsize=(9, 6))
for _, row in summary.iterrows():
    label = row["method"] + ("" if np.isnan(row["factors"]) else f" f{int(row['factors'])}") + (
        "" if np.isnan(row["alpha"]) else f" a{row['alpha']:g}")
    x = max(row["train_seconds"], 1e-3)
    axis.scatter(x, row[f"hit_rate@{K}"], color="#0f7c7b" if row["is_pareto"] else "#94a3b8", s=45)
    axis.annotate(label, (x, row[f"hit_rate@{K}"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
axis.set_xscale("log")
axis.set_xlabel("Tiempo de ajuste (s, escala logarítmica)")
axis.set_ylabel(f"HitRate@{K} (promedio de semillas)")
axis.set_title("Calidad y costo: frontera de Pareto (verde = no dominado)")
figure.tight_layout()
figure.savefig(reports / "pareto_hit_vs_time.png", dpi=170)
plt.close(figure)

# 4. Usuarios fríos: se retira por completo a un 10 % de usuarios del entrenamiento.
rng = np.random.default_rng(42)
cold_users = {int(u) for u in rng.choice(users, size=max(1, round(0.10 * len(users))), replace=False)}
train_warm = train[~train["userId"].isin(cold_users)]
items_w = np.array(sorted(train_warm["movieId"].unique()))
pos_w = {int(movie): index for index, movie in enumerate(items_w)}
count_w, _, _ = popularity_scores(train_warm, items_w)
rows_w = [row_of[int(movie)] for movie in items_w]
cold_ranks = {"fallback_0_ratings": [], "popularity_after_5_ratings": [],
              "content_after_5_ratings": [], "random": []}
for user in sorted(cold_users):
    movie = int(evaluable.loc[evaluable["userId"].eq(user), "movieId"].iloc[0])
    if movie not in pos_w:
        continue
    group = history[user].head(ONBOARDING)
    seen = np.array([pos_w[int(m)] for m in group["movieId"] if int(m) in pos_w], dtype=int)
    cold_ranks["fallback_0_ratings"].append(target_rank(count_w, pos_w[movie], EMPTY))
    cold_ranks["popularity_after_5_ratings"].append(target_rank(count_w, pos_w[movie], seen))
    content = content_profile_scores([row_of[int(m)] for m in group["movieId"]], group["rating"].to_numpy(),
                                     genre_matrix, rows_w)
    cold_ranks["content_after_5_ratings"].append(target_rank(content, pos_w[movie], seen))
    cold_ranks["random"].append(target_rank(rng.random(len(items_w)), pos_w[movie], seen))
cold_start = {name: summarize_ranks(values, K) for name, values in cold_ranks.items()}
cold_start["note"] = ("Usuarios retirados por completo del entrenamiento. La factorización no puede puntuarlos; "
                      "se compara el respaldo de popularidad con un perfil de contenido de 5 valoraciones.")

# 5. Tres perfiles: historial amplio, historial pequeño y usuario nuevo.
titles = movies.drop_duplicates("movieId").set_index("movieId")
sizes = pd.Series({user: len(history[user]) for user in users})
profile_rows = []
profile_lists = {}
for name, user in (("historial_amplio", int(sizes.idxmax())), ("historial_pequeno", int(sizes.idxmin()))):
    for alpha_profile in args.alphas:
        ranked = hybrid_rerank(mf_scores(profile_model, user), content_vector[user], seen_idx[user],
                               alpha_profile, POOL)[:K]
        assert not set(ranked.tolist()) & set(seen_idx[user].tolist()), "La lista contiene ítems ya vistos"
        profile_lists[(name, alpha_profile)] = {int(i) for i in ranked}
        for position, item in enumerate(ranked, start=1):
            profile_rows.append({"profile": name, "user_id": user, "history_size": int(sizes[user]),
                                 "method": f"híbrido alpha={alpha_profile:g}", "personalized": True,
                                 "rank": position, "movieId": int(items[item]),
                                 "title": titles.loc[int(items[item]), "title"],
                                 "genres": titles.loc[int(items[item]), "genres"]})
first_alpha, last_alpha = args.alphas[0], args.alphas[-1]
profile_overlap = {name: len(profile_lists[(name, first_alpha)] & profile_lists[(name, last_alpha)]) / K
                   for name in ("historial_amplio", "historial_pequeno")}
fallback = top_k(count, EMPTY, K)
new_user_id = int(ratings["userId"].max()) + 1
assert new_user_id not in history and len(fallback) == K
for position, item in enumerate(fallback, start=1):
    profile_rows.append({"profile": "usuario_nuevo", "user_id": new_user_id, "history_size": 0,
                         "method": "respaldo: popularidad por cantidad (no personalizada)", "personalized": False,
                         "rank": position, "movieId": int(items[item]),
                         "title": titles.loc[int(items[item]), "title"], "genres": titles.loc[int(items[item]), "genres"]})
profiles = pd.DataFrame(profile_rows)
profiles.to_csv(reports / "lab09_profiles.csv", index=False)
profiles[profiles["profile"].eq("usuario_nuevo")][["movieId", "title", "genres"]].to_csv(
    reports / "cold_start_fallback.csv", index=False)

audit["cold_start"] = cold_start
audit["profile_model"] = {"factors": profile_factors, "seed": args.seeds[0]}
audit["profile_alphas"] = args.alphas
audit["profile_overlap_between_first_and_last_alpha"] = profile_overlap
audit["arguments"] = {"factors": args.factors, "seeds": args.seeds, "epochs": args.epochs, "alphas": args.alphas,
                      "max_users": args.max_users}
audit["environment"] = {"python": platform.python_version(), "os": platform.platform(),
                        "processor": platform.processor(), "numpy": np.__version__, "pandas": pd.__version__}
(reports / "lab09_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
print("\n=== Auditoría de la partición ===")
print(json.dumps({key: audit[key] for key in audit if key not in ("cold_start", "environment", "arguments")}, indent=2))
print("\n=== Resumen (promedio de semillas) ===")
print(summary.round(4).to_string(index=False))
print("\n=== Usuarios fríos ===")
print(json.dumps(cold_start, indent=2))
print("\n=== Perfiles ===")
print(profiles[["profile", "method", "rank", "title", "genres"]].to_string(index=False))
print(f"\nCoincidencia entre las listas de alpha extremos ({first_alpha:g} vs {last_alpha:g}): {profile_overlap}")
print("\nArchivos en reports/: lab09_runs.csv, lab09_summary.csv, lab09_profiles.csv, lab09_audit.json, "
      "pareto_hit_vs_time.png, cold_start_fallback.csv")
