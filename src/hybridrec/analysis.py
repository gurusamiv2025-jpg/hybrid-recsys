"""Aggregation across seeds, figures, and failure-case analysis."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import GENRES, MovieLens
from .evaluate import COLD, USER_BUCKETS
from .experiment import SeedResult
from .models import GateParams, gate_weights

# Reference categorical palette, fixed order (hero series first); baselines muted.
COLORS = {
    "Adaptive hybrid": "#2a78d6", "ALS (CF only)": "#eb6834", "Content only": "#1baf7a",
    "Fixed 50/50": "#eda100", "DemoPop": "#e87ba4", "Popularity": "#8a8984",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_ORDER = list(COLORS)[::-1]  # baselines first in tables, hybrid last


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


# --------------------------------------------------------------------------- #
# Tables                                                                      #
# --------------------------------------------------------------------------- #
def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return format(v, floatfmt)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def headline(results: list[SeedResult], k: int) -> pd.DataFrame:
    """Mean ± std across seeds of per-slice means (± SE if only one seed)."""
    metrics = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}"]
    per_seed = []
    for r in results:
        pu = r.per_user
        both = pd.concat([pu.assign(slice="all users"), pu])
        g = both.groupby(["model", "slice"])[metrics].mean()
        ci = r.per_user_ci.groupby("model")[["precision", "recall", "ndcg"]].mean()
        ci.columns = metrics
        ci["slice"] = "cold items (<5)"
        g = pd.concat([g, ci.reset_index().set_index(["model", "slice"])])
        per_seed.append(g.assign(seed=r.seed))
    allr = pd.concat(per_seed)
    mean = allr.groupby(level=[0, 1])[metrics].mean()
    if len(results) > 1:
        spread = allr.groupby(level=[0, 1])[metrics].std()
    else:  # standard error over users
        r = results[0]
        both = pd.concat([r.per_user.assign(slice="all users"), r.per_user])
        spread = both.groupby(["model", "slice"])[metrics].sem()
        ci = r.per_user_ci.groupby("model")[["precision", "recall", "ndcg"]].sem()
        ci.columns = metrics
        ci["slice"] = "cold items (<5)"
        spread = pd.concat([spread, ci.reset_index().set_index(["model", "slice"])])
    out = mean.copy()
    for m in metrics:
        out[m] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(mean[m], spread.loc[mean.index, m])]
    out = out.reset_index()
    slice_order = ["all users", "warm users (>=5)", "cold users (<5)", "cold items (<5)"]
    out["_s"] = out["slice"].map(slice_order.index)
    out["_m"] = out["model"].map(MODEL_ORDER.index)
    return out.sort_values(["_s", "_m"]).drop(columns=["_s", "_m"]).reset_index(drop=True)


def lift_table(results: list[SeedResult], k: int) -> pd.DataFrame:
    """Relative NDCG lift of each model over the global-popularity baseline, per slice."""
    rows = []
    for r in results:
        pu = pd.concat([r.per_user.assign(slice="all users"), r.per_user])
        g = pu.groupby(["slice", "model"])[f"ndcg@{k}"].mean().unstack()
        ci = r.per_user_ci.groupby("model")["ndcg"].mean().rename("cold items (<5)")
        g = pd.concat([g, ci.to_frame().T])
        rows.append(g.div(g["Popularity"], axis=0) - 1)
    lift = sum(rows) / len(rows)
    lift = lift[[m for m in MODEL_ORDER if m != "Popularity"]]
    return lift.map(lambda v: f"{v:+.0%}").reset_index().rename(columns={"index": "slice"})


def bucket_table(results: list[SeedResult], k: int) -> pd.DataFrame:
    g = pd.concat([r.per_user for r in results]).groupby(["bucket", "model"])[f"ndcg@{k}"].mean()
    t = g.unstack()[MODEL_ORDER]
    order = [b[2] for b in USER_BUCKETS]
    t = t.reindex([b for b in order if b in t.index])
    n = results[0].per_user.query("model == 'Popularity'").groupby("bucket").size()
    t.insert(0, "users (seed 0)", n.reindex(t.index).astype(int))
    return t.reset_index().rename(columns={"bucket": "train interactions"})


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
def plot_buckets(results: list[SeedResult], k: int, out: Path) -> None:
    t = bucket_table(results, k).set_index("train interactions").drop(columns="users (seed 0)")
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for m in MODEL_ORDER:
        hero = m == "Adaptive hybrid"
        ax.plot(x, t[m], color=COLORS[m], lw=2.6 if hero else 1.8,
                ls="--" if m == "Popularity" else "-", marker="o", ms=5 if hero else 4,
                label=m, zorder=3 if hero else 2)
    ax.set_xticks(x, t.index)
    ax.axvspan(-0.4, 2.5, color="#f0efec", zorder=0)
    ax.text(1, ax.get_ylim()[1] * 0.97, "cold-start users", ha="center", va="top", color=INK2, fontsize=9)
    _style(ax, f"NDCG@{k} by amount of user history", "training interactions per user", f"NDCG@{k}")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_gate(gate: GateParams, out: Path) -> None:
    n = np.arange(0, 201, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), dpi=150, sharey=True)
    for ax, n_i, title in [(axes[0], 200, "established item (200 interactions)"),
                           (axes[1], 2, "cold item (2 interactions)")]:
        _, w_cf, w_cb, w_pop = gate_weights(n, np.array(float(n_i)), gate)
        ax.stackplot(n, w_pop, w_cb, w_cf, colors=["#e87ba4", "#1baf7a", "#eb6834"],
                     labels=["demographic popularity", "content profile", "collaborative filtering"],
                     edgecolor="white", linewidth=0.5)
        ax.set_xscale("symlog", linthresh=5)
        ax.set_xticks([0, 1, 2, 5, 10, 20, 50, 100, 200], ["0", "1", "2", "5", "10", "20", "50", "100", "200"])
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 1)
        _style(ax, title, "user's training interactions", "blend weight" if ax is axes[0] else "")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], frameon=False, fontsize=8, ncol=3, loc="lower center")
    fig.suptitle("Tuned gate: " + ", ".join(f"{k}={v:.3g}" for k, v in asdict(gate).items()),
                 x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def plot_mmr(mmr: pd.DataFrame, k: int, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150, sharey=True)
    series = [(0.0, "diversity only", COLORS["Adaptive hybrid"]),
              (0.3, "diversity + novelty (0.3)", COLORS["ALS (CF only)"])]
    for ax, col, label in [(axes[0], "ild", "intra-list diversity"), (axes[1], "coverage", "catalogue coverage")]:
        for nw, name, color in series:
            # lambda = 1 (no re-ranking) is the shared starting point of both curves
            d = mmr[(mmr["novelty_weight"] == nw) | (mmr["lambda"] == 1.0)].sort_values("lambda", ascending=False)
            ax.plot(d[col], d[f"ndcg@{k}"], color=color, lw=2, marker="o", ms=5, label=name)
            for _, r in d.iloc[[-1]].iterrows():
                ax.annotate(f"λ={r['lambda']:.1f}", (r[col], r[f"ndcg@{k}"]), textcoords="offset points",
                            xytext=(-8, -12), fontsize=8, color=INK2)
        first = mmr[mmr["lambda"] == 1.0].iloc[0]
        ax.annotate("λ=1.0 (no re-rank)", (first[col], first[f"ndcg@{k}"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=INK2)
        _style(ax, f"Accuracy vs {label}", label, f"NDCG@{k}" if ax is axes[0] else "")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Failure analysis (first seed, adaptive hybrid)                              #
# --------------------------------------------------------------------------- #
def _genre_matrix(ml: MovieLens, item_ids: pd.Index) -> np.ndarray:
    m = ml.movies.set_index("item_id").loc[item_ids, "genres"]
    return np.array([[g in gs for g in GENRES] for gs in m], dtype=float)


def failure_analysis(ml: MovieLens, r: SeedResult, k: int, out_dir: Path) -> dict[str, str]:
    data, es, es_ci, split = r.data, r.es, r.es_ci, r.split
    hyb = r.models["Adaptive hybrid"]
    pu = r.per_user.query("model == 'Adaptive hybrid'").set_index("user")
    ndcg = pu[f"ndcg@{k}"]
    item_counts = data.item_counts
    pop_pct = item_counts.argsort().argsort() / (len(item_counts) - 1)
    G = _genre_matrix(ml, data.item_ids)
    titles = ml.movies.set_index("item_id").loc[data.item_ids, "title"].to_numpy()
    sections: dict[str, str] = {}

    # --- 1. user-level drivers of failure --------------------------------- #
    rel = es.relevant.toarray()
    rel_pop = (rel * pop_pct).sum(1) / rel.sum(1)           # mean popularity pct of held-out likes
    train_bin = (es.exclude > 0).toarray()
    tg = train_bin @ G
    teg = rel @ G
    drift = 1 - (tg * teg).sum(1) / (np.linalg.norm(tg, axis=1) * np.linalg.norm(teg, axis=1) + 1e-12)
    drift[train_bin.sum(1) == 0] = np.nan
    f = pd.DataFrame({"ndcg": ndcg.loc[es.users].to_numpy(), "n_train": data.user_counts[es.users],
                      "held_out_popularity": rel_pop, "genre_drift": drift, "n_relevant": rel.sum(1)})
    f["slice"] = np.where(f.n_train < COLD, "cold", "warm")
    f["niche quartile"] = pd.qcut(f.held_out_popularity, 4,
                                  labels=["Q1 (most niche)", "Q2", "Q3", "Q4 (most mainstream)"])
    t1 = f.pivot_table(index="niche quartile", columns="slice", values="ndcg", aggfunc="mean", observed=True)
    t1["share of zero-hit users (all)"] = f.groupby("niche quartile", observed=True)["ndcg"].apply(lambda s: (s == 0).mean())
    corr = []
    for col in ["n_train", "held_out_popularity", "genre_drift", "n_relevant"]:
        for sl in ["cold", "warm"]:
            s = f[f.slice == sl].dropna(subset=[col])
            rho, p = spearmanr(s[col], s.ndcg)
            corr.append({"feature": col, "slice": sl, "spearman_rho": rho, "p_value": p})
    sections["niche"] = md_table(t1.reset_index())
    sections["correlations"] = md_table(pd.DataFrame(corr), floatfmt=".3g")

    # --- 2. personalisation collapse for zero-history users ---------------- #
    topk = r.topk["Adaptive hybrid"]
    n_u = data.user_counts[es.users]

    def mean_jaccard(rows: np.ndarray, sample: int = 300) -> float:
        rng = np.random.default_rng(0)
        rows = rng.choice(rows, size=min(sample, len(rows)), replace=False)
        sets = [set(topk[i]) for i in rows]
        js = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
        return float(np.mean(js))
    pers = pd.DataFrame([{"users": lbl, "mean pairwise Jaccard of top-10": mean_jaccard(np.where(m)[0]),
                          "distinct items recommended": len(np.unique(topk[m])),
                          f"NDCG@{k}": ndcg.loc[es.users[m]].mean()}
                         for lbl, m in [("0 interactions", n_u == 0), ("1-4 interactions", (n_u > 0) & (n_u < COLD)),
                                        ("warm (>=5)", n_u >= COLD)]])
    sections["personalisation"] = md_table(pers)

    # --- 3. popularity bias: what we recommend vs what users go on to like -- #
    bias = pd.DataFrame([{"users": lbl,
                          "popularity pct of held-out likes": np.nanmean(rel_pop[m]),
                          "popularity pct of hybrid top-10": pop_pct[topk[m]].mean(),
                          "popularity pct of ALS top-10": pop_pct[r.topk["ALS (CF only)"][m]].mean()}
                         for lbl, m in [("cold (<5)", n_u < COLD), ("warm (>=5)", n_u >= COLD)]])
    sections["popularity_bias"] = md_table(bias)

    # --- 4. cold items: which ones the hybrid still cannot rank ------------ #
    cand = es_ci.candidates
    s_ci = hyb.score(es_ci.users).astype(np.float64)
    s_ci[es_ci.exclude.nonzero()] = -np.inf
    s_ci[:, ~cand] = -np.inf
    ranks = (-s_ci).argsort(1).argsort(1)                     # 0 = top of the cold list
    rr, ii = es_ci.relevant.nonzero()
    has_genome = (data.item_blocks["genome"].getnnz(axis=1) > 0 if "genome" in data.item_blocks
                  else np.zeros(data.n_items, bool))
    pairs = pd.DataFrame({"item": ii, "rank": ranks[rr, ii]})
    pairs["hit@10"] = pairs["rank"] < k
    pairs["n_train_item"] = item_counts[ii]
    pairs["has_genome"] = has_genome[ii]
    pairs["n_genres"] = G[ii].sum(1)
    pairs["item_bucket"] = pd.cut(pairs.n_train_item, [-1, 0, 2, 4], labels=["0", "1-2", "3-4"])
    t4 = pairs.groupby(["item_bucket", "has_genome"], observed=True).agg(
        pairs=("hit@10", "size"), hit_rate_at_10=("hit@10", "mean"), median_rank=("rank", "median")).reset_index()
    sections["cold_items"] = md_table(t4, floatfmt=".3f")

    per_item = pairs.groupby("item").agg(likes=("hit@10", "size"), hit_rate=("hit@10", "mean"),
                                         median_rank=("rank", "median"))
    per_item["title"] = titles[per_item.index]
    per_item["genres"] = ml.movies.set_index("item_id").loc[data.item_ids[per_item.index], "genres"].str.join("|").to_numpy()
    per_item["train_n"] = item_counts[per_item.index]
    per_item["genome"] = has_genome[per_item.index]
    worst = per_item[per_item.likes >= 20].sort_values(["hit_rate", "likes"], ascending=[True, False]).head(8)
    best = per_item[per_item.likes >= 20].sort_values("hit_rate", ascending=False).head(5)
    sections["worst_cold_items"] = md_table(worst.reset_index(drop=True)[
        ["title", "genres", "train_n", "genome", "likes", "hit_rate", "median_rank"]], floatfmt=".2f")
    sections["best_cold_items"] = md_table(best.reset_index(drop=True)[
        ["title", "genres", "train_n", "genome", "likes", "hit_rate", "median_rank"]], floatfmt=".2f")

    # --- 5. concrete examples -------------------------------------------- #
    ex = []
    row_of = {u: i for i, u in enumerate(es.users)}
    train = split.train
    picks = []
    c = f[(f.n_train.between(2, 4)) & (f.ndcg == 0) & (f.n_relevant >= 5)].sort_values("n_relevant", ascending=False)
    if len(c):
        picks.append(("Cold user whose few ratings were misleading", es.users[c.index[0]]))
    w = f[(f.slice == "warm") & (f.ndcg == 0) & (f.n_relevant >= 5)].sort_values("genre_drift", ascending=False)
    if len(w):
        picks.append(("Warm user with strong taste drift", es.users[w.index[0]]))
    for label, u in picks:
        i = row_of[u]
        uid = data.user_ids[u]
        hist = train[train.user_id == uid].sort_values("ts")
        hist_items = data.item_ids.get_indexer(hist.item_id)
        likes = np.where(rel[i])[0]
        likes = likes[np.argsort(-item_counts[likes])][:8]
        hist_str = "; ".join(f"{titles[j]} ({int(rt)}★)" for j, rt in zip(hist_items[-8:], hist.rating.to_numpy()[-8:]))
        ex.append(f"**{label}** (user {uid}, {len(hist)} training ratings, "
                  f"genre drift {drift[i]:.2f}, NDCG@{k} = {ndcg.loc[u]:.2f})\n\n"
                  f"- Training history (last ≤8): {hist_str}\n"
                  f"- Hybrid top-{k}: {'; '.join(titles[j] for j in topk[i])}\n"
                  f"- Actually liked later (sample): {'; '.join(titles[j] for j in likes)}\n")
    sections["examples"] = "\n".join(ex)

    f.to_csv(out_dir / "failure_user_features.csv", index=False)
    return sections
