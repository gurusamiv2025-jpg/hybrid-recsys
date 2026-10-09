"""End-to-end experiment: split -> tune on validation -> refit -> evaluate on test."""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from .data import MovieLens
from .evaluate import (COLD, EvalSet, evaluate_scores, make_eval_set, score_batched,
                       user_bucket)
from .features import TrainData, build_train_data, combine_blocks
from .metrics import beyond_accuracy, ranking_mask, ranking_metrics, top_k
from .models import (AdaptiveHybrid, ContentProfile, DemographicPopularity, FixedHybrid,
                     GateParams, ImplicitALS, Popularity)
from .rerank import mmr_rerank
from .split import Split, make_split


@dataclass
class Config:
    k: int = 10
    horizon: int | None = 20   # judge each user on their next N held-out interactions
    factors: int = 64
    iterations: int = 12
    als_grid: list[dict] = field(default_factory=lambda: [
        {"alpha": a, "reg": r} for a, r in itertools.product([5.0, 10.0, 20.0], [30.0, 100.0])])
    # Random search over the gate, seeded for reproducibility.
    gate_space: dict[str, tuple[float, float, bool]] = field(default_factory=lambda: {
        # name: (low, high, log-uniform?)
        "k_user": (1.0, 300.0, True), "k_item": (1.0, 300.0, True),
        "k_content": (0.5, 500.0, True), "rho": (0.0, 1.0, False),
        "sharpness": (0.5, 12.0, True), "cf_cap": (0.3, 1.0, False)})
    gate_trials: int = 200
    gate_refine_rounds: int = 2
    gate_refine_per_center: int = 10
    mmr_lams: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
    mmr_candidates: int = 100
    ablations: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "genre": {"genre": 1.0},
        "genre+year+title": {"genre": 1.0, "year": 0.3, "title": 0.3},
        "genome": {"genome": 1.0},
        "all (default)": {"genre": 1.0, "year": 0.3, "title": 0.3, "genome": 2.0},
    })

    def quick(self) -> "Config":
        """A much smaller search for fast iteration / CI."""
        return replace(self, iterations=6, als_grid=[{"alpha": 10.0, "reg": 100.0}],
                       gate_trials=20, gate_refine_rounds=1, gate_refine_per_center=3,
                       mmr_lams=(1.0, 0.8))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_components(data: TrainData, cfg: Config, als_params: dict, seed: int):
    cf = ImplicitALS(factors=cfg.factors, iterations=cfg.iterations, seed=seed, **als_params).fit(data)
    return cf, ContentProfile().fit(data), DemographicPopularity().fit(data)


def _ndcg(scores: np.ndarray, es: EvalSet, k: int) -> np.ndarray:
    return ranking_metrics(top_k(scores, es.exclude, k, es.candidates), es.relevant)["ndcg"]


def _draw(cfg: Config, rng: np.random.Generator, center: GateParams | None = None,
          scale: float = 0.15) -> GateParams:
    """Sample a gate uniformly over the search space (log-uniform where flagged),
    or, given ``center``, perturb it by ``scale`` of each dimension's range."""
    kw = {}
    for name, (lo, hi, logscale) in cfg.gate_space.items():
        f = (np.log, np.exp) if logscale else (lambda x: x, lambda x: x)
        a, b = f[0](lo), f[0](hi)
        x = rng.uniform(a, b) if center is None else             np.clip(f[0](getattr(center, name)) + rng.normal(0, scale * (b - a)), a, b)
        kw[name] = float(f[1](x))
    return GateParams(**kw)


def search_gates(cfg: Config, seed: int, evaluate) -> pd.DataFrame:
    """Random search followed by local refinement around the best candidates.

    Two anchors are always evaluated: the default gate and a gate equivalent to
    the fixed 50/50 blend, so tuning can never do worse (on validation) than
    the naive mix. ``evaluate(GateParams) -> dict`` returns the metrics row.
    """
    rng = np.random.default_rng(seed + 2000)
    trials = [GateParams(), GateParams(k_user=1e-6, k_item=1e-6, k_content=1e-6, rho=1.0, cf_cap=0.5)]
    trials += [_draw(cfg, rng) for _ in range(cfg.gate_trials)]
    rows = [evaluate(p) for p in trials]
    df = pd.DataFrame(rows).sort_values("objective", ascending=False)
    for round_ in range(cfg.gate_refine_rounds):
        centers = [GateParams(**{n: r[n] for n in cfg.gate_space}) for _, r in df.head(5).iterrows()]
        scale = 0.15 / (round_ + 1)
        rows += [evaluate(_draw(cfg, rng, c, scale)) for c in centers for _ in range(cfg.gate_refine_per_center)]
        df = pd.DataFrame(rows).sort_values("objective", ascending=False)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Validation tuning                                                           #
# --------------------------------------------------------------------------- #
def tune(ml: MovieLens, split: Split, cfg: Config, seed: int) -> tuple[dict, GateParams, dict]:
    """Tune ALS and the gate on a validation split carved out of *training* data only."""
    vsplit = make_split(split.train, seed=seed + 1000)
    vdata = build_train_data(ml, vsplit.train)
    cold_items = vdata.item_counts < COLD
    es = make_eval_set(vdata, vsplit.test, candidates=~cold_items, horizon=cfg.horizon)
    es_ci = make_eval_set(vdata, vsplit.test, candidates=cold_items)
    warm = vdata.user_counts[es.users] >= COLD
    es_warm = EvalSet(es.users[warm], es.relevant[warm], es.exclude[warm], es.candidates)

    als_log = []
    best_als, best = None, -1.0
    for params in cfg.als_grid:
        cf = ImplicitALS(factors=cfg.factors, iterations=cfg.iterations, seed=seed, **params).fit(vdata)
        nd = _ndcg(score_batched(cf.score, es_warm.users), es_warm, cfg.k).mean()
        als_log.append({**params, "val_ndcg_warm": nd})
        log(f"  ALS {params}: val NDCG@{cfg.k} (warm) = {nd:.4f}")
        if nd > best:
            best_als, best, best_cf = params, nd, cf

    hybrid = AdaptiveHybrid(best_cf, ContentProfile().fit(vdata),
                            DemographicPopularity().fit(vdata)).fit(vdata)
    comp, comp_ci = hybrid.components(es.users), hybrid.components(es_ci.users)
    cold = ~warm
    mask = ranking_mask(es.exclude, (len(es.users), vdata.n_items), es.candidates)
    mask_ci = ranking_mask(es_ci.exclude, (len(es_ci.users), vdata.n_items), es_ci.candidates)

    def evaluate(p: GateParams) -> dict:
        nd = ranking_metrics(top_k(hybrid.combine(comp, es.users, p), None, cfg.k, mask=mask),
                             es.relevant)["ndcg"]
        nd_ci = ranking_metrics(top_k(hybrid.combine(comp_ci, es_ci.users, p), None, cfg.k, mask=mask_ci),
                                es_ci.relevant)["ndcg"].mean()
        row = {**asdict(p), "warm": nd[warm].mean(), "cold_user": nd[cold].mean(), "cold_item": nd_ci}
        # Macro-average over the three regimes so the gate is not tuned only for
        # the (numerically dominant) warm users.
        row["objective"] = np.mean([row["warm"], row["cold_user"], row["cold_item"]])
        return row

    gate_df = search_gates(cfg, seed, evaluate)
    top = gate_df.iloc[0]
    gate = GateParams(**{name: float(top[name]) for name in cfg.gate_space})
    log(f"  best gate {gate} objective={top.objective:.4f} "
        f"(warm {top.warm:.4f}, cold-user {top.cold_user:.4f}, cold-item {top.cold_item:.4f})")
    return best_als, gate, {"als": als_log, "gate": gate_df}


# --------------------------------------------------------------------------- #
# Test evaluation                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class SeedResult:
    seed: int
    als_params: dict
    gate: GateParams
    per_user: pd.DataFrame          # long: model x user, full catalogue
    per_user_ci: pd.DataFrame       # long: model x user, cold-item catalogue
    beyond: pd.DataFrame            # model x user-slice beyond-accuracy
    mmr: pd.DataFrame
    ablation: pd.DataFrame
    tuning: dict
    # kept for the failure analysis of the first seed
    data: TrainData | None = None
    split: Split | None = None
    models: dict | None = None
    topk: dict | None = None
    es: EvalSet | None = None
    es_ci: EvalSet | None = None


def run_seed(ml: MovieLens, cfg: Config, seed: int, keep_artifacts: bool = False) -> SeedResult:
    log(f"=== seed {seed} ===")
    split = make_split(ml.ratings, seed=seed)
    log(f"split: {len(split.train):,} train / {len(split.test):,} test interactions")
    log("tuning on validation split")
    als_params, gate, tuning = tune(ml, split, cfg, seed)

    log("refitting on full training split")
    data = build_train_data(ml, split.train)
    cf, cb, pop = fit_components(data, cfg, als_params, seed)
    hybrid = AdaptiveHybrid(cf, cb, pop, gate).fit(data)
    models = {
        "Popularity": Popularity().fit(data),
        "DemoPop": pop,
        "ALS (CF only)": cf,
        "Content only": cb,
        "Fixed 50/50": FixedHybrid(cf, cb, 0.5),
        "Adaptive hybrid": hybrid,
    }

    # Two disjoint protocols: user slices are ranked over the warm catalogue
    # (items with >= 5 training interactions); item cold-start is ranked over the
    # cold catalogue only. Mixing them lets the bulk of a cold item's ratings,
    # which the split moved to test, dominate every user's relevant set.
    cold_items = data.item_counts < COLD
    es = make_eval_set(data, split.test, candidates=~cold_items, horizon=cfg.horizon)
    es_ci = make_eval_set(data, split.test, candidates=cold_items)
    n_u = data.user_counts[es.users]
    meta = pd.DataFrame({"user": es.users, "n_train": n_u, "bucket": user_bucket(n_u),
                         "slice": np.where(n_u < COLD, "cold users (<5)", "warm users (>=5)")})

    per_user, per_user_ci, beyond, topks = [], [], [], {}
    for name, model in models.items():
        scores = score_batched(model.score, es.users)
        pu, idx = evaluate_scores(scores, es, k=2 * cfg.k)
        m10 = ranking_metrics(idx[:, :cfg.k], es.relevant)
        pu = meta.assign(model=name, **{f"{k}@{cfg.k}": v for k, v in m10.items()},
                         **{f"{k}@{2 * cfg.k}": pu[k].to_numpy() for k in ("precision", "recall", "ndcg", "hit")})
        per_user.append(pu)
        topks[name] = idx[:, :cfg.k]
        for sl, mask in [("all", np.ones(len(n_u), bool)), ("cold users (<5)", n_u < COLD),
                         ("warm users (>=5)", n_u >= COLD)]:
            beyond.append({"model": name, "slice": sl,
                           **beyond_accuracy(idx[mask, :cfg.k], data.item_counts, cb.K)})

        ci_scores = score_batched(model.score, es_ci.users)
        pci, _ = evaluate_scores(ci_scores, es_ci, k=cfg.k)
        per_user_ci.append(pci.assign(model=name, n_train=data.user_counts[es_ci.users]))
        r = per_user[-1][f"ndcg@{cfg.k}"]
        log(f"  {name:<16} NDCG@{cfg.k} all={r.mean():.4f} "
            f"cold={r[n_u < COLD].mean():.4f} warm={r[n_u >= COLD].mean():.4f} "
            f"cold-items={pci['ndcg'].mean():.4f}")

    log("MMR re-ranking sweep")
    hyb_scores = score_batched(hybrid.score, es.users)
    cand = top_k(hyb_scores, es.exclude, cfg.mmr_candidates, es.candidates)
    mmr_rows = []
    for lam in cfg.mmr_lams:
        for nov_w in (0.0, 0.3):
            if lam == 1.0 and nov_w:
                continue
            idx = np.vstack([mmr_rerank(hyb_scores[s:s + 1000], cand[s:s + 1000], cb.K, data.item_counts,
                                        k=cfg.k, lam=lam, novelty_weight=nov_w)
                             for s in range(0, len(es.users), 1000)])
            m = ranking_metrics(idx, es.relevant)
            ba = beyond_accuracy(idx, data.item_counts, cb.K)
            mmr_rows.append({"lambda": lam, "novelty_weight": nov_w,
                             **{f"{k}@{cfg.k}": v.mean() for k, v in m.items()}, **ba})
            if lam == 0.8 and nov_w == 0.0:
                topks["Adaptive hybrid + MMR"] = idx
    mmr = pd.DataFrame(mmr_rows)

    log("content-feature ablation")
    abl_rows = []
    for label, weights in cfg.ablations.items():
        if "genome" in weights and "genome" not in data.item_blocks:
            continue
        d2 = replace(data, item_x=combine_blocks(data.item_blocks, weights))
        cb2 = ContentProfile().fit(d2)
        cf.fit_mapping(d2)
        h2 = AdaptiveHybrid(cf, cb2, pop, gate).fit(d2)
        nd_cb = _ndcg(score_batched(cb2.score, es.users), es, cfg.k)
        nd_h = _ndcg(score_batched(h2.score, es.users), es, cfg.k)
        nd_ci = _ndcg(score_batched(h2.score, es_ci.users), es_ci, cfg.k)
        cold_u = n_u < COLD
        abl_rows.append({"features": label,
                         "content_only_cold_users": nd_cb[cold_u & (n_u > 0)].mean(),
                         "content_only_warm_users": nd_cb[~cold_u].mean(),
                         "hybrid_cold_users": nd_h[cold_u].mean(),
                         "hybrid_cold_items": nd_ci.mean()})
    cf.fit_mapping(data)  # restore default mapping

    res = SeedResult(seed, als_params, gate, pd.concat(per_user, ignore_index=True),
                     pd.concat(per_user_ci, ignore_index=True), pd.DataFrame(beyond), mmr,
                     pd.DataFrame(abl_rows), tuning)
    if keep_artifacts:
        res.data, res.split, res.models, res.topk, res.es, res.es_ci = data, split, models, topks, es, es_ci
    return res
