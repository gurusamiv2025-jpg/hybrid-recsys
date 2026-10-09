"""Fast unit tests on synthetic data (no download needed): python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridrec.metrics import ranking_metrics, top_k  # noqa: E402
from hybridrec.models import AdaptiveHybrid, GateParams, gate_weights  # noqa: E402
from hybridrec.rerank import mmr_rerank  # noqa: E402
from hybridrec.split import make_split  # noqa: E402


def test_metrics_hand_computed():
    # one user, relevant items {1, 3}; ranking [1, 2, 3]
    rel = sp.csr_matrix(np.array([[0, 1, 0, 1, 0]], dtype=bool))
    m = ranking_metrics(np.array([[1, 2, 3]]), rel)
    assert m["precision"][0] == pytest.approx(2 / 3)
    assert m["recall"][0] == pytest.approx(1.0)
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    assert m["ndcg"][0] == pytest.approx(dcg / idcg)
    assert m["hit"][0] == 1.0


def test_top_k_excludes_seen_and_respects_candidates():
    scores = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    seen = sp.csr_matrix(np.array([[1, 0, 0, 0, 0]]))
    assert top_k(scores, seen, 2).tolist() == [[1, 2]]
    cand = np.array([False, False, False, True, True])
    assert top_k(scores, seen, 2, candidates=cand).tolist() == [[3, 4]]


def _synthetic_ratings(n_users=200, n_items=120, per_user=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for t, i in enumerate(rng.choice(n_items, per_user, replace=False)):
            rows.append((u, int(i), int(rng.integers(1, 6)), u * 1000 + t))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "ts"])


def test_split_cold_start_properties():
    df = _synthetic_ratings()
    s = make_split(df, cold_user_frac=0.2, cold_item_frac=0.1, max_cold=4,
                   min_item_ratings_for_cold=20, seed=1)
    assert len(s.train) + len(s.test) == len(df)
    assert not set(map(tuple, s.train[["user_id", "item_id"]].values)) & \
        set(map(tuple, s.test[["user_id", "item_id"]].values))
    uc = s.train.groupby("user_id").size().reindex(s.cold_users, fill_value=0)
    ic = s.train.groupby("item_id").size().reindex(s.cold_items, fill_value=0)
    assert (uc <= 4).all() and (ic <= 4).all()
    # cold users keep their *earliest* interactions
    for u in s.cold_users[:10]:
        tr, te = s.train[s.train.user_id == u], s.test[s.test.user_id == u]
        if len(tr) and len(te):
            assert tr.ts.max() < te.ts.max()


def test_gate_shifts_weight_with_history():
    h = AdaptiveHybrid(None, None, None, GateParams(k_user=10, k_item=10, k_content=2, rho=0.0))
    h.user_counts = np.array([0, 2, 50, 500], dtype=np.float32)
    h.item_counts = np.array([0, 1000], dtype=np.float32)
    _, w_cf, w_cb, w_pop = h.gate(np.arange(4))
    warm_item = 1
    assert np.allclose(w_cf + w_cb + w_pop, 1)
    assert w_cf[0, warm_item] == 0 and w_pop[0, warm_item] == 1          # brand-new user
    assert np.all(np.diff(w_cf[:, warm_item]) > 0)                       # CF trust grows
    assert w_cb[1, warm_item] > w_cf[1, warm_item]                       # sparse user leans on content
    assert w_cf[3, 0] == 0                                               # unseen item: no learned-CF weight


def test_gate_nests_fixed_blend():
    p = GateParams(k_user=1e-6, k_item=1e-6, k_content=1e-6, rho=1.0, cf_cap=0.5)
    _, w_cf, w_cb, w_pop = gate_weights(np.array([[3.0], [300.0]]), np.array([[0.0, 2.0, 500.0]]), p)
    assert np.allclose(w_cf, 0.5) and np.allclose(w_cb, 0.5) and np.allclose(w_pop, 0.0, atol=1e-6)


def test_mmr_lambda_one_is_identity():
    rng = np.random.default_rng(0)
    scores = rng.random((3, 30))
    cand = np.argsort(-scores, axis=1)[:, :15]
    sim = rng.random((30, 30))
    out = mmr_rerank(scores, cand, sim, np.ones(30), k=10, lam=1.0)
    assert (out == cand[:, :10]).all()


def test_combine_matches_reference_formula():
    from hybridrec.models import zscore_rows
    rng = np.random.default_rng(0)
    n_users, n_items = 6, 9
    h = AdaptiveHybrid(None, None, None, GateParams(k_user=4, k_item=7, k_content=3, rho=0.3,
                                                    sharpness=2.5, cf_cap=0.8))
    h.user_counts = np.array([0, 1, 3, 6, 40, 900], dtype=np.float32)
    h.item_counts = rng.integers(0, 50, n_items).astype(np.float32)
    comp = {k: rng.normal(size=(n_users, n_items)).astype(np.float32) for k in ("cf", "cf_mapped", "cb", "pop")}
    users = np.arange(n_users)
    beta, w_cf, w_cb, w_pop = h.gate(users)
    cf = zscore_rows(beta * comp["cf"] + (1 - beta) * comp["cf_mapped"])
    ref = w_cf * cf + w_cb * comp["cb"] + w_pop * comp["pop"]
    assert np.allclose(h.combine(comp, users), ref, atol=1e-5)
