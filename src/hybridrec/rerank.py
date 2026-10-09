"""Diversity / novelty re-ranking with Maximal Marginal Relevance (Carbonell & Goldstein 1998)."""
from __future__ import annotations

import numpy as np


def mmr_rerank(scores: np.ndarray, candidates: np.ndarray, sim: np.ndarray, item_counts: np.ndarray,
               k: int = 10, lam: float = 0.8, novelty_weight: float = 0.0) -> np.ndarray:
    """Greedy re-rank of each user's candidate list.

    scores      users x n_items model scores
    candidates  users x C item indices (the model's top-C, best first)
    sim         n_items x n_items content similarity in [-1, 1]

    At each step pick argmax over remaining candidates of
        lam * relevance + novelty_weight * novelty - (1 - lam) * max_sim_to_already_picked
    where relevance is the model score min-max scaled within the candidate list
    and novelty is the item's scaled self-information -log p(i).
    """
    n_users, C = candidates.shape
    rel = np.take_along_axis(scores, candidates, axis=1).astype(np.float64)
    lo, hi = rel.min(1, keepdims=True), rel.max(1, keepdims=True)
    rel = (rel - lo) / np.maximum(hi - lo, 1e-12)

    p = (item_counts + 1) / (item_counts.sum() + len(item_counts))
    nov = -np.log2(p)
    nov = (nov - nov.min()) / max(nov.max() - nov.min(), 1e-12)
    base = lam * rel + novelty_weight * nov[candidates]

    S = sim[candidates[:, :, None], candidates[:, None, :]]          # users x C x C
    rows = np.arange(n_users)
    chosen = np.zeros((n_users, k), dtype=np.int64)
    max_sim = np.zeros((n_users, C))
    taken = np.zeros((n_users, C), dtype=bool)
    for step in range(k):
        obj = base - (1 - lam) * max_sim if step else base.copy()
        obj[taken] = -np.inf
        j = obj.argmax(1)
        chosen[:, step] = j
        taken[rows, j] = True
        max_sim = np.maximum(max_sim, S[rows, j, :])
    return np.take_along_axis(candidates, chosen, axis=1)
