"""Vectorised top-K ranking metrics and beyond-accuracy measures."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def ranking_mask(exclude: sp.csr_matrix | None, shape: tuple[int, int],
                 candidates: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Additive mask for ``top_k``: -inf on seen / non-candidate items, and a
    tiny random jitter elsewhere so a model with flat scores (e.g. CF for a
    user with no history) is not silently rewarded by item-id order."""
    m = (np.random.default_rng(seed).random(shape) * 1e-6).astype(np.float32)
    if exclude is not None:
        m[exclude.nonzero()] = -np.inf
    if candidates is not None:
        m[:, ~candidates] = -np.inf
    return m


def top_k(scores: np.ndarray, exclude: sp.csr_matrix | None, k: int,
          candidates: np.ndarray | None = None, seed: int = 0,
          mask: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k best items per row, best first.

    ``exclude`` (rows aligned with ``scores``) masks already-seen items;
    ``candidates`` (bool, n_items) restricts the ranking to a sub-catalogue.
    A precomputed ``mask`` from ``ranking_mask`` can be passed instead to save
    work when the same users are ranked many times (gate tuning).
    """
    if mask is None:
        mask = ranking_mask(exclude, scores.shape, candidates, seed)
    s = scores.astype(np.float32) + mask
    part = np.argpartition(-s, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def ranking_metrics(topk_idx: np.ndarray, relevant: sp.csr_matrix) -> dict[str, np.ndarray]:
    """Per-user Precision@K, Recall@K, NDCG@K and HitRate@K (binary relevance)."""
    k = topk_idx.shape[1]
    rel = relevant.toarray().astype(bool) if sp.issparse(relevant) else relevant
    hits = np.take_along_axis(rel, topk_idx, axis=1)
    n_rel = rel.sum(axis=1)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = hits @ disc
    idcg = np.cumsum(disc)[np.minimum(n_rel, k) - 1]
    return {
        "precision": hits.sum(1) / k,
        "recall": hits.sum(1) / n_rel,
        "ndcg": dcg / idcg,
        "hit": hits.any(1).astype(float),
    }


def beyond_accuracy(topk_idx: np.ndarray, item_counts: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    """Catalogue coverage, novelty (mean self-information), intra-list diversity and
    mean popularity percentile of the recommended items."""
    n_items = len(item_counts)
    p = (item_counts + 1) / (item_counts.sum() + n_items)
    k = topk_idx.shape[1]
    s = sim[topk_idx[:, :, None], topk_idx[:, None, :]]          # users x k x k
    ild = 1 - (s.sum(axis=(1, 2)) - np.trace(s, axis1=1, axis2=2)) / (k * (k - 1))
    pop_rank = item_counts.argsort().argsort() / (n_items - 1)    # 0 = least popular
    return {
        "coverage": len(np.unique(topk_idx)) / n_items,
        "novelty": float(-np.log2(p[topk_idx]).mean()),
        "ild": float(ild.mean()),
        "pop_pct": float(pop_rank[topk_idx].mean()),
    }
