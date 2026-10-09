"""Evaluation protocol: per-user metrics on the full catalogue and on the cold-item sub-catalogue."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .features import TrainData
from .metrics import ranking_metrics, top_k

COLD = 5  # "cold" = fewer than this many training interactions
USER_BUCKETS = [(0, 0, "0"), (1, 2, "1-2"), (3, 4, "3-4"),
                (5, 49, "5-49"), (50, 199, "50-199"), (200, 10**9, "200+")]


def user_bucket(n: np.ndarray) -> np.ndarray:
    out = np.empty(len(n), dtype=object)
    for lo, hi, label in USER_BUCKETS:
        out[(n >= lo) & (n <= hi)] = label
    return out


@dataclass
class EvalSet:
    users: np.ndarray            # dense user indices
    relevant: sp.csr_matrix      # len(users) x n_items, held-out items rated >= min_rating
    exclude: sp.csr_matrix       # len(users) x n_items, training interactions
    candidates: np.ndarray | None = None  # bool mask over items, None = full catalogue


def make_eval_set(data: TrainData, test: pd.DataFrame, min_rating: int = 4,
                  candidates: np.ndarray | None = None, horizon: int | None = None) -> EvalSet:
    """Relevant = held-out items rated >= ``min_rating``.

    ``horizon`` keeps only each user's next ``horizon`` held-out interactions
    (chronologically, among candidate items) before picking the positives, so
    every user is judged on the same "what do they watch next" window. Without
    it, cold users -- whose whole future history is held out -- have relevant
    sets ~10x larger than warm users', which inflates popularity baselines.
    """
    if candidates is not None:
        test = test[candidates[data.item_ids.get_indexer(test["item_id"])]]
    if horizon:
        test = test.sort_values("ts", kind="stable").groupby("user_id").head(horizon)
    pos = test[test["rating"] >= min_rating]
    u, i = data.encode(pos)
    users = np.unique(u)
    row = np.searchsorted(users, u)
    rel = sp.csr_matrix((np.ones(len(u), dtype=bool), (row, i)), shape=(len(users), data.n_items))
    return EvalSet(users, rel, data.R[users], candidates)


def score_batched(score_fn, users: np.ndarray, batch: int = 2048) -> np.ndarray:
    return np.vstack([score_fn(users[s:s + batch]) for s in range(0, len(users), batch)])


def evaluate_scores(scores: np.ndarray, es: EvalSet, k: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    idx = top_k(scores, es.exclude, k, es.candidates)
    m = ranking_metrics(idx, es.relevant)
    return pd.DataFrame({"user": es.users, **m}), idx


def summarise(per_user: pd.DataFrame, by: str | None = None,
              metrics=("precision", "recall", "ndcg", "hit")) -> pd.DataFrame:
    """Mean and standard error of each metric, optionally per slice."""
    g = per_user.groupby(by, sort=False) if by else per_user.assign(_all="all").groupby("_all")
    mean = g[list(metrics)].mean()
    se = g[list(metrics)].sem().add_suffix("_se")
    return pd.concat([mean, se, g.size().rename("n_users")], axis=1)
