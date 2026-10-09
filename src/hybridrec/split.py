"""Train/test split that deliberately manufactures cold-start users and items.

ML-1M only contains users with >= 20 ratings, so cold start has to be simulated:

* Cold users (``cold_user_frac`` of users): only their first ``k_u`` ratings
  (chronologically, ``k_u`` drawn uniformly from 0..max_cold) are visible at
  training time; everything after goes to test. This mimics a new sign-up.
* Cold items (``cold_item_frac`` of items with enough ratings to evaluate):
  only their earliest ``k_i`` training ratings (0..max_cold) are kept; all other
  ratings of the item move to test. This mimics a newly added title.
* Warm users: per-user chronological split, the last ``warm_test_frac`` of each
  user's history is test.

Slices are later defined from *actual* training counts (< 5 = cold), so the
evaluation does not depend on how the split was built.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    cold_users: np.ndarray
    cold_items: np.ndarray


def make_split(
    ratings: pd.DataFrame,
    cold_user_frac: float = 0.15,
    cold_item_frac: float = 0.10,
    warm_test_frac: float = 0.2,
    max_cold: int = 4,
    min_item_ratings_for_cold: int = 20,
    seed: int = 0,
) -> Split:
    rng = np.random.default_rng(seed)
    df = ratings.sort_values(["ts", "user_id", "item_id"], kind="stable").reset_index(drop=True)

    users = df["user_id"].unique()
    cold_users = rng.choice(users, size=int(round(cold_user_frac * len(users))), replace=False)
    k_user = pd.Series(rng.integers(0, max_cold + 1, size=len(cold_users)), index=cold_users)

    # --- user-level chronological assignment -------------------------------
    pos = df.groupby("user_id").cumcount().to_numpy()        # 0-based rank in user's history
    n_user = df.groupby("user_id")["item_id"].transform("size").to_numpy()
    is_cold_user = df["user_id"].isin(cold_users).to_numpy()
    keep_cold = pos < df["user_id"].map(k_user).fillna(-1).to_numpy()
    keep_warm = pos < np.floor((1 - warm_test_frac) * n_user)
    in_train = np.where(is_cold_user, keep_cold, keep_warm)

    # --- item-level cold start ---------------------------------------------
    item_counts = df["item_id"].value_counts()
    eligible = item_counts.index[item_counts >= min_item_ratings_for_cold].to_numpy()
    cold_items = rng.choice(eligible, size=int(round(cold_item_frac * len(eligible))), replace=False)
    k_item = pd.Series(rng.integers(0, max_cold + 1, size=len(cold_items)), index=cold_items)

    is_cold_item = df["item_id"].isin(cold_items).to_numpy()
    # rank of each *training* interaction within its item, oldest first
    train_rank = np.full(len(df), np.iinfo(np.int64).max)
    train_rank[in_train] = df[in_train].groupby("item_id").cumcount().to_numpy()
    keep_item = train_rank < df["item_id"].map(k_item).fillna(-1).to_numpy()
    in_train = in_train & (~is_cold_item | keep_item)

    return Split(
        train=df[in_train].reset_index(drop=True),
        test=df[~in_train].reset_index(drop=True),
        cold_users=np.sort(cold_users),
        cold_items=np.sort(cold_items),
    )
