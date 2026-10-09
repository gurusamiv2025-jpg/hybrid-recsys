"""Item content features, user demographic features and the TrainData container."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import GENRES, MovieLens, tokenize_title

DEFAULT_BLOCK_WEIGHTS = {"genre": 1.0, "year": 0.3, "title": 0.3, "genome": 2.0}


def item_feature_blocks(movies: pd.DataFrame, genome: pd.DataFrame | None) -> dict[str, sp.csr_matrix]:
    """One L2-normalised sparse block per metadata source, rows aligned with ``movies``."""
    blocks: dict[str, sp.csr_matrix] = {}

    g = np.array([[genre in gs for genre in GENRES] for gs in movies["genres"]], dtype=np.float32)
    blocks["genre"] = sp.csr_matrix(normalize(g))

    # Decade one-hot, softened so neighbouring decades are partially similar.
    decades = np.arange(1910, 2010, 10)
    d = np.clip(movies["year"].to_numpy(), decades[0], decades[-1])[:, None]
    blocks["year"] = sp.csr_matrix(normalize(np.exp(-((d - decades[None, :]) / 10.0) ** 2)).astype(np.float32))

    tfidf = TfidfVectorizer(min_df=2, stop_words="english", token_pattern=r"(?u)\b[a-z][a-z0-9]+\b")
    blocks["title"] = tfidf.fit_transform(movies["title"].map(tokenize_title)).astype(np.float32).tocsr()

    if genome is not None:
        mat = genome.reindex(movies["item_id"].to_numpy())
        has = mat.notna().all(axis=1).to_numpy()
        vals = mat.to_numpy(np.float32, copy=True)
        # Centre each tag so cosine reflects what is *distinctive* about a film,
        # not the shared baseline relevance every movie has for every tag.
        vals[has] -= vals[has].mean(axis=0, keepdims=True)
        vals[~has] = 0.0
        blocks["genome"] = sp.csr_matrix(normalize(vals))
    return blocks


def combine_blocks(blocks: dict[str, sp.csr_matrix], weights: dict[str, float]) -> sp.csr_matrix:
    """Weighted concat so that x_i . x_j = sum_b w_b cos_b(i, j), then row-normalise."""
    mats = [blocks[name] * np.sqrt(w) for name, w in weights.items() if name in blocks and w > 0]
    return sp.csr_matrix(normalize(sp.hstack(mats).tocsr()))


def user_demographics(users: pd.DataFrame) -> np.ndarray:
    onehots = [pd.get_dummies(users[c].astype(str), prefix=c) for c in ("gender", "age", "occupation")]
    return pd.concat(onehots, axis=1).to_numpy(np.float32)


@dataclass
class TrainData:
    """Everything a model may look at during training, in dense index space."""
    user_ids: pd.Index
    item_ids: pd.Index
    R: sp.csr_matrix                 # users x items, explicit rating (0 = unobserved)
    item_x: sp.csr_matrix            # items x d content features (rows L2-normalised)
    user_demo: np.ndarray            # users x d_u one-hot demographics
    item_blocks: dict[str, sp.csr_matrix] = field(repr=False, default_factory=dict)

    @property
    def n_users(self) -> int:
        return self.R.shape[0]

    @property
    def n_items(self) -> int:
        return self.R.shape[1]

    @property
    def user_counts(self) -> np.ndarray:
        return np.diff(self.R.indptr)

    @property
    def item_counts(self) -> np.ndarray:
        return np.diff(self.R.tocsc().indptr)

    def encode(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.user_ids.get_indexer(df["user_id"]), self.item_ids.get_indexer(df["item_id"])


def build_train_data(ml: MovieLens, train: pd.DataFrame,
                     block_weights: dict[str, float] | None = None) -> TrainData:
    # Catalogue = every movie that was ever rated, so cold items with zero
    # training interactions are still candidates.
    movies = ml.movies[ml.movies["item_id"].isin(ml.ratings["item_id"].unique())].reset_index(drop=True)
    users = ml.users.sort_values("user_id").reset_index(drop=True)
    user_ids, item_ids = pd.Index(users["user_id"]), pd.Index(movies["item_id"])

    rows, cols = user_ids.get_indexer(train["user_id"]), item_ids.get_indexer(train["item_id"])
    R = sp.csr_matrix((train["rating"].to_numpy(np.float32), (rows, cols)),
                      shape=(len(user_ids), len(item_ids)))
    blocks = item_feature_blocks(movies, ml.genome)
    item_x = combine_blocks(blocks, block_weights or DEFAULT_BLOCK_WEIGHTS)
    return TrainData(user_ids, item_ids, R, item_x, user_demographics(users), blocks)
