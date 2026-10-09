"""Recommenders. Every model exposes ``fit(data)`` and ``score(users) -> (len(users), n_items)``.

Scores only need to be comparable within a user's row; the evaluator masks
already-seen items and takes the top-K.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .features import TrainData


def zscore_rows(s: np.ndarray) -> np.ndarray:
    mu = s.mean(axis=1, keepdims=True)
    sd = s.std(axis=1, keepdims=True)
    return (s - mu) / np.maximum(sd, 1e-8)


# --------------------------------------------------------------------------- #
# Baselines                                                                   #
# --------------------------------------------------------------------------- #
class Popularity:
    """Global popularity: log(1 + #training interactions)."""
    name = "Popularity"

    def fit(self, data: TrainData) -> "Popularity":
        self.pop = np.log1p(data.item_counts).astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.pop, (len(users), len(self.pop))).copy()


class DemographicPopularity:
    """Naive-Bayes popularity conditioned on gender, age bucket and occupation.

    score(u, i) = log P(i) + sum_g [log P(i | g = v_g(u)) - log P(i)]
    with each conditional shrunk towards P(i) by ``shrinkage`` pseudo-interactions.
    Needs zero interaction history, so it is the fallback for brand-new users.
    """
    name = "DemoPop"

    def __init__(self, shrinkage: float = 500.0):
        self.shrinkage = shrinkage

    def fit(self, data: TrainData) -> "DemographicPopularity":
        B = (data.R > 0).astype(np.float32)
        D = data.user_demo
        counts = np.asarray(B.sum(axis=0)).ravel()
        p_global = (counts + 1.0) / (counts.sum() + len(counts))
        C = np.asarray(D.T @ B)                                   # attr-value x items
        p_cond = (C + self.shrinkage * p_global) / (C.sum(axis=1, keepdims=True) + self.shrinkage)
        self.D = D
        self.log_ratio = (np.log(p_cond) - np.log(p_global)).astype(np.float32)
        self.log_global = np.log(p_global).astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.log_global[None, :] + self.D[users] @ self.log_ratio


# --------------------------------------------------------------------------- #
# Collaborative filtering: implicit-feedback ALS (Hu, Koren & Volinsky 2008)  #
# --------------------------------------------------------------------------- #
class ImplicitALS:
    """Weighted matrix factorisation on the observed/unobserved matrix.

    Every rated item counts as a positive (p=1) with confidence
    ``1 + alpha * rating / 5``; unobserved cells are p=0 with confidence 1.
    After training, a ridge regression maps item *content* features to the
    learned item factors (fitted on well-observed items) so items with little
    or no history can still be placed in the latent space.
    """
    name = "ALS"

    def __init__(self, factors: int = 64, reg: float = 1.0, alpha: float = 10.0,
                 iterations: int = 12, map_min_count: int = 20, map_reg: float = 1.0,
                 seed: int = 0, verbose: bool = False):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed, self.verbose = iterations, seed, verbose
        self.map_min_count, self.map_reg = map_min_count, map_reg

    @staticmethod
    def _solve(Cui: sp.csr_matrix, Y: np.ndarray, reg: float) -> np.ndarray:
        f = Y.shape[1]
        YtY = Y.T @ Y + reg * np.eye(f)
        X = np.zeros((Cui.shape[0], f))
        indptr, indices, conf = Cui.indptr, Cui.indices, Cui.data
        for u in range(Cui.shape[0]):
            lo, hi = indptr[u], indptr[u + 1]
            if lo == hi:
                continue  # no history -> zero vector (scores are all 0)
            Yu, c = Y[indices[lo:hi]], conf[lo:hi]
            A = YtY + (Yu.T * (c - 1.0)) @ Yu
            X[u] = np.linalg.solve(A, Yu.T @ c)
        return X

    def fit(self, data: TrainData) -> "ImplicitALS":
        rng = np.random.default_rng(self.seed)
        C = data.R.copy().astype(np.float64)
        C.data = 1.0 + self.alpha * C.data / 5.0
        Ct = C.T.tocsr()
        self.U = rng.normal(0, 0.01, (data.n_users, self.factors))
        self.V = rng.normal(0, 0.01, (data.n_items, self.factors))
        for it in range(self.iterations):
            t = time.time()
            # thousands of tiny f x f solves: multi-threaded BLAS only adds overhead
            with threadpool_limits(1):
                self.U = self._solve(C, self.V, self.reg)
                self.V = self._solve(Ct, self.U, self.reg)
            if self.verbose:
                print(f"  ALS iter {it + 1}/{self.iterations} ({time.time() - t:.1f}s)")

        return self.fit_mapping(data)

    def fit_mapping(self, data: TrainData) -> "ImplicitALS":
        """Ridge map from item content features to learned item factors, fitted
        on well-observed items; gives every item (even unseen) a factor guess."""
        well = data.item_counts >= self.map_min_count
        ridge = Ridge(alpha=self.map_reg, fit_intercept=True).fit(data.item_x[well], self.V[well])
        self.V_mapped = ridge.predict(data.item_x)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.U[users] @ self.V.T).astype(np.float32)

    def score_mapped(self, users: np.ndarray) -> np.ndarray:
        return (self.U[users] @ self.V_mapped.T).astype(np.float32)


# --------------------------------------------------------------------------- #
# Content-based                                                               #
# --------------------------------------------------------------------------- #
class ContentProfile:
    """User profile = rating-weighted sum of the content vectors of rated items.

    Weights are ``rating - 2.5`` so 1-2 star ratings push *away* from similar
    films. Score is the dot product with each candidate's content vector,
    computed through the item-item similarity matrix K = X X^T.
    """
    name = "Content"

    def __init__(self, neutral: float = 2.5):
        self.neutral = neutral

    def fit(self, data: TrainData) -> "ContentProfile":
        self.K = (data.item_x @ data.item_x.T).toarray().astype(np.float32)
        W = data.R.copy()
        W.data = W.data - self.neutral
        self.W = W.tocsr()
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.W[users] @ self.K, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Hybrids                                                                     #
# --------------------------------------------------------------------------- #
class FixedHybrid:
    """The naive baseline the brief warns against: a constant 50/50 mix."""

    def __init__(self, cf: ImplicitALS, cb: ContentProfile, w_cf: float = 0.5):
        self.cf, self.cb, self.w_cf = cf, cb, w_cf
        self.name = f"Fixed {int(w_cf * 100)}/{int(round((1 - w_cf) * 100))}"

    def fit(self, data: TrainData) -> "FixedHybrid":
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.w_cf * zscore_rows(self.cf.score(users))
                + (1 - self.w_cf) * zscore_rows(self.cb.score(users)))


@dataclass
class GateParams:
    k_user: float = 10.0     # user interactions at which CF reaches half its maximum trust
    k_item: float = 10.0     # item interactions at which learned factors get half the trust
    k_content: float = 2.0   # user interactions at which content profile beats demographic prior
    rho: float = 0.5         # how far content-mapped item factors are trusted vs learned ones
    sharpness: float = 1.0   # Hill exponent: >1 makes the cold -> warm hand-over crisper
    cf_cap: float = 1.0      # maximum CF weight; < 1 keeps some content even for heavy users


def _hill(n: np.ndarray, k: float, s: float) -> np.ndarray:
    # n^s / (n^s + k^s) written as 1 / (1 + (k/n)^s): no overflow for large n
    # and s, and n = 0 maps cleanly to 0.
    with np.errstate(divide="ignore", over="ignore"):
        return 1.0 / (1.0 + (k / np.asarray(n, dtype=np.float64)) ** s)


def user_gates(n_u: np.ndarray, p: GateParams) -> tuple[np.ndarray, np.ndarray]:
    """(alpha, gamma): trust in a user's CF factors and in their content profile."""
    return _hill(n_u, p.k_user, p.sharpness), _hill(n_u, p.k_content, p.sharpness)


def item_gate(n_i: np.ndarray, p: GateParams) -> np.ndarray:
    """beta: trust in an item's learned (vs content-mapped) CF factors."""
    return n_i / (n_i + p.k_item)


def gate_weights(n_u: np.ndarray, n_i: np.ndarray, p: GateParams):
    """Blend weights for users with ``n_u`` and items with ``n_i`` training
    interactions (broadcast: n_u is a column, n_i a row). Returns
    (beta, w_cf, w_cb, w_pop) with w_cf + w_cb + w_pop = 1 everywhere."""
    alpha, gamma = user_gates(n_u, p)
    beta = item_gate(n_i, p)
    w_cf = p.cf_cap * alpha * (beta + (1 - beta) * p.rho)
    return beta, w_cf, (1 - w_cf) * gamma, (1 - w_cf) * (1 - gamma)


class AdaptiveHybrid:
    """Confidence-gated blend of CF, content and demographic popularity.

    For user u with n_u training interactions and item i with n_i:

        alpha_u = n_u^s / (n_u^s + k_user^s)       trust in the user's CF factors
        gamma_u = n_u^s / (n_u^s + k_content^s)    trust in the user's content profile
        beta_i  = n_i / (n_i + k_item)             trust in the item's learned CF factors

        CF(u,i) = beta_i * <p_u, q_i> + (1 - beta_i) * <p_u, q_hat_i>   (q_hat = content-mapped factors)
        w_cf    = cf_cap * alpha_u * (beta_i + (1 - beta_i) * rho)
        w_cb    = (1 - w_cf) * gamma_u
        w_pop   = (1 - w_cf) * (1 - gamma_u)

        score   = w_cf * z(CF) + w_cb * z(Content) + w_pop * z(DemoPop)

    A brand-new user (n_u = 0) is served purely by demographic popularity, a
    user with a handful of ratings by a content/popularity mix, and an
    established user mostly by CF, with a smooth hand-over in between. On the
    item side a new item's CF score comes from its content-mapped factors and
    ``rho`` sets how much CF weight it keeps. With k_user, k_content -> 0 and
    rho = 1 the gate reduces to a fixed ``cf_cap`` blend, so the fixed mix is a
    special case. All constants are tuned on a validation split carved out of
    training data.
    """
    name = "Adaptive hybrid"

    def __init__(self, cf: ImplicitALS, cb: ContentProfile, pop: DemographicPopularity,
                 params: GateParams | None = None):
        self.cf, self.cb, self.pop = cf, cb, pop
        self.params = params or GateParams()

    def fit(self, data: TrainData) -> "AdaptiveHybrid":
        self.user_counts = data.user_counts.astype(np.float32)
        self.item_counts = data.item_counts.astype(np.float32)
        return self

    def components(self, users: np.ndarray) -> dict[str, np.ndarray]:
        """Raw component scores; cached by the tuner so the gate search is cheap."""
        return {"cf": self.cf.score(users), "cf_mapped": self.cf.score_mapped(users),
                "cb": zscore_rows(self.cb.score(users)), "pop": zscore_rows(self.pop.score(users))}

    def gate(self, users: np.ndarray, p: GateParams | None = None):
        return gate_weights(self.user_counts[users][:, None], self.item_counts[None, :], p or self.params)

    def combine(self, comp: dict[str, np.ndarray], users: np.ndarray,
                p: GateParams | None = None) -> np.ndarray:
        # Same maths as gate_weights, factored so only a few full users x items
        # float32 arrays are materialised (this runs hundreds of times in tuning):
        #   score = w_cf * CF + (1 - w_cf) * mix,  mix = gamma * CB + (1 - gamma) * POP
        p = p or self.params
        alpha, gamma = (g.astype(np.float32) for g in user_gates(self.user_counts[users][:, None], p))
        beta = item_gate(self.item_counts, p)[None, :].astype(np.float32)
        w_cf = (np.float32(p.cf_cap) * alpha) * (beta + (1 - beta) * np.float32(p.rho))
        cf = zscore_rows(comp["cf_mapped"] + beta * (comp["cf"] - comp["cf_mapped"]))
        mix = comp["pop"] + gamma * (comp["cb"] - comp["pop"])
        return mix + w_cf * (cf - mix)

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.combine(self.components(users), users).astype(np.float32)
