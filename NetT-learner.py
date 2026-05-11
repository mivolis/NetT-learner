import os, math, copy, argparse, json, random, time
import numpy as np
from math import log, pi
from typing import Dict, Tuple, List
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy.special import expit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import r2_score

from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

model_dir = "gcn_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
os.makedirs("cv_models", exist_ok=True)

# =========================
# Global Config / Utilities
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_adj_numpy(A: np.ndarray) -> np.ndarray:
    """Symmetric normalization with self-loops."""
    A = A.copy()
    n = A.shape[0]
    # A = A + np.eye(n)
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    return D_inv_sqrt @ A @ D_inv_sqrt


def adj_to_sparse_tensor(A_normal: np.ndarray, device) -> torch.Tensor:
    A_coo = sp.coo_matrix(A_normal)
    indices = torch.LongTensor(np.vstack((A_coo.row, A_coo.col)))
    values = torch.FloatTensor(A_coo.data)
    shape = torch.Size(A_coo.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)


# ================
# Data Generation
# ================

def make_sbm_adjacency(num: int, seed: int, sizes=None, q: int = 2):
    """
    Generate an undirected q=2 SBM adjacency matrix without self-loops, then add 1s on the diagonal for degree and exposure calculations.
    Returns:
      A_mat: (n, n) binary matrix with diagonal entries equal to 1.
      blocks: (n,) community labels from 0 to q-1.
    """
    assert q == 2, "The current implementation assumes two communities; extend it if more communities are needed."
    # Compute p_in / p_out
    p_in  = max(0.0, min(1.0, 2.0 * np.log(num) / num))
    p_out = max(0.0, min(1.0, (np.log(num)) / (2.0 * num)))

    # Split communities evenly or use the specified sizes
    if sizes is None:
        a = num // 2
        b = num - a
        sizes = [a, b]
    else:
        assert sum(sizes) == num and len(sizes) == q

    # Construct the probability matrix
    P = np.full((q, q), p_out, dtype=float)
    # np.fill_diagonal(P, p_in)

    # Generate the SBM
    G = nx.stochastic_block_model(sizes, P, seed=seed, directed=False, selfloops=False)
    A_mat = nx.to_numpy_array(G, nodelist=range(num), dtype=int)

    # For compatibility with the original degree/exposure calculations, set the diagonal to 1, equivalent to adding self-loops.
    np.fill_diagonal(A_mat, 1)

    # Community labels
    blocks = np.concatenate([np.full(s, i, dtype=int) for i, s in enumerate(sizes)])
    return A_mat, blocks

def _graphon_prob(x, y, setting: int):
    """
    Given node coordinates x, y in [0, 1], return the graphon value P(x, y).
    setting is in {1, ..., 10} and corresponds one-to-one with the specified formulas.
    """
    import numpy as np
    X, Y = np.meshgrid(x, y, indexing="ij")
    eps = 1e-12

    if setting == 1:
        P = np.exp(- (np.power(X, 0.7) + np.power(Y, 0.7)))
    elif setting == 2:
        P = np.exp(- np.power(np.maximum(X, Y), 0.75))
    elif setting == 3:
        P = np.exp(-0.5 * (np.minimum(X, Y) + np.sqrt(X) + np.sqrt(Y)))
    elif setting == 4:
        P = 1.0 / (1.0 + np.exp(- (np.power(np.maximum(X, Y), 2) +
                                   np.power(np.minimum(X, Y), 4))))
    elif setting == 5:
        P = np.abs(X - Y)
    elif setting == 6:
        P = (X * Y) / 2.0
    elif setting == 7:
        R2 = X*X + Y*Y
        P = (R2/3.0) * np.cos(1.0 / np.clip(R2, eps, None)) + 0.15
    elif setting == 8:
        S = X + Y
        P = (S/3.0) * np.cos(1.0 / np.clip(S, eps, None)) + 0.15
    elif setting == 9:
        P = np.sin(10.0*np.pi*(X + Y - 5.0))/5.0 + 0.5
    elif setting == 10:
        term1 = np.exp(np.sin(6.0 / (np.power(1.0 - X, 2) + Y*Y)))
        term2 = np.exp(np.sin(6.0 / (X*X + np.power(1.0 - Y, 2))))
        P = 0.25 * np.minimum(term1, term2)
    else:
        raise ValueError("graphon setting must be an integer in 1..10")

    # Clip to [0, 1] and symmetrize
    P = np.clip(P, 0.0, 1.0)
    P = 0.5 * (P + P.T)
    np.fill_diagonal(P, 0.0)  # No self-loops are allowed during sampling
    return P


def _sample_undirected_from_P(P: np.ndarray, seed: int) -> np.ndarray:
    """
    Sample one undirected binary adjacency matrix from the symmetric probability matrix P, with zero diagonal.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    A = np.zeros((n, n), dtype=np.uint8)
    iu = np.triu_indices(n, k=1)        # Upper triangle, excluding the diagonal
    A[iu] = rng.binomial(1, P[iu]).astype(np.uint8)
    A = A + A.T
    # The diagonal is already 0
    return A


def make_graphon_adjacency(num: int,
                           seed: int,
                           setting: int = 6,
                           coord: str = "unif",
                           x_given=None):
    rng = np.random.default_rng(seed)

    # 1. Determine x
    if x_given is not None:
        print("Using generated covariate")
        x = np.asarray(x_given, dtype=float)
        if x.shape[0] != num:
            raise ValueError(f"x_given length {x.shape[0]} != num={num}")
    else:
        if coord == "unif":
            x = rng.uniform(0.0, 1.0, size=num)
        elif coord == "seq":
            x = np.linspace(1/num, 1.0, num)
        else:
            raise ValueError("coord must be 'unif' or 'seq'")

    # 2. Graphon probability matrix and sampling
    P = _graphon_prob(x, x, setting=setting)
    A = _sample_undirected_from_P(P, seed=seed)

    # 3. Add self-loops to make the logic consistent with SBM / ER
    A_mat = A.astype(int)
    np.fill_diagonal(A_mat, 1)

    blocks = np.zeros(num, dtype=int)  # Placeholder
    return A_mat, blocks, P, x

def f(u: np.ndarray) -> np.ndarray:
    """Nonlinearity used in gcn DGP."""
    return np.tanh(u)

def estimate_oracle_gbar_large_sample(
    k: int,
    seed: int,
    balance: float,
    large_n: int = 100000,
):
    """
    Approximate oracle G_bar using the large-sample mean of the treatment propensity.
    When exposure = row-normalized(A) @ Z and Z does not depend on degree,
    the large-sample mean of G can be stably approximated by the large-sample mean of Z.
    """
    rng_big = np.random.default_rng(seed + 20250320)

    x1_big = rng_big.normal(0, 1, size=large_n)
    x2_big = rng_big.normal(0, 1, size=large_n)
    X_rest_big = rng_big.normal(0, 1, (large_n, k - 2)) if k >= 2 else np.zeros((large_n, 0))
    X_big = np.column_stack((x1_big, x2_big, X_rest_big))

    coef_2, coef_1 = 3, 4

    # remove degree in treatment generation, but keep balance as intercept shift
    intercept = -balance

    logits_big = intercept + coef_2 * x2_big + coef_1 * x1_big + X_big[:, 2:].sum(axis=1)
    logits_big = np.clip(logits_big, -709, 709)
    probs_big = expit(logits_big)

    return float(np.mean(probs_big))
    
def generate_data(num: int, p_edge: float, k: int, seed: int, balance: float, y_model: str, graph: str = "sbm"):
    """
    graph: 'sbm' (default) or 'er'. When graph is 'sbm', ignore p_edge and use
           p_in = 2 log n / n and p_out = log n / (2n) to generate a two-community SBM.
    All other logic remains unchanged.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # features
    x1 = rng.normal(0, 1, size=num)
    x2 = rng.normal(0, 1, size=num)
    X_rest = rng.normal(0, 1, (num, k - 2)) if k >= 2 else np.zeros((num, 0))
    X = np.column_stack((x1, x2, X_rest))

    # ====== Modify here: SBM or ER ======
    if graph.lower() == "sbm":
        A_mat, blocks = make_sbm_adjacency(num=num, seed=seed, sizes=None, q=2)
    elif graph.lower() == "graphon":
        A_mat, blocks, P_graphon, x_coord = make_graphon_adjacency(
            num=num,
            seed=seed,
            setting=6,
            coord="unif",
            x_given=x1
        )
    else:
        # Compatibility with the old ER version
        A = nx.erdos_renyi_graph(num, p=p_edge, seed=seed)
        A_mat = nx.to_numpy_array(A, nodelist=range(num), dtype=int)
        # np.fill_diagonal(A_mat, 1)  # Consistent with the old logic: self-loops are used for degree/exposure
        blocks = np.zeros(num, dtype=int)  # Placeholder
    # Raw adjacency for exposure: no self-loop
    # A_exposure = A_mat.copy().astype(float)
    np.fill_diagonal(A_mat, 0)
    
    # ===============================
    degree = A_mat.sum(axis=1)
    
    # -------------------------
    # Z generation: REMOVE degree
    # -------------------------
    coef_2, coef_1 = 3, 4
    
    # Keep balance as a global intercept adjustment term
    intercept = -balance
    
    def gen_Z(local_seed):
        rng_local = np.random.default_rng(local_seed)
        logits = intercept + coef_2 * x2 + coef_1 * x1 + X[:, 2:].sum(axis=1)
        logits = np.clip(logits, -709, 709)
        probs = expit(logits)
        gbar_obs_proxy = float(np.mean(probs))   # Propensity mean for the current sample
        Z_draw = rng_local.binomial(1, probs).astype(float)
        return Z_draw, gbar_obs_proxy
    
    Z, G_bar = gen_Z(seed)
    tick = 1
    while len(np.unique(Z)) < 2 and tick < 50:
        Z, G_bar = gen_Z(seed + 100 * tick)
        tick += 1
    
    # Additional quantity: large-sample oracle G_bar
    G_bar_oracle = estimate_oracle_gbar_large_sample(
        k=k,
        seed=seed,
        balance=balance,
        large_n=100000,
    )
    # exposure G

    def gcn_true_forward(X_in: np.ndarray, A_normal: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
        H1 = A_normal @ X_in @ W1
        H1 = np.maximum(H1, 0)   # ReLU
        H2 = A_normal @ H1 @ W2
        return H2.flatten()

    beta_z = np.repeat(0.0, k)
    beta_g = np.repeat(0.0, k)

    if y_model == "linear":
        G = np.zeros(num)
        for i in range(num):
            treated_neighbors = np.sum(A_mat[i] * Z)
            total_neighbors = np.sum(A_mat[i])
            G[i] = treated_neighbors / total_neighbors if total_neighbors > 0 else 0.0

        # A_normal = normalize_adj_numpy(A_mat)  # (n,n)
    
        # ---- 2) define exposure G (choose ONE consistent definition) ----
        # Option A (recommended): exposure uses SAME A_normal
        # G = (A_normal @ Z).astype(float)
        
        X_neighbor = np.zeros_like(X)
        for i in range(num):
            neighbors = np.where(A_mat[i] == 1)[0]
            if len(neighbors):
                X_neighbor[i] = X[neighbors].mean(axis=0)
                
        np.fill_diagonal(A_mat, 1)

        # Y models; keep the original logic
        eps = rng.normal(0, 1, size=num)
        beta_z = np.array([1]*(k))   # for Z*X
        beta_g = np.array([1]*(k))   # for G*X

        ZX = (X * Z[:, None])   # Multiply each column by Z
        GX = (X * G[:, None])   # Multiply each column by G

        Y = (3
            + X[:, 0] + X[:, 1] + X[:, 2:].sum(axis=1)
            + 0.5 * X_neighbor[:, 0] + 0.5 * X_neighbor[:, 1] + 0.5 * X_neighbor[:, 2:].sum(axis=1)
            + 3 * Z
            + 2 * G
            + Z * G
            + 0.2 * ZX @ beta_z       # <-- heterogeneous direct effect in X
            + 0.2 * GX @ beta_g       # <-- heterogeneous peer effect in X
            + eps)
        ground = "linear"
        W1 = W2 = None
    
    elif y_model == "gcn":
        print("Generating GCN_MLP-structured outcome (encoder on X, head on [h,Z,G]).")
        np.fill_diagonal(A_mat, 0)
        rng = np.random.default_rng(seed)

        A_normal = normalize_adj_numpy(A_mat)  # (n,n)
    
        G = (A_normal @ Z).astype(float)
        
        X_neighbor = A_normal @ X
        np.fill_diagonal(A_mat, 1)
        A_normal = normalize_adj_numpy(A_mat)
        hidden = 32
        W1 = rng.normal(0.0, 1.0, size=(k, hidden))      # (k, hidden)
        W2 = rng.normal(0.0, 1.0, size=(hidden, 1)) # (hidden, 1)
    
        # ---- 4) compute graph term: ReLU(A (ReLU(A X W1) W2)) ----
        H0 = A_normal @ (X @ W1)            # (n, hidden)
        H1 = np.maximum(H0, 0.0)            # ReLU
    
        H2 = H1 @ W2                        # (n, 1)
        H2 = A_normal @ H2                  # (n, 1)
        term_graph = np.maximum(H2, 0.0).reshape(-1)  # (n,)
    
        # ---- 5) add linear treatment/exposure term ----
        alpha1, alpha2, alpha3 = 3.0, 2.0, 1.0   # <-- tune these
        eps = rng.normal(0.0, 0.1, size=num)     # noise

        beta_z = np.array([1]*(k))   # for Z*X
        beta_g = np.array([1]*(k))   # for G*X

        ZX = (X * Z[:, None])   # Multiply each column by Z
        GX = (X * G[:, None])   # Multiply each column by G
        
        Y = (term_graph + alpha1*Z + alpha2*G + alpha3*(Z*G) 
             + eps
            + 0.2*ZX @ beta_z + 0.2*GX @ beta_g
            )       # <-- heterogeneous peer effect in X
        ground = "gcn"

        W_dgp = {"W1": W1, "W2": W2, "alpha": (alpha1, alpha2, alpha3)}
        
    else:
        raise ValueError("graph must be one of {'sbm','er','graphon'}")

    # standardize Y
    scaler = StandardScaler()
    Y_stdzd = scaler.fit_transform(Y.reshape(-1, 1)).flatten()
    Y_mean, Y_std = float(scaler.mean_[0]), float(np.sqrt(scaler.var_[0]))

    A_normal = normalize_adj_numpy(A_mat)
    # A_normal = adj_to_sparse_tensor(A_normal, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    out = {
        "y_model": y_model, "X_raw": X, "X_neighbor": X_neighbor, "Z": Z, "G": G, "A": A_mat,
        "Y": Y_stdzd, "Y_mean": Y_mean, "Y_std": Y_std, "Y_raw": Y,
        "G_bar": G_bar,
        "G_bar_oracle": G_bar_oracle, 
        "beta_z": beta_z, "beta_g": beta_g,
        "A_normal": A_normal, "num": num, "k": k,
        "ground": ground, "degree": degree,
        "blocks": blocks,
        "graph_type": graph.lower()
    }
    if ground == "gcn":
        out.update({
            "W1": W1, "W2": W2,
            "alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3,
        })

    return out

def kernel_smoother_kr_rbf_cv(
    X, y, cv: int = 5,
    return_meta: bool = False,
    log_csv_path: str | None = None,
    random_state: int = 0,
):
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    base = KernelRidge(kernel="rbf")
    param_grid = {
        "alpha": [1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 5e-1, 1e-1],
        "gamma": [1e-3, 5e-2, 1e-2, 5e-1, 1e-1, 1.0],
    }

    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    gs.fit(X, y)
    yhat = gs.predict(X)

    meta = {
        "best_params": gs.best_params_,
        "best_val_mse": float(-gs.best_score_),  # Note that best_score_ is negative MSE
    }

    if log_csv_path is not None:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        df = pd.DataFrame(gs.cv_results_)

        # Convert to positive MSE; cv_results_ stores negative MSE
        if "mean_test_score" in df.columns:
            df["mean_test_mse"] = -df["mean_test_score"]
        if "mean_train_score" in df.columns:
            df["mean_train_mse"] = -df["mean_train_score"]

        # splitK_test_score -> splitK_test_mse
        split_cols = [c for c in df.columns if c.startswith("split") and c.endswith("_test_score")]
        for c in split_cols:
            df[c.replace("_test_score", "_test_mse")] = -df[c]

        df.to_csv(log_csv_path, index=False)

        # Also save the best parameters
        with open(log_csv_path.replace(".csv", "_best_params.json"), "w") as f:
            json.dump(meta, f, indent=2)

    if return_meta:
        return yhat, meta
    return yhat
    
def _ensure_1d_float(y):
    y = np.asarray(y, dtype=float)
    return y.reshape(-1)

def kernel_smoother_mlp_cv(
    X, y,
    cv: int = 5,
    return_meta: bool = False,
    log_csv_path: str | None = None,
    random_state: int = 0,
    param_grid: dict | None = None,
    max_iter: int = 2000,
):
    X = np.asarray(X, dtype=float)
    y = _ensure_1d_float(y)

    if param_grid is None:
        param_grid = {
            "mlp__hidden_layer_sizes": [(64,), (64, 64), (128, 64)],
            "mlp__alpha": [1e-4, 1e-3, 1e-2]
        }
        
    model = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPRegressor(
            random_state=random_state,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
        )),
    ])

    cv_obj = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_obj,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        return_train_score=True,
    )

    gs.fit(X, y)

    yhat = gs.predict(X)

    meta = {
        "best_params": gs.best_params_,
        "best_val_mse": float(-gs.best_score_),
    }

    if log_csv_path is not None:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        df = pd.DataFrame(gs.cv_results_)

        # Convert neg MSE -> MSE
        if "mean_test_score" in df.columns:
            df["mean_test_mse"] = -df["mean_test_score"]
        if "mean_train_score" in df.columns:
            df["mean_train_mse"] = -df["mean_train_score"]

        split_cols = [c for c in df.columns if c.startswith("split") and c.endswith("_test_score")]
        for c in split_cols:
            df[c.replace("_test_score", "_test_mse")] = -df[c]

        df.to_csv(log_csv_path, index=False)
        with open(log_csv_path.replace(".csv", "_best_params.json"), "w") as f:
            json.dump(meta, f, indent=2)

    if return_meta:
        return yhat, meta
    return yhat

def kernel_smoother_knn_uniform(
    X, y,
    cv: int = 5,
    return_meta: bool = False,
    log_csv_path: str | None = None,
    random_state: int = 0,
    param_grid: dict | None = None,
):
    """
    Version 3: sklearn KNeighborsRegressor with custom Gaussian weights
    - Equivalent to a local average using a finite neighborhood and a Gaussian kernel
    """
    X = np.asarray(X, dtype=float)
    y = _ensure_1d_float(y)

    # Default grid (keep it reasonable; change as needed)
    if param_grid is None:
        param_grid = {
            "knn__n_neighbors": [3, 5, 7, 9, 15, 25, 35],
            "knn__p": [1, 2],  # 1=Manhattan, 2=Euclidean
        }

    model = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("knn", KNeighborsRegressor(weights="uniform")),
    ])

    cv_obj = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_obj,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        return_train_score=True,
    )
    gs.fit(X, y)

    yhat = gs.predict(X)

    meta = {
        "best_params": gs.best_params_,
        "best_val_mse": float(-gs.best_score_),
    }

    if log_csv_path is not None:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        df = pd.DataFrame(gs.cv_results_)

        if "mean_test_score" in df.columns:
            df["mean_test_mse"] = -df["mean_test_score"]
        if "mean_train_score" in df.columns:
            df["mean_train_mse"] = -df["mean_train_score"]

        split_cols = [c for c in df.columns if c.startswith("split") and c.endswith("_test_score")]
        for c in split_cols:
            df[c.replace("_test_score", "_test_mse")] = -df[c]

        df.to_csv(log_csv_path, index=False)
        with open(log_csv_path.replace(".csv", "_best_params.json"), "w") as f:
            json.dump(meta, f, indent=2)

    if return_meta:
        return yhat, meta
    return yhat
# Your snippet references these names, but does not include imports for them in the shown part.
# Keep them exactly as-is from your full script. Here, I only keep the dict exactly as you set it.
KERNEL_SMOOTHERS = {
    "kr_rbf": kernel_smoother_kr_rbf_cv,
    # "mlp": kernel_smoother_mlp_cv,
    # "knn_u": kernel_smoother_knn_uniform,
}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, copy, random
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =========================
# Utilities (method1 needs)
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adj_to_sparse_tensor(A_normal: np.ndarray, device) -> torch.Tensor:
    A_coo = sp.coo_matrix(A_normal)
    indices = torch.LongTensor(np.vstack((A_coo.row, A_coo.col)))
    values = torch.FloatTensor(A_coo.data)
    shape = torch.Size(A_coo.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

# =========================
# GCN model (method1 needs)
# =========================

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, norm_adj):
        # Message passing with normalized adjacency
        x = torch.spmm(norm_adj, x)
        
        # Feature transformation
        x = self.linear(x)
        
        return x
        
class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=1, dropout=0.1):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(in_features, hidden_features)
        self.bn1 = torch.nn.BatchNorm1d(hidden_features)
        self.gc2 = GCNLayer(hidden_features, out_features)
        self.dropout = dropout
        
    def forward(self, x, norm_adj):
        x = self.gc1(x, norm_adj)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, norm_adj)
        return x  # No activation for regression


class GCN_MLP(nn.Module):
    def __init__(self, x_dim, hidden_features=32, out_features=1, dropout=0.2,
                 add_zg=False, add_interactionz=False, add_interactiong=False):
        super().__init__()
        self.dropout = dropout
        self.add_zg = add_zg
        # self.add_interactions = add_interactions
        self.add_interactionz = add_interactionz
        self.add_interactiong = add_interactiong

        # --- GNN backbone only sees X ---
        self.gcn1 = GCNLayer(x_dim, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, hidden_features)

        self.ln1 = nn.LayerNorm(hidden_features)
        self.ln2 = nn.LayerNorm(hidden_features)

        self.proj_xz = nn.Linear(x_dim, hidden_features)  # proj X*Z to hidden
        self.proj_xg = nn.Linear(x_dim, hidden_features)  # proj X*G to hidden

        # residual projections
        self.res0 = nn.Linear(x_dim, hidden_features, bias=False)

        # --- Head gets (h, Z, G, optional interactions) ---
        head_in = hidden_features + 1
        if add_zg:
            head_in += 1
        if add_interactionz:
            # optional: let head learn heterogeneity via h*Z and h*G
            head_in += hidden_features
        if add_interactiong:
            # optional: let head learn heterogeneity via h*Z and h*G
            head_in += hidden_features
            
        # head_in += 2 * hidden_features

        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_features),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, X, edge_index, Z, G, edge_weight=None):
        # shapes
        if Z.dim() == 1: Z = Z.unsqueeze(1)
        if G.dim() == 1: G = G.unsqueeze(1)

        # --- Layer 1 ---
        h0 = self.gcn1(X, edge_index)
        h1 = self.ln1(h0)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        # h1 = h1 + self.res0(X)   # residual to prevent oversmoothing / ease linear DGP

        # --- Layer 2 ---
        h2 = self.gcn2(h1, edge_index)
        h2 = self.ln2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        # h2 = h2 + h1             # residual in hidden space

        feats = [h2, G]

        if self.add_zg:
            feats.append(Z * G)

        if self.add_interactionz:
            feats.append(self.proj_xz(X * Z))  # (n, hidden)
        if self.add_interactiong:
            feats.append(self.proj_xg(X * G))

        head_x = torch.cat(feats, dim=1)
        return self.head(head_x)


# class GCN_MLP(nn.Module):
#     def __init__(self, x_dim, hidden_features=32, out_features=1, dropout=0.2,
#                  add_zg=True, add_interactions=False):
#         super().__init__()
#         self.dropout = dropout
#         self.add_zg = add_zg
#         self.add_interactions = add_interactions

#         # --- GNN backbone only sees X ---
#         self.gcn1 = GCNLayer(x_dim, hidden_features)
#         self.gcn2 = GCNLayer(hidden_features, hidden_features)

#         self.ln1 = nn.LayerNorm(hidden_features)
#         self.ln2 = nn.LayerNorm(hidden_features)

#         self.proj_xz = nn.Linear(x_dim, hidden_features)  # proj X*Z to hidden
#         self.proj_xg = nn.Linear(x_dim, hidden_features)  # proj X*G to hidden

#         # residual projections
#         self.res0 = nn.Linear(x_dim, hidden_features, bias=False)

#         # --- Head gets (h, Z, G, optional interactions) ---
#         head_in = hidden_features + 2
#         if add_zg:
#             head_in += 1
#         if add_interactions:
#             # optional: let head learn heterogeneity via h*Z and h*G
#             head_in += 2 * hidden_features

#         head_in += 2 * hidden_features

#         self.head = nn.Sequential(
#             nn.Linear(head_in, hidden_features),
#             # nn.ReLU(),
#             # nn.Dropout(dropout),
#             nn.Linear(hidden_features, out_features),
#         )

#     def forward(self, X, edge_index, Z, G, edge_weight=None):
#         # shapes
#         if Z.dim() == 1: Z = Z.unsqueeze(1)
#         if G.dim() == 1: G = G.unsqueeze(1)

#         # --- Layer 1 ---
#         h0 = self.gcn1(X, edge_index)
#         h1 = self.ln1(h0)
#         h1 = F.relu(h1)
#         h1 = F.dropout(h1, p=self.dropout, training=self.training)
#         # h1 = h1 + self.res0(X)   # residual to prevent oversmoothing / ease linear DGP

#         # --- Layer 2 ---
#         h2 = self.gcn2(h1, edge_index)
#         h2 = self.ln2(h2)
#         h2 = F.relu(h2)
#         h2 = F.dropout(h2, p=self.dropout, training=self.training)
#         # h2 = h2 + h1             # residual in hidden space

#         feats = [h2, Z, G]
#         feats.append(self.proj_xz(X * Z))  # (n, hidden)
#         feats.append(self.proj_xg(X * G))

#         if self.add_zg:
#             feats.append(Z * G)

#         if self.add_interactions:
#             feats.append(h0 * Z)   # (n,hidden)
#             feats.append(h0 * G)

#         head_x = torch.cat(feats, dim=1)
#         return self.head(head_x)


# =========================
# Train/Eval (method1 needs)
# =========================

def train(epoch, model, optimizer, X_t, Z_t, G_t, Y_t, A_norm, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(X_t, A_norm, Z_t, G_t).squeeze()
    loss = F.mse_loss(out[train_mask], Y_t[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())

def evaluate(model, X_t, Z_t, G_t, Y_t, A_norm, train_mask, test_mask):
    model.eval()
    with torch.no_grad():
        pred = model(X_t, A_norm, Z_t, G_t).squeeze()

    y_train = Y_t[train_mask].detach().cpu().numpy()
    y_test  = Y_t[test_mask].detach().cpu().numpy()
    p_train = pred[train_mask].detach().cpu().numpy()
    p_test  = pred[test_mask].detach().cpu().numpy()

    train_r2 = r2_score(y_train, p_train)
    test_r2  = r2_score(y_test, p_test)

    train_mse = float(np.mean((p_train - y_train) ** 2))
    test_mse  = float(np.mean((p_test  - y_test)  ** 2))

    return train_r2, test_r2, train_mse, test_mse, pred.detach().cpu().numpy()
# ============================================================
# method1 (kept exactly; only formatting to be runnable)
# ============================================================
def calculate_direct_effects_iterative_symmetric(
    X_only, Y_mean, Y_std, model_Z0, model_Z1,
    A_normalized, A_matrix, Z, G, X_neighbor
):
    num_nodes = X_only.shape[0]
    device = next(model_Z0.parameters()).device

    X_t = torch.FloatTensor(X_only).to(device)
    G_t = torch.FloatTensor(G).to(device)

    direct_pred = np.zeros(num_nodes, dtype=float)

    for i in range(num_nodes):
        Z1 = Z.copy(); Z1[i] = 1.0
        Z0 = Z.copy(); Z0[i] = 0.0

        Z1_t = torch.FloatTensor(Z1).to(device)
        Z0_t = torch.FloatTensor(Z0).to(device)

        with torch.no_grad():
            y1_std = model_Z1(X_t, A_normalized, Z1_t, G_t)[i].squeeze()
            y0_std = model_Z0(X_t, A_normalized, Z0_t, G_t)[i].squeeze()

        y1 = y1_std.cpu().numpy() * Y_std + Y_mean
        y0 = y0_std.cpu().numpy() * Y_std + Y_mean
        direct_pred[i] = y1 - y0

    return direct_pred

def recompute_G_from_Z(A_normal_dense, Z_vec):
    return (A_normal_dense @ Z_vec).astype(float)
    
def calculate_peer_effects_iterative_symmetric(
    X_only, Y_mean, Y_std, model_Z0,
    A_normalized, A_matrix, Z, G, X_neighbor
):
    num_nodes = X_only.shape[0]
    device = next(model_Z0.parameters()).device

    X_t = torch.FloatTensor(X_only).to(device)
    G_t = torch.FloatTensor(G).to(device)

    peer_pred = np.zeros(num_nodes, dtype=float)
    A_noself = A_matrix.copy()
    np.fill_diagonal(A_noself, 0)

    for i in range(num_nodes):
        neighbors = np.where(A_noself[i] > 0)[0]

        # scenario 1: keep observed Z (but force i to 0)
        Z_obs = Z.copy()
        Z_obs[i] = 0.0

        # scenario 2: set neighbors' Z to 0 as well (i already 0)
        Z_g0 = Z_obs.copy()
        if len(neighbors) > 0:
            Z_g0[neighbors] = 0.0

        Z_obs_t = torch.FloatTensor(Z_obs).to(device)
        Z_g0_t  = torch.FloatTensor(Z_g0).to(device)

        G0 = G.copy(); G0[i] = 0.0
        G_0 = torch.FloatTensor(G0).to(device)

        # G_obs = recompute_G_from_Z(A_normal_dense, Z_obs)
        # G_g0  = recompute_G_from_Z(A_normal_dense, Z_g0)

        # Z_obs_t = torch.FloatTensor(Z_obs).to(device)
        # Z_g0_t  = torch.FloatTensor(Z_g0).to(device)
        # G_obs_t = torch.FloatTensor(G_obs).to(device)
        # G_g0_t  = torch.FloatTensor(G_g0).to(device)
        
        with torch.no_grad():
            y_g_obs_std = model_Z0(X_t, A_normalized, Z_g0_t, G_t)[i].squeeze()
            y_g0_std    = model_Z0(X_t, A_normalized, Z_g0_t, G_0)[i].squeeze()

        y_g_obs = y_g_obs_std.cpu().numpy() * Y_std + Y_mean
        y_g0    = y_g0_std.cpu().numpy()    * Y_std + Y_mean
        peer_pred[i] = y_g_obs - y_g0

    return peer_pred

def ensemble_direct_peer_effects_from_folds_nocounterfact_i_line(
    X_enhanced, X_neighbor, Z, G, Y_mean, Y_std, A_norm, A_matrix,
    num_folds=3, model_class=None, in_features=None, hidden_features=32,
    out_features=1, device="cpu", model_Z10 = None
):
    """
    Load model_Z1 / model_Z0 from each fold,
    compute direct / peer predictions for each node, and then average across folds.

    Returns:
      direct_true_node : (n,)
      direct_pred_avg  : (n,)
      peer_true_node   : (n,)
      peer_pred_avg    : (n,)
    """
    direct_pred_folds = []
    peer_pred_folds   = []

    for fold in range(num_folds):
        # Load models
        model_Z1 = GCN_MLP(x_dim=k, hidden_features=32, out_features=1, add_zg=True, add_interactionz=True, add_interactiong=True).to(device)
        model_Z10 = GCN_MLP(x_dim=k, hidden_features=32, out_features=1, add_zg=True, add_interactionz=True, add_interactiong=True).to(device)
        model_Z0 = GCN_MLP(x_dim=k, hidden_features=32, out_features=1, add_zg=True, add_interactionz=True, add_interactiong=False).to(device)

        model_Z1.load_state_dict(torch.load(f"cv_models/fold{fold}_Z1.pt", map_location=device))
        model_Z10.load_state_dict(torch.load(f"cv_models/fold{fold}_Z10.pt", map_location=device))
        model_Z0.load_state_dict(torch.load(f"cv_models/fold{fold}_Z0.pt", map_location=device))
        model_Z1.eval()
        model_Z10.eval()
        model_Z0.eval()

        # node-level predicted effects for this fold
        direct_pred = calculate_direct_effects_iterative_symmetric(
            X_enhanced, Y_mean, Y_std,
            model_Z10, model_Z1,
            A_norm, A_matrix, Z, G, X_neighbor
        )
        peer_pred = calculate_peer_effects_iterative_symmetric(
            X_enhanced, Y_mean, Y_std,
            model_Z0,
            A_norm, A_matrix, Z, G, X_neighbor
        )

        direct_pred_folds.append(direct_pred)
        peer_pred_folds.append(peer_pred)

    direct_pred_folds = np.stack(direct_pred_folds, axis=0)  # (num_folds, n)
    peer_pred_folds   = np.stack(peer_pred_folds,   axis=0)

    # Average across folds to obtain the final node-level predictions
    direct_pred_avg = direct_pred_folds.mean(axis=0)         # (n,)
    peer_pred_avg   = peer_pred_folds.mean(axis=0)           # (n,)

    # Ground truth: consistently use the DGP helper, compatible with both linear and GCN DGPs
    direct_true_node, peer_true_node = true_direct_peer_gcn_nodewise_L1(data)

    return direct_true_node, direct_pred_avg, peer_true_node, peer_pred_avg

def true_direct_peer_gcn_nodewise(data: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    For the GCN DGP, return the true direct / peer effect for each node at Level 1.
    direct_true[i] = f(3 + X_i beta2) - f(0) + G_bar + G_bar * (X_i beta3)
    peer_true[i]   = 2 * G_bar + G_bar * (X_i beta3)
    """
    X = data["X_raw"]          # (n, k)
    G = data["G"].astype(float)
    n = G.shape[0]
    beta_z = data["beta_z"]
    beta_g = data["beta_g"]

    Gbar = float(data.get("G_bar_oracle", data.get("G_bar", np.mean(G))))

    if data["y_model"] == "linear":
        direct_true_L2 = 3.0 + Gbar + 0.2 * X @ beta_z
        peer_true_L2   = 2.0 * Gbar + 0.2 * (X @ beta_g) * Gbar
        return direct_true_L2, peer_true_L2
        
    W1 = np.asarray(data["W1"], dtype=float)        # (k, hidden)
    W2 = np.asarray(data["W2"], dtype=float)        # (hidden, 1)
    a1 = float(data.get("alpha1", 1.0))
    a2 = float(data.get("alpha2", 1.0))
    a3 = float(data.get("alpha3", 1.0))

    # Gbar = float(np.mean(G))

    direct_true_L2 = a1 + a3 * Gbar + 0.2*X @ beta_z
    peer_true_L2   = a2 * Gbar + 0.2*Gbar * (X @ beta_g)

    return direct_true_L2, peer_true_L2


def true_direct_peer_gcn_nodewise_L1(data: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    For the GCN DGP, return the true direct / peer effect for each node at Level 1.
    direct_true[i] = f(3 + X_i beta2) - f(0) + G_bar + G_bar * (X_i beta3)
    peer_true[i]   = 2 * G_bar + G_bar * (X_i beta3)
    """
    X = data["X_raw"]          # (n, k)
    G = data["G"].astype(float)
    n = G.shape[0]
    beta_z = data["beta_z"]
    beta_g = data["beta_g"]
    
    if data["y_model"] == "linear":
        direct_true_L1 = 3.0 + G + 0.2*X @ beta_z
        peer_true_L1   = 2.0 * G + 0.2*(X @ beta_g) * G
        return direct_true_L1, peer_true_L1
        
    W1 = np.asarray(data["W1"], dtype=float)        # (k, hidden)
    W2 = np.asarray(data["W2"], dtype=float)        # (hidden, 1)
    a1 = float(data.get("alpha1", 1.0))
    a2 = float(data.get("alpha2", 1.0))
    a3 = float(data.get("alpha3", 1.0))

    Gbar = float(np.mean(G))

    direct_true_L1 = a1 + a3 * G + 0.2*X @ beta_z
    peer_true_L1   = a2 * G + 0.2*G * (X @ beta_g)
    
    return direct_true_L1, peer_true_L1


def method1_gcn_tlearner(
    data: dict,
    seed: int,
    device,
    hidden: int = 16,
    epochs: int = 800,
    y_model: str = "linear",
    print_every: int = 200,
    test_ratio: float = 0.2,
):
    import os, copy
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import r2_score

    print("EPOCHS: ", epochs)
    set_seed(seed)

    # -------------------- directories --------------------
    os.makedirs("cv_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # -------------------- tensors --------------------
    method_t0 = time.time()
    X_np = data["X_raw"]                       # (n, k)
    X_only = data["X_raw"]
    Z_np = data["Z"].astype(float)             # (n,)
    G_np = data["G"].astype(float)             # (n,)
    
    A_norm = adj_to_sparse_tensor(data["A_normal"], device)
    A_matrix = data["A"]
    
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    Z_t = torch.tensor(Z_np, dtype=torch.float32, device=device)
    G_t = torch.tensor(G_np, dtype=torch.float32, device=device)
    Y_t = torch.tensor(data["Y"], dtype=torch.float32, device=device)
    
    k = X_np.shape[1]  # x_dim

    num = data["num"]

    # ============================================================
    #  Single split (NO K-fold)
    # ============================================================
    rng = np.random.default_rng(seed)
    all_idx = np.arange(num)

    idx1 = all_idx[Z_np > 0.5]
    idx0 = all_idx[Z_np <= 0.5]

    if len(idx1) == 0 or len(idx0) == 0:
        raise ValueError(f"Cannot do T-learner split: len(Z=1)={len(idx1)}, len(Z=0)={len(idx0)}")

    # choose n_test ensuring both classes appear in test if possible
    n_test = int(round(test_ratio * num))
    if n_test < 2:
        n_test = 2
    n_test = min(n_test, num - 1)  # keep at least 1 in train

    # allocate test counts proportionally, but guarantee >=1 each
    n_test1 = int(round(n_test * len(idx1) / num))
    n_test1 = max(1, min(n_test1, len(idx1) - 1 if len(idx1) > 1 else 1))
    n_test0 = n_test - n_test1
    if n_test0 < 1:
        n_test0 = 1
        n_test1 = n_test - 1
    n_test0 = min(n_test0, len(idx0) - 1 if len(idx0) > 1 else 1)

    test_idx_1 = rng.choice(idx1, size=n_test1, replace=False)
    test_idx_0 = rng.choice(idx0, size=n_test0, replace=False)
    test_idx = np.concatenate([test_idx_1, test_idx_0]).astype(int)
    rng.shuffle(test_idx)

    test_mask = torch.zeros(num, dtype=torch.bool, device=device)
    test_mask[test_idx] = True
    train_mask = ~test_mask

    threshold = 0.5
    Z_bool = (Z_t > threshold)
    train_mask_Z1 = train_mask & Z_bool
    train_mask_Z0 = train_mask & ~Z_bool
    test_mask_Z1  = test_mask & Z_bool
    test_mask_Z0  = test_mask & ~Z_bool

    if train_mask_Z1.sum().item() == 0 or train_mask_Z0.sum().item() == 0:
        raise ValueError(
            f"Train split missing a group: train_Z1={train_mask_Z1.sum().item()}, train_Z0={train_mask_Z0.sum().item()}"
        )
    if test_mask_Z1.sum().item() == 0 or test_mask_Z0.sum().item() == 0:
        raise ValueError(
            f"Test split missing a group: test_Z1={test_mask_Z1.sum().item()}, test_Z0={test_mask_Z0.sum().item()}"
        )

    # ============================================================
    #  Train (with retry) for single split
    # ============================================================
    fold = 0  # keep fold=0 naming so your downstream ensemble loader still works

    max_retries = 3
    retry_count = 0

    best_val_r2_Z1 = -float("inf")
    best_val_r2_Z0 = -float("inf")
    best_model_Z1_state = None
    best_model_Z0_state = None

    while retry_count < max_retries:
        torch.manual_seed(seed + fold + retry_count)
        
        model_Z1 = GCN_MLP(x_dim=k, hidden_features=32, out_features=1, add_zg=True, add_interactionz=True, add_interactiong=True).to(device)
        model_Z10 = GCN_MLP(x_dim=k, hidden_features=32, out_features=1, add_zg=True, add_interactionz=True, add_interactiong=True).to(device)
        model_Z0 = GCN_MLP(x_dim=k, hidden_features=32, out_features=1, add_zg=True, add_interactionz=True, add_interactiong=False).to(device)

        optimizer_Z1 = torch.optim.AdamW(model_Z1.parameters(), lr=0.01)
        optimizer_Z10 = torch.optim.AdamW(model_Z10.parameters(), lr=0.01)
        optimizer_Z0 = torch.optim.AdamW(model_Z0.parameters(), lr=0.01)

        best_val_r2_Z1 = -float("inf")
        best_val_r2_Z10 = -float("inf")
        best_val_r2_Z0 = -float("inf")
        best_model_Z1_state = None
        best_model_Z10_state = None
        best_model_Z0_state = None

        for epoch in range(epochs):
            loss_Z1 = train(epoch, model_Z1, optimizer_Z1, X_t, Z_t, G_t, Y_t, A_norm, train_mask_Z1)
            loss_Z10 = train(epoch, model_Z10, optimizer_Z10, X_t, Z_t, G_t, Y_t, A_norm, train_mask_Z0)
            loss_Z0 = train(epoch, model_Z0, optimizer_Z0, X_t, Z_t, G_t, Y_t, A_norm, train_mask_Z0)
            
            train_r2_Z1, val_r2_Z1, train_mse_Z1, *_ = evaluate(
                model_Z1, X_t, Z_t, G_t, Y_t, A_norm, train_mask_Z1, test_mask_Z1
            )
            train_r2_Z0, val_r2_Z0, train_mse_Z0, *_ = evaluate(
                model_Z0, X_t, Z_t, G_t, Y_t, A_norm, train_mask_Z0, test_mask_Z0
            )
            train_r2_Z10, val_r2_Z10, train_mse_Z10, *_ = evaluate(
                model_Z10, X_t, Z_t, G_t, Y_t, A_norm, train_mask_Z0, test_mask_Z0
            )

            if val_r2_Z1 > best_val_r2_Z1:
                best_val_r2_Z1 = val_r2_Z1
                best_model_Z1_state = copy.deepcopy(model_Z1.state_dict())

            if val_r2_Z0 > best_val_r2_Z0:
                best_val_r2_Z0 = val_r2_Z0
                best_model_Z0_state = copy.deepcopy(model_Z0.state_dict())
            
            if val_r2_Z10 > best_val_r2_Z10:
                best_val_r2_Z10 = val_r2_Z10
                best_model_Z10_state = copy.deepcopy(model_Z10.state_dict())

            if (epoch + 1) % print_every == 0:
                print(
                    f"Epoch: {epoch+1:03d}, "
                    f"Loss_Z1: {loss_Z1:.4f}, Train R²_Z1: {train_r2_Z1:.4f}, Val R²_Z1: {val_r2_Z1:.4f}"
                )
                print(
                    f"Epoch: {epoch+1:03d}, "
                    f"Loss_Z0: {loss_Z0:.4f}, Train R²_Z0: {train_r2_Z0:.4f}, Val R²_Z0: {val_r2_Z0:.4f}"
                )
                print(
                    f"Epoch: {epoch+1:03d}, "
                    f"Loss_Z10: {loss_Z10:.4f}, Train R²_Z10: {train_r2_Z10:.4f}, Val R²_Z10: {val_r2_Z10:.4f}"
                )

        if best_val_r2_Z1 < 0.001 or best_val_r2_Z0 < 0.001:
            retry_count += 1
            print(f"Retrying single-split training due to low R². retry={retry_count}/{max_retries}")
        else:
            break

    model_Z1.load_state_dict(best_model_Z1_state)
    model_Z0.load_state_dict(best_model_Z0_state)
    # model_Z10.load_state_dict(best_model_Z10_state)

    _, test_r2_Z1, *_ = evaluate(model_Z1, X_t, Z_t, G_t, Y_t, A_norm, train_mask_Z1, test_mask_Z1)
    _, test_r2_Z0, *_ = evaluate(model_Z0, X_t, Z_t, G_t, Y_t, A_norm, train_mask_Z0, test_mask_Z0)
    _, test_r2_Z10, *_ = evaluate(model_Z10, X_t, Z_t, G_t, Y_t, A_norm, train_mask_Z0, test_mask_Z0)

    print(f"[Single Split] Test R² Z=1: {test_r2_Z1:.4f}, Z=0: {test_r2_Z0:.4f}")

    # keep fold0 filenames so your ensemble function can load
    torch.save(best_model_Z1_state, f"cv_models/fold{fold}_Z1.pt")
    torch.save(best_model_Z0_state, f"cv_models/fold{fold}_Z0.pt")
    torch.save(best_model_Z10_state, f"cv_models/fold{fold}_Z10.pt")

    # ========= 2) Level-1 node-wise true / predicted direct & peer =========
    k_folds = 1
    direct_true_L1, direct_pred_L1, peer_true_L1, peer_pred_L1 = \
        ensemble_direct_peer_effects_from_folds_nocounterfact_i_line(
            X_only, data["X_neighbor"], data["Z"], data["G"],
            data["Y_mean"], data["Y_std"], A_norm, A_matrix,
            num_folds=k_folds,
            model_class=GCN_MLP,
            in_features=k,
            hidden_features=32,
            out_features=1,
            device=device,
            model_Z10 = model_Z10
        )

    # ========= 3) Level-2 smoothing =========
    X_raw = data["X_raw"]
    graph_type = data.get("graph_type", "unknown")
    node_ids = np.arange(num)

    direct_true_L2, peer_true_L2 = true_direct_peer_gcn_nodewise(data)

    summary_rows = []
    cv_log_dir = "cv_logs"
    kernel_select_rows = []
    
    best_kernel = None
    best_kernel_score = np.inf

    runtime_prekernel_sec = time.time() - method_t0
    kernel_runtime_dict = {}
    
    for kernel_name, smoother in KERNEL_SMOOTHERS.items():
        kernel_t0 = time.time()
        print(f"[method1] Running kernel smoother: {kernel_name}")

        direct_hat_L2, meta_d = smoother(
            X_raw, direct_pred_L1,
            return_meta=True,
            log_csv_path=f"{cv_log_dir}/method1_{kernel_name}_direct_cv_results_n{num}_seed{seed}_{graph_type}.csv",
            random_state=seed,
        )
    
        # --- peer: CV + log ---
        peer_hat_L2, meta_p = smoother(
            X_raw, peer_pred_L1,
            return_meta=True,
            log_csv_path=f"{cv_log_dir}/method1_{kernel_name}_peer_cv_results_n{num}_seed{seed}_{graph_type}.csv",
            random_state=seed,
        )
    
        # Score used for kernel selection; the average is recommended
        kernel_score = 0.5 * (meta_d["best_val_mse"] + meta_p["best_val_mse"])

        kernel_select_rows.append({
            "method": "method1",
            "num": num, "seed": seed, "graph": graph_type,
            "kernel": kernel_name,
            "best_val_mse_direct": meta_d["best_val_mse"],
            "best_val_mse_peer": meta_p["best_val_mse"],
            "kernel_score": kernel_score,
            "best_params_direct": json.dumps(meta_d["best_params"]),
            "best_params_peer": json.dumps(meta_p["best_params"]),
        })
    
        if kernel_score < best_kernel_score:
            best_kernel_score = kernel_score
            best_kernel = kernel_name

        R2_L2_direct = r2_score(np.asarray(direct_true_L2).ravel(),
                                np.asarray(direct_hat_L2).ravel())
        R2_L2_peer   = r2_score(np.asarray(peer_true_L2).ravel(),
                                np.asarray(peer_hat_L2).ravel())
        print(f"[{kernel_name}] R2_L2_direct = {R2_L2_direct:.4f}, "
              f"R2_L2_peer = {R2_L2_peer:.4f}")

        kernel_runtime_dict[f"runtime_kernel_{kernel_name}_sec"] = time.time() - kernel_t0

        direct_sqerr_L1 = (direct_pred_L1 - direct_true_L1) ** 2
        peer_sqerr_L1   = (peer_pred_L1  - peer_true_L1)  ** 2
        direct_sqerr_L2 = (direct_hat_L2 - direct_true_L2) ** 2
        peer_sqerr_L2   = (peer_hat_L2   - peer_true_L2)   ** 2

        overall_direct_true_L1 = float(direct_true_L1.mean())
        overall_peer_true_L1   = float(peer_true_L1.mean())
        overall_direct_pred_L1 = float(direct_pred_L1.mean())
        overall_peer_pred_L1   = float(peer_pred_L1.mean())

        overall_direct_true_L2 = float(direct_true_L2.mean())
        overall_peer_true_L2   = float(peer_true_L2.mean())
        overall_direct_hat_L2  = float(direct_hat_L2.mean())
        overall_peer_hat_L2    = float(peer_hat_L2.mean())

        df_nodes = pd.DataFrame({
            "num": num,
            "seed": seed,
            "graph": graph_type,
            "kernel": kernel_name,
            "node_id": node_ids,

            "direct_true_L1": direct_true_L1,
            "peer_true_L1":   peer_true_L1,
            "direct_pred_L1": direct_pred_L1,
            "peer_pred_L1":   peer_pred_L1,

            "direct_true_L2": direct_true_L2,
            "peer_true_L2":   peer_true_L2,
            "direct_hat_L2":  direct_hat_L2,
            "peer_hat_L2":    peer_hat_L2,

            "direct_sqerr_L1": direct_sqerr_L1,
            "peer_sqerr_L1":   peer_sqerr_L1,
            "direct_sqerr_L2": direct_sqerr_L2,
            "peer_sqerr_L2":   peer_sqerr_L2,
        })
        out_name = f"results/method1_{kernel_name}_n{num}_seed{seed}_{graph_type}.csv"
        df_nodes.to_csv(out_name, index=False)
        print(f"[method1] node-level effects saved to {out_name} (rows = {df_nodes.shape[0]})")

        summary_rows.append({
            "num": num,
            "seed": seed,
            "graph": graph_type,
            "kernel": kernel_name,

            "overall_direct_true_L1": overall_direct_true_L1,
            "overall_direct_pred_L1": overall_direct_pred_L1,
            "overall_peer_true_L1":   overall_peer_true_L1,
            "overall_peer_pred_L1":   overall_peer_pred_L1,
            "MSE_L1_direct": float(np.mean(np.abs(direct_pred_L1 - direct_true_L1)**2)),
            "MSE_L1_peer":   float(np.mean(np.abs(peer_pred_L1  - peer_true_L1)**2)),
            "MAE_L1_direct": float(np.mean(np.abs(direct_pred_L1 - direct_true_L1))),
            "MAE_L1_peer":   float(np.mean(np.abs(peer_pred_L1  - peer_true_L1))),

            "overall_direct_true_L2": overall_direct_true_L2,
            "overall_direct_hat_L2":  overall_direct_hat_L2,
            "overall_peer_true_L2":   overall_peer_true_L2,
            "overall_peer_hat_L2":    overall_peer_hat_L2,
            "MSE_L2_direct": float(np.mean(np.abs(direct_hat_L2 - direct_true_L2)**2)),
            "MSE_L2_peer":   float(np.mean(np.abs(peer_hat_L2   - peer_true_L2)**2)),
            "MAE_L2_direct": float(np.mean(np.abs(direct_hat_L2 - direct_true_L2))),
            "MAE_L2_peer":   float(np.mean(np.abs(peer_hat_L2   - peer_true_L2))),
        })
        mae_L1_direct = float(np.mean(np.abs(direct_pred_L1 - direct_true_L1)))
        mae_L1_peer   = float(np.mean(np.abs(peer_pred_L1   - peer_true_L1)))
        mse_L1_direct = float(np.mean(np.abs(direct_pred_L1 - direct_true_L1)**2))
        mse_L1_peer   = float(np.mean(np.abs(peer_pred_L1   - peer_true_L1)**2))
        
        print(f"[{kernel_name}] MAE_L1_direct = {mae_L1_direct:.6f}, MAE_L1_peer = {mae_L1_peer:.6f}")
        print(f"[{kernel_name}] MSE_L1_direct = {mse_L1_direct:.6f}, MSE_L1_peer = {mse_L1_peer:.6f}")

        mae_L2_direct = float(np.mean(np.abs(direct_hat_L2 - direct_true_L2)))
        mae_L2_peer   = float(np.mean(np.abs(peer_hat_L2   - peer_true_L2)))
        mse_L2_direct = float(np.mean(np.abs(direct_hat_L2 - direct_true_L2)**2))
        mse_L2_peer   = float(np.mean(np.abs(peer_hat_L2   - peer_true_L2)**2))
        
        print(f"[{kernel_name}] MAE_L2_direct = {mae_L2_direct:.6f}, MAE_L2_peer = {mae_L2_peer:.6f}")
        print(f"[{kernel_name}] MSE_L2_direct = {mse_L2_direct:.6f}, MSE_L2_peer = {mse_L2_peer:.6f}")

    df_sel = pd.DataFrame(kernel_select_rows).sort_values("kernel_score")
    df_sel.to_csv(f"{cv_log_dir}/method1_kernel_selection_n{num}_seed{seed}_{graph_type}.csv", index=False)
    print(f"[method1] Best kernel by CV(MSE): {best_kernel} (score={best_kernel_score:.6g})")
    
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_name = f"results/method1_kernel_comparison_n{num}_seed{seed}_{graph_type}.csv"
        df_summary.to_csv(summary_name, index=False)
        print(f"[method1] kernel comparison summary saved to {summary_name}")

    # return {
    #     "direct_true_overall": float(direct_true_L2.mean()),
    #     "peer_true_overall":   float(peer_true_L2.mean()),
    #     "direct_pred":         float(direct_hat_L2.mean()),
    #     "peer_pred":           float(peer_hat_L2.mean()),
    #     "overall_pred":        float(direct_hat_L2.mean() + peer_hat_L2.mean()),

    #     "direct_mse": float(((direct_hat_L2 - direct_true_L2) ** 2).mean()),
    #     "peer_mse":   float(((peer_hat_L2  - peer_true_L2)  ** 2).mean()),

    #     "direct_pred_node":  direct_hat_L2,
    #     "peer_pred_node":    peer_hat_L2,
    #     "direct_true_node":  direct_true_L2,
    #     "peer_true_node":    peer_true_L2,
    #     "direct_sqerr_node": (direct_hat_L2 - direct_true_L2) ** 2,
    #     "peer_sqerr_node":   (peer_hat_L2  - peer_true_L2)  ** 2,
    # }
    out = {
        "direct_true_overall": float(direct_true_L2.mean()),
        "peer_true_overall":   float(peer_true_L2.mean()),
        "direct_pred":         float(direct_hat_L2.mean()),
        "peer_pred":           float(peer_hat_L2.mean()),
        "overall_pred":        float(direct_hat_L2.mean() + peer_hat_L2.mean()),

        "direct_mse": float(((direct_hat_L2 - direct_true_L2) ** 2).mean()),
        "peer_mse":   float(((peer_hat_L2  - peer_true_L2)  ** 2).mean()),

        "direct_pred_node":  direct_hat_L2,
        "peer_pred_node":    peer_hat_L2,
        "direct_true_node":  direct_true_L2,
        "peer_true_node":    peer_true_L2,
        "direct_sqerr_node": (direct_hat_L2 - direct_true_L2) ** 2,
        "peer_sqerr_node":   (peer_hat_L2  - peer_true_L2)  ** 2,

        "runtime_prekernel_sec": runtime_prekernel_sec,
    }
    out.update(kernel_runtime_dict)
    return out

def _predict_prob_gi(z_i, degree_i, x_neigh_i, glm_result):
    x = np.concatenate(([1, z_i, degree_i], x_neigh_i))  # Add the intercept
    pi_i = glm_result.predict([x])[0]  # Predicted probability
    return pi_i


def method3_outcome_regression(data: Dict, seed: int, y_model: str = "linear"):
    """
    Method 3 (OR + CATE smoothing):

    Level-1:
      - True:  τ_true(G_i|X_i), Δ_true(G_i|X_i)  (per node)
      - Est:   τ_pred(G_i|X_i), Δ_pred(G_i|X_i) from outcome regression

    Level-2 (CATE in X):
      - Oracle CATE: τ_true(X_i), Δ_true(X_i)  = smoother_X( Level-1 true )
      - Est CATE:    τ_hat(X_i),  Δ_hat(X_i)   = smoother_X( Level-1 pred )

    Output:
      1) One node-level CSV for each kernel, including Level 1 and Level 2 results.
      2) One summary CSV recording overall scalars and MAE.
    """
    set_seed(seed)

    method_t0 = time.time()
    X_raw = data["X_raw"]            # (n, k)
    X_neighbor = data["X_neighbor"]  # (n, k)
    Z = data["Z"].astype(float)      # (n,)
    G = data["G"].astype(float)      # (n,)
    Y_stdzd = data["Y"]              # Standardized Y
    Y_std = data["Y_std"]
    Y_mean = data["Y_mean"]
    beta_z = data["beta_z"]
    beta_g = data["beta_g"]
    n, k = X_raw.shape

    graph_type = data.get("graph_type", "unknown")
    num = data["num"]
    node_ids = np.arange(n)

    # ---------- 1. Fit outcome regressions for Z=1 / Z=0 using Y_stdzd ----------
    def _linreg_outcome_by_treatment(X, Z, G, X_neighbor, Y_stdzd, treatment):
        idx = np.where(Z == treatment)[0]
        G_col = G.reshape(-1, 1)
        outcome = Y_stdzd[idx]
        covariates = np.hstack((X[idx, :], X_neighbor[idx, :], G_col[idx, :]))
        model = LinearRegression()
        model.fit(covariates, outcome)
        return model

    reg_treat   = _linreg_outcome_by_treatment(X_raw, Z, G, X_neighbor, Y_stdzd, 1.0)
    reg_control = _linreg_outcome_by_treatment(X_raw, Z, G, X_neighbor, Y_stdzd, 0.0)
    
    # ---- after fitting reg_treat / reg_control ----
    idx1 = np.where(Z == 1.0)[0]
    X1  = np.hstack((X_raw[idx1, :], X_neighbor[idx1, :], G[idx1].reshape(-1, 1)))
    y1  = Y_stdzd[idx1]
    r2_reg1 = reg_treat.score(X1, y1)
    
    idx0 = np.where(Z == 0.0)[0]
    X0  = np.hstack((X_raw[idx0, :], X_neighbor[idx0, :], G[idx0].reshape(-1, 1)))
    y0  = Y_stdzd[idx0]
    r2_reg2 = reg_control.score(X0, y0)
    
    print(f"[Outcome OR] reg1 (Z=1) train R2 = {r2_reg1:.4f}  (n={len(idx1)})")
    print(f"[Outcome OR] reg2 (Z=0) train R2 = {r2_reg2:.4f}  (n={len(idx0)})")

    print("Outcome regression fitting done.")

    # ---------- 2. Level-1 prediction: node-level predicted direct / peer effects on the original Y scale ----------
    direct_pred_L1 = np.zeros(n)
    peer_pred_L1   = np.zeros(n)

    for i in range(n):
        feat_g = np.hstack((X_raw[i, :], X_neighbor[i, :], G[i])).reshape(1, -1)
        feat_0 = np.hstack((X_raw[i, :], X_neighbor[i, :], 0.0)).reshape(1, -1)

        # direct: Y_hat(Z=1,G_i) - Y_hat(Z=0,G_i)
        y1_std = reg_treat.predict(feat_g)[0]
        y0_std = reg_control.predict(feat_g)[0]
        y1 = y1_std * Y_std + Y_mean
        y0 = y0_std * Y_std + Y_mean
        direct_pred_L1[i] = y1 - y0

        # peer: Y_hat(Z=0,G_i) - Y_hat(Z=0,0)
        y00_std = reg_control.predict(feat_0)[0]
        y00 = y00_std * Y_std + Y_mean
        peer_pred_L1[i] = y0 - y00

    print("Level-1 predicted node-wise direct/peer effects computed.")

    # ---------- 3. Level-1 truth: tau(G_i | X_i) and Delta(G_i | X_i) under the linear DGP ----------
    # This is the original definition: true Level 1, not based on G_bar
    # direct_true_L1 = 3.0 + G + X_raw @ beta_z
    # peer_true_L1   = 2.0 * G + (X_raw @ beta_g) * G
    # print("Level-1 true node-wise effects computed.")

    # direct_true_L2 = 3.0 + np.mean(G) + X_raw @ beta_z
    # peer_true_L2   = 2.0 * np.mean(G) + (X_raw @ beta_g) * np.mean(G)
    if data.get("ground") == "gcn":
        direct_true_L1, peer_true_L1  = true_direct_peer_gcn_nodewise_L1(data)
        direct_true_L2, peer_true_L2 = true_direct_peer_gcn_nodewise(data)
    else:
        direct_true_L1 = 3.0 + G + 0.2*X_raw @ beta_z
        peer_true_L1   = 2.0 * G + 0.2*(X_raw @ beta_g) * G
        direct_true_L2, peer_true_L2 = true_direct_peer_gcn_nodewise(data)

    print("Level-2 true node-wise effects computed.")

    # Overall scalar at Level 1; use this later if you want the overall ATE
    overall_direct_true_L1 = float(direct_true_L1.mean())
    overall_peer_true_L1   = float(peer_true_L1.mean())
    overall_direct_pred_L1 = float(direct_pred_L1.mean())
    overall_peer_pred_L1   = float(peer_pred_L1.mean())

    # Prepare the summary row
    summary_rows = []
    cv_log_dir = "cv_logs"
    kernel_select_rows = []
    
    best_kernel = None
    best_kernel_score = np.inf

    runtime_prekernel_sec = time.time() - method_t0
    kernel_runtime_dict = {}

    for kernel_name, smoother in KERNEL_SMOOTHERS.items():
        kernel_t0 = time.time()
        print(f"[method3] Running kernel smoother: {kernel_name}")

        direct_hat_L2, meta_d = smoother(
            X_raw, direct_pred_L1,
            return_meta=True,
            log_csv_path=f"{cv_log_dir}/method1_{kernel_name}_direct_cv_results_n{num}_seed{seed}_{graph_type}.csv",
            random_state=seed,
        )
    
        # --- peer: CV + log ---
        peer_hat_L2, meta_p = smoother(
            X_raw, peer_pred_L1,
            return_meta=True,
            log_csv_path=f"{cv_log_dir}/method1_{kernel_name}_peer_cv_results_n{num}_seed{seed}_{graph_type}.csv",
            random_state=seed,
        )
    
        # Score used for kernel selection; the average is recommended
        kernel_score = 0.5 * (meta_d["best_val_mse"] + meta_p["best_val_mse"])

        kernel_select_rows.append({
            "method": "method1",
            "num": num, "seed": seed, "graph": graph_type,
            "kernel": kernel_name,
            "best_val_mse_direct": meta_d["best_val_mse"],
            "best_val_mse_peer": meta_p["best_val_mse"],
            "kernel_score": kernel_score,
            "best_params_direct": json.dumps(meta_d["best_params"]),
            "best_params_peer": json.dumps(meta_p["best_params"]),
        })
    
        if kernel_score < best_kernel_score:
            best_kernel_score = kernel_score
            best_kernel = kernel_name
        R2_L2_direct = r2_score(
            np.asarray(direct_true_L2).ravel(),
            np.asarray(direct_hat_L2).ravel()
        )
        R2_L2_peer = r2_score(
            np.asarray(peer_true_L2).ravel(),
            np.asarray(peer_hat_L2).ravel()
        )
        print(f"[{kernel_name}] R2_L2_direct = {R2_L2_direct:.4f}, "
              f"R2_L2_peer = {R2_L2_peer:.4f}")
        kernel_runtime_dict[f"runtime_kernel_{kernel_name}_sec"] = time.time() - kernel_t0

        # 4.3 Node-wise error (Level-1 & Level-2)
        direct_sqerr_L1 = (direct_pred_L1 - direct_true_L1) ** 2
        peer_sqerr_L1   = (peer_pred_L1  - peer_true_L1)  ** 2
        direct_sqerr_L2 = (direct_hat_L2 - direct_true_L2) ** 2
        peer_sqerr_L2   = (peer_hat_L2   - peer_true_L2)   ** 2

        # 4.4 Overall: average the CATE curve over X again to obtain a scalar, if desired
        overall_direct_true_L2 = float(direct_true_L2.mean())
        overall_peer_true_L2   = float(peer_true_L2.mean())
        overall_direct_hat_L2  = float(direct_hat_L2.mean())
        overall_peer_hat_L2    = float(peer_hat_L2.mean())

        # 4.5 Node-level saving: save both Level 1 and Level 2 results
        df_nodes = pd.DataFrame({
            "num": num,
            "seed": seed,
            "graph": graph_type,
            "kernel": kernel_name,
            "node_id": node_ids,

            # Level-1 true & pred
            "direct_true_L1": direct_true_L1,
            "peer_true_L1":   peer_true_L1,
            "direct_pred_L1": direct_pred_L1,
            "peer_pred_L1":   peer_pred_L1,

            # Level-2 true (oracle CATE) & hat
            "direct_true_L2": direct_true_L2,
            "peer_true_L2":   peer_true_L2,
            "direct_hat_L2":  direct_hat_L2,
            "peer_hat_L2":    peer_hat_L2,

            # node-level sq error
            "direct_sqerr_L1": direct_sqerr_L1,
            "peer_sqerr_L1":   peer_sqerr_L1,
            "direct_sqerr_L2": direct_sqerr_L2,
            "peer_sqerr_L2":   peer_sqerr_L2,
        })

        out_name = (
            f"results/method3_{kernel_name}"
            f"_n{num}_seed{seed}_{graph_type}.csv"
        )
        df_nodes.to_csv(out_name, index=False)
        print(f"[method3] node-level effects saved to {out_name} "
              f"(rows = {df_nodes.shape[0]})")

        # 4.6 Summary record: useful for comparing different kernels / levels later
        summary_rows.append({
            "num": num,
            "seed": seed,
            "graph": graph_type,
            "kernel": kernel_name,

            # Level-1 overall
            "overall_direct_true_L1": overall_direct_true_L1,
            "overall_direct_pred_L1": overall_direct_pred_L1,
            "overall_peer_true_L1":   overall_peer_true_L1,
            "overall_peer_pred_L1":   overall_peer_pred_L1,
            "MAE_L1_direct": float(np.mean(np.abs(direct_pred_L1 - direct_true_L1))),
            "MAE_L1_peer":   float(np.mean(np.abs(peer_pred_L1  - peer_true_L1))),
            "MSE_L1_direct": float(np.mean(np.abs(direct_pred_L1 - direct_true_L1)**2)),
            "MSE_L1_peer":   float(np.mean(np.abs(peer_pred_L1  - peer_true_L1)**2)),

            # Level-2 overall
            "overall_direct_true_L2": overall_direct_true_L2,
            "overall_direct_hat_L2":  overall_direct_hat_L2,
            "overall_peer_true_L2":   overall_peer_true_L2,
            "overall_peer_hat_L2":    overall_peer_hat_L2,
            "MAE_L2_direct": float(np.mean(np.abs(direct_hat_L2 - direct_true_L2))),
            "MAE_L2_peer":   float(np.mean(np.abs(peer_hat_L2   - peer_true_L2))),
            "MSE_L2_direct": float(np.mean(np.abs(direct_hat_L2 - direct_true_L2)**2)),
            "MSE_L2_peer":   float(np.mean(np.abs(peer_hat_L2   - peer_true_L2)**2)),
        })

        mae_L1_direct = float(np.mean(np.abs(direct_pred_L1 - direct_true_L1)))
        mae_L1_peer   = float(np.mean(np.abs(peer_pred_L1   - peer_true_L1)))
        mse_L1_direct = float(np.mean(np.abs(direct_pred_L1 - direct_true_L1)**2))
        mse_L1_peer   = float(np.mean(np.abs(peer_pred_L1   - peer_true_L1)**2))
        
        print(f"[{kernel_name}] MAE_L1_direct = {mae_L1_direct:.6f}, MAE_L1_peer = {mae_L1_peer:.6f}")
        print(f"[{kernel_name}] MSE_L1_direct = {mse_L1_direct:.6f}, MSE_L1_peer = {mse_L1_peer:.6f}")

        mae_L2_direct = float(np.mean(np.abs(direct_hat_L2 - direct_true_L2)))
        mae_L2_peer   = float(np.mean(np.abs(peer_hat_L2   - peer_true_L2)))
        mse_L2_direct = float(np.mean(np.abs(direct_hat_L2 - direct_true_L2)**2))
        mse_L2_peer   = float(np.mean(np.abs(peer_hat_L2   - peer_true_L2)**2))
        
        print(f"[{kernel_name}] MAE_L2_direct = {mae_L2_direct:.6f}, MAE_L2_peer = {mae_L2_peer:.6f}")
        print(f"[{kernel_name}] MSE_L2_direct = {mse_L2_direct:.6f}, MSE_L2_peer = {mse_L2_peer:.6f}")

    df_sel = pd.DataFrame(kernel_select_rows).sort_values("kernel_score")
    df_sel.to_csv(f"{cv_log_dir}/method3_kernel_selection_n{num}_seed{seed}_{graph_type}.csv", index=False)
    print(f"[method3] Best kernel by CV(MSE): {best_kernel} (score={best_kernel_score:.6g})")
    
    # 5. Save the summary CSV
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_name = f"results/method3_kernel_comparison_n{num}_seed{seed}_{graph_type}.csv"
        df_summary.to_csv(summary_name, index=False)
        print(f"[method3] kernel comparison summary saved to {summary_name}")
    else:
        print("[method3] No kernels in KERNEL_SMOOTHERS; nothing saved.")

    # return {
    #         "direct_true_L2": overall_direct_true_L2,
    #         "direct_hat_L2":  overall_direct_hat_L2,
    #         "peer_true_L2":   overall_peer_true_L2,
    #         "peer_hat_L2":    overall_peer_hat_L2,
    #         "MSE_L2_direct": float(np.mean(np.abs(direct_hat_L2 - direct_true_L2)**2)),
    #         "MSE_L2_peer":   float(np.mean(np.abs(peer_hat_L2   - peer_true_L2)**2)),
    # }
    out = {
        "direct_true_L2": overall_direct_true_L2,
        "direct_hat_L2":  overall_direct_hat_L2,
        "peer_true_L2":   overall_peer_true_L2,
        "peer_hat_L2":    overall_peer_hat_L2,
        "MSE_L2_direct": float(np.mean(np.abs(direct_hat_L2 - direct_true_L2)**2)),
        "MSE_L2_peer":   float(np.mean(np.abs(peer_hat_L2   - peer_true_L2)**2)),
        "runtime_prekernel_sec": runtime_prekernel_sec,
    }
    out.update(kernel_runtime_dict)
    return out


METHODS = {
    1: ("gcn_tlearner", method1_gcn_tlearner),
    3: ("outcome_regression", method3_outcome_regression),
}


def run_all(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    device = get_device()
    print(f"[Device] {device}")

    conflict_report = {
        "device_hardcode": "No hardcoded .cuda(); device-agnostic.",
        "adjacency_self_loops": "Self-loops + symmetric normalization everywhere.",
        "Y_standardization": "Train on standardized Y; de-standardize effects.",
        "RNG": "Single set_seed() per iteration; no inner reseeding.",
        "ground_truth": "For y_model='gcn' uses exact GCN DGP-based direct/peer formulas (noiseless).",
    }
    with open(os.path.join(outdir, "conflict_report.json"), "w") as f:
        json.dump(conflict_report, f, indent=2)

    for num in args.nums:
        p_edge = args.coef * math.log(num) / num
        for it in range(args.iterations):
            seed = args.seed + it
            data = generate_data(
                num=num, p_edge=p_edge, k=args.k, seed=seed,
                balance=args.balance, y_model=args.y_model
            )
            for m in args.methods:
                mname, mfn = METHODS[m]
                t0 = time.time()
                try:
                    if m in (1,5,6):
                        res = mfn(data, seed, device, y_model=args.y_model)
                    else:
                        res = mfn(data, seed, y_model=args.y_model)
                except Exception as e:
                    res = {"error": str(e)}
                dt = time.time() - t0

                row = {
                    "num": num, "iteration": it, "seed": seed,
                    "k": args.k, "coef": args.coef, "balance": args.balance,
                    "y_model": args.y_model, "method": mname, "seconds": dt
                }
                row.update(res)
                df = pd.DataFrame([row])
                fname = f"results_{mname}_n{num}_k{args.k}_coef{args.coef}_bal{args.balance}_{args.y_model}.csv"
                fpath = os.path.join(outdir, fname)
                hdr = not os.path.exists(fpath)
                df.to_csv(fpath, mode="a", header=hdr, index=False)
                print(f"[Saved] {fpath} (+1 row)")

    print(f"Done. Outputs in: {outdir}")

# add_inter = False
num_list   = [1000]
# [1000,2000,3000,4000,5000]
k          = 10
coef       = 1.0
balance    = 1
seed       = 7
y_model    = "linear"   # "linear" or "gcn"
graphs     = ['sbm']
# ["graphon", "sbm", "er"]      # "er" or "sbm"
method_ids = [1,3]
# [1,2,3,4,5,6]

import math, time, os, pandas as pd
device = get_device()
os.makedirs("results", exist_ok=True)

all_rows = []  # Summary-level results will be stored here

for graph in graphs:
    for num in num_list:
        p_edge = coef * math.log(num) / num
        for it in range(1, 2):  # iteration = 1..5
            cur_seed = seed + it
    
            # ===== Generate data =====
            data = generate_data(
                num=num,
                p_edge=p_edge,   # This argument is ignored when graph="sbm"
                k=k,
                seed=cur_seed,
                balance=balance,
                y_model=y_model,
                graph=graph
            )
            print(f"\n=== num={num}, it={it}, seed={cur_seed} data generation completed ===")
    
            # ===== Run all methods =====
            for mid in method_ids:
                mname, mfn = METHODS[mid]
                t0 = time.time()
                try:
                    # Note: only methods 5 and 6 need device; method 4 does not
                    if mid in (1, 5, 6):
                        res = mfn(data, cur_seed, device, y_model=y_model)
                    else:
                        res = mfn(data, cur_seed, y_model=y_model)
                except Exception as e:
                    print(f"[method{mid} - {mname}] error: {e}")
                    res = {"error": str(e)}
                sec = time.time() - t0

                print(f"[method{mid} - {mname}] done in {sec:.2f}s")
                # print(res)
    
                # If an error occurs, record only one error row
                if "error" in res:
                    all_rows.append({
                        "num": num,
                        "iter": it,
                        "seed": cur_seed,
                        "graph": graph,
                        "y_model": y_model,
                        "method_id": mid,
                        "method": mname,
                        "runtime_sec": sec,
                        "error": res["error"],
                    })
                    continue
    
                # ===== 1) Summary level: save scalars only =====
                row = {
                    "num": num,
                    "iter": it,
                    "seed": cur_seed,
                    "graph": graph,
                    "y_model": y_model,
                    "method_id": mid,
                    "method": mname,
                    "runtime_sec": sec,
                }
    
                # Save available scalar values; missing keys are skipped automatically
                for key in [
                    "direct_pred", "peer_pred", "overall_pred",
                    "direct_true_overall", "peer_true_overall",
                    "direct_mse", "peer_mse",
                    "direct_true_L2", "direct_hat_L2", "peer_true_L2", "peer_hat_L2",
                    "MSE_L2_direct", "MSE_L2_peer",
                ]:
                    if key in res and np.isscalar(res[key]):
                        row[key] = res[key]
                
                # Automatically include runtime_* fields as well
                for key, val in res.items():
                    if key.startswith("runtime_") and np.isscalar(val):
                        row[key] = val
                all_rows.append(row)
    
                # ===== 2) Node-level CSV: peer / direct arrays =====
                # Save only when the method actually returns node-level arrays
                if all(
                    k in res
                    for k in [
                        "direct_pred_node", "peer_pred_node",
                        "direct_true_node", "peer_true_node",
                        "direct_sqerr_node", "peer_sqerr_node",
                    ]
                ):
                    n_nodes = len(res["direct_pred_node"])
                    node_ids = np.arange(n_nodes)
    
                    df_nodes = pd.DataFrame({
                        "num": num,
                        "iter": it,
                        "seed": cur_seed,
                        "graph": graph,
                        "y_model": y_model,
                        "method_id": mid,
                        "method": mname,
                        "node_id": node_ids,
    
                        "direct_true": res["direct_true_node"],
                        "peer_true":   res["peer_true_node"],
                        "direct_pred": res["direct_pred_node"],
                        "peer_pred":   res["peer_pred_node"],
                        "direct_sqerr": res["direct_sqerr_node"],
                        "peer_sqerr":   res["peer_sqerr_node"],
                    })
    
                    out_name = (
                        f"results/method{mid}_{mname}"
                        f"_n{num}_seed{cur_seed}_{graph}_nodes.csv"
                    )
                    df_nodes.to_csv(out_name, index=False)
                    print(f"[method{mid} - {mname}] node-level results saved to {out_name}")
                else:
                    print(f"[method{mid} - {mname}] No node-level arrays returned; skip node CSV saving.")
    
# ===== Finally save summary =====
if all_rows:
    df_summary = pd.DataFrame(all_rows)
    summary_name = f"results/methods_{','.join(map(str, method_ids))}_summary.csv"
    df_summary.to_csv(summary_name, index=False)
    print(f"\nSummary saved to {summary_name} (total {len(df_summary)} rows)")
else:
    print("No successful results; summary was not generated.")