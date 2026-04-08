from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from .types import SimulationData
from .utils import set_seed

def normalize_adj_numpy(A: np.ndarray) -> np.ndarray:
    A = A.copy()
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    return D_inv_sqrt @ A @ D_inv_sqrt

def make_sbm_adjacency(num: int, seed: int, sizes=None, q: int = 2):
    assert q == 2
    p_in = max(0.0, min(1.0, 2.0 * np.log(num) / num))
    p_out = max(0.0, min(1.0, np.log(num) / (2.0 * num)))
    if sizes is None:
        a = num // 2
        b = num - a
        sizes = [a, b]
    P = np.full((q, q), p_out, dtype=float)
    G = nx.stochastic_block_model(sizes, P, seed=seed, directed=False, selfloops=False)
    A_mat = nx.to_numpy_array(G, nodelist=range(num), dtype=int)
    np.fill_diagonal(A_mat, 1)
    blocks = np.concatenate([np.full(s, i, dtype=int) for i, s in enumerate(sizes)])
    return A_mat, blocks

def _graphon_prob(x, y, setting: int):
    X, Y = np.meshgrid(x, y, indexing="ij")
    eps = 1e-12
    if setting == 6:
        P = (X * Y) / 2.0
    else:
        raise ValueError("This package scaffold keeps setting=6, matching the original default branch.")
    P = np.clip(P, 0.0, 1.0)
    P = 0.5 * (P + P.T)
    np.fill_diagonal(P, 0.0)
    return P

def _sample_undirected_from_P(P: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    A = np.zeros((n, n), dtype=np.uint8)
    iu = np.triu_indices(n, k=1)
    A[iu] = rng.binomial(1, P[iu]).astype(np.uint8)
    return A + A.T

def make_graphon_adjacency(num: int, seed: int, setting: int = 6, coord: str = "unif", x_given=None):
    rng = np.random.default_rng(seed)
    if x_given is not None:
        x = np.asarray(x_given, dtype=float)
    else:
        x = rng.uniform(0.0, 1.0, size=num) if coord == "unif" else np.linspace(1/num, 1.0, num)
    P = _graphon_prob(x, x, setting=setting)
    A = _sample_undirected_from_P(P, seed=seed)
    A_mat = A.astype(int)
    np.fill_diagonal(A_mat, 1)
    blocks = np.zeros(num, dtype=int)
    return A_mat, blocks, P, x

def estimate_oracle_gbar_large_sample(k: int, seed: int, balance: float, large_n: int = 100000):
    rng_big = np.random.default_rng(seed + 20250320)
    x1_big = rng_big.normal(0, 1, size=large_n)
    x2_big = rng_big.normal(0, 1, size=large_n)
    X_rest_big = rng_big.normal(0, 1, (large_n, k - 2)) if k >= 2 else np.zeros((large_n, 0))
    X_big = np.column_stack((x1_big, x2_big, X_rest_big))
    intercept = -balance
    logits_big = intercept + 3 * x2_big + 4 * x1_big + X_big[:, 2:].sum(axis=1)
    logits_big = np.clip(logits_big, -709, 709)
    probs_big = expit(logits_big)
    return float(np.mean(probs_big))

def generate_data(num: int, p_edge: float, k: int, seed: int, balance: float, y_model: str, graph: str = "sbm") -> SimulationData:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0, 1, size=num)
    x2 = rng.normal(0, 1, size=num)
    X_rest = rng.normal(0, 1, (num, k - 2)) if k >= 2 else np.zeros((num, 0))
    X = np.column_stack((x1, x2, X_rest))

    if graph.lower() == "sbm":
        A_mat, blocks = make_sbm_adjacency(num=num, seed=seed, sizes=None, q=2)
    elif graph.lower() == "graphon":
        A_mat, blocks, _, _ = make_graphon_adjacency(num=num, seed=seed, setting=6, coord="unif", x_given=x1)
    else:
        A = nx.erdos_renyi_graph(num, p=p_edge, seed=seed)
        A_mat = nx.to_numpy_array(A, nodelist=range(num), dtype=int)
        blocks = np.zeros(num, dtype=int)

    degree = A_mat.sum(axis=1)
    intercept = -balance

    def gen_Z(local_seed):
        rng_local = np.random.default_rng(local_seed)
        logits = intercept + 3 * x2 + 4 * x1 + X[:, 2:].sum(axis=1)
        logits = np.clip(logits, -709, 709)
        probs = expit(logits)
        z_draw = rng_local.binomial(1, probs).astype(float)
        return z_draw, float(np.mean(probs))

    Z, G_bar = gen_Z(seed)
    tick = 1
    while len(np.unique(Z)) < 2 and tick < 50:
        Z, G_bar = gen_Z(seed + 100 * tick)
        tick += 1

    G_bar_oracle = estimate_oracle_gbar_large_sample(k=k, seed=seed, balance=balance, large_n=100000)
    np.fill_diagonal(A_mat, 1)

    beta_z = np.repeat(0.0, k)
    beta_g = np.repeat(0.0, k)
    W1 = W2 = None
    alpha1 = alpha2 = alpha3 = None

    if y_model == "linear":
        G = np.zeros(num)
        for i in range(num):
            treated_neighbors = np.sum(A_mat[i] * Z)
            total_neighbors = np.sum(A_mat[i])
            G[i] = treated_neighbors / total_neighbors if total_neighbors > 0 else 0.0

        X_neighbor = np.zeros_like(X)
        for i in range(num):
            neighbors = np.where(A_mat[i] == 1)[0]
            if len(neighbors):
                X_neighbor[i] = X[neighbors].mean(axis=0)

        eps = np.random.default_rng(seed).normal(0, 1, size=num)
        beta_z = np.array([1] * k)
        beta_g = np.array([1] * k)
        ZX = X * Z[:, None]
        GX = X * G[:, None]
        Y = (
            3
            + X[:, 0] + X[:, 1] + X[:, 2:].sum(axis=1)
            + 0.5 * X_neighbor[:, 0] + 0.5 * X_neighbor[:, 1] + 0.5 * X_neighbor[:, 2:].sum(axis=1)
            + 3 * Z
            + 2 * G
            + Z * G
            + 0.2 * ZX @ beta_z
            + 0.2 * GX @ beta_g
            + eps
        )
        ground = "linear"
    elif y_model == "gcn":
        A_normal = normalize_adj_numpy(A_mat)
        G = (A_normal @ Z).astype(float)
        X_neighbor = A_normal @ X
        hidden = 32
        W1 = rng.normal(0.0, 1.0, size=(k, hidden))
        W2 = rng.normal(0.0, 1.0, size=(hidden, 1))
        H0 = A_normal @ (X @ W1)
        H1 = np.maximum(H0, 0.0)
        H2 = H1 @ W2
        H2 = A_normal @ H2
        term_graph = np.maximum(H2, 0.0).reshape(-1)
        alpha1, alpha2, alpha3 = 3.0, 2.0, 1.0
        eps = rng.normal(0.0, 0.1, size=num)
        beta_z = np.array([1] * k)
        beta_g = np.array([1] * k)
        ZX = X * Z[:, None]
        GX = X * G[:, None]
        Y = term_graph + alpha1 * Z + alpha2 * G + alpha3 * (Z * G) + eps + 0.2 * ZX @ beta_z + 0.2 * GX @ beta_g
        ground = "gcn"
    else:
        raise ValueError("y_model must be one of {'linear', 'gcn'}")

    scaler = StandardScaler()
    Y_stdzd = scaler.fit_transform(Y.reshape(-1, 1)).flatten()
    Y_mean = float(scaler.mean_[0])
    Y_std = float(np.sqrt(scaler.var_[0]))
    A_normal = normalize_adj_numpy(A_mat)

    return SimulationData(
        y_model=y_model,
        X_raw=X,
        X_neighbor=X_neighbor,
        Z=Z,
        G=G,
        A=A_mat,
        Y=Y_stdzd,
        Y_mean=Y_mean,
        Y_std=Y_std,
        Y_raw=Y,
        G_bar=G_bar,
        G_bar_oracle=G_bar_oracle,
        beta_z=beta_z,
        beta_g=beta_g,
        A_normal=A_normal,
        num=num,
        k=k,
        ground=ground,
        degree=degree,
        blocks=blocks,
        graph_type=graph.lower(),
        W1=W1,
        W2=W2,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
    )
