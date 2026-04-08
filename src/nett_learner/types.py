from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationData:
    y_model: str
    X_raw: np.ndarray
    X_neighbor: np.ndarray
    Z: np.ndarray
    G: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    Y_mean: float
    Y_std: float
    Y_raw: np.ndarray
    G_bar: float
    G_bar_oracle: float
    beta_z: np.ndarray
    beta_g: np.ndarray
    A_normal: np.ndarray
    num: int
    k: int
    ground: str
    degree: np.ndarray
    blocks: np.ndarray
    graph_type: str
    W1: np.ndarray | None = None
    W2: np.ndarray | None = None
    alpha1: float | None = None
    alpha2: float | None = None
    alpha3: float | None = None
