from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from .types import SimulationData
from .smoothing import KERNEL_SMOOTHERS
from .utils import set_seed

def _true_direct_peer_L1(data: SimulationData):
    X = data.X_raw
    G = data.G.astype(float)
    if data.y_model == "linear":
        direct_true_L1 = 3.0 + G + 0.2 * X @ data.beta_z
        peer_true_L1 = 2.0 * G + 0.2 * (X @ data.beta_g) * G
        return direct_true_L1, peer_true_L1
    direct_true_L1 = float(data.alpha1) + float(data.alpha3) * G + 0.2 * X @ data.beta_z
    peer_true_L1 = float(data.alpha2) * G + 0.2 * G * (X @ data.beta_g)
    return direct_true_L1, peer_true_L1

def _true_direct_peer_L2(data: SimulationData):
    X = data.X_raw
    Gbar = float(data.G_bar_oracle if data.G_bar_oracle is not None else data.G_bar)
    if data.y_model == "linear":
        direct_true_L2 = 3.0 + Gbar + 0.2 * X @ data.beta_z
        peer_true_L2 = 2.0 * Gbar + 0.2 * (X @ data.beta_g) * Gbar
        return direct_true_L2, peer_true_L2
    direct_true_L2 = float(data.alpha1) + float(data.alpha3) * Gbar + 0.2 * X @ data.beta_z
    peer_true_L2 = float(data.alpha2) * Gbar + 0.2 * Gbar * (X @ data.beta_g)
    return direct_true_L2, peer_true_L2

@dataclass
class NetTLinear:
    kernel: str = "kr_rbf"
    seed: int = 0
    cv: int = 5

    def fit(self, data: SimulationData) -> "NetTLinear":
        set_seed(self.seed)

        def _fit_by_treatment(treatment: float):
            idx = np.where(data.Z == treatment)[0]
            G_col = data.G.reshape(-1, 1)
            outcome = data.Y[idx]
            covariates = np.hstack((data.X_raw[idx, :], data.X_neighbor[idx, :], G_col[idx, :]))
            model = LinearRegression()
            model.fit(covariates, outcome)
            return model

        self.reg_treat_ = _fit_by_treatment(1.0)
        self.reg_control_ = _fit_by_treatment(0.0)
        return self

    def direct_effects_nodewise(self, data: SimulationData) -> np.ndarray:
        n = data.num
        out = np.zeros(n)
        for i in range(n):
            feat_g = np.hstack((data.X_raw[i, :], data.X_neighbor[i, :], data.G[i])).reshape(1, -1)
            y1_std = self.reg_treat_.predict(feat_g)[0]
            y0_std = self.reg_control_.predict(feat_g)[0]
            out[i] = (y1_std - y0_std) * data.Y_std
        return out

    def peer_effects_nodewise_control_surface(self, data: SimulationData) -> np.ndarray:
        n = data.num
        out = np.zeros(n)
        for i in range(n):
            feat_g = np.hstack((data.X_raw[i, :], data.X_neighbor[i, :], data.G[i])).reshape(1, -1)
            feat_0 = np.hstack((data.X_raw[i, :], data.X_neighbor[i, :], 0.0)).reshape(1, -1)
            y0_g_std = self.reg_control_.predict(feat_g)[0]
            y00_std = self.reg_control_.predict(feat_0)[0]
            out[i] = (y0_g_std - y00_std) * data.Y_std
        return out

    def estimate_effects(self, data: SimulationData, X_cate=None, kernel: str | None = None) -> dict:
        direct = self.direct_effects_nodewise(data)
        peer = self.peer_effects_nodewise_control_surface(data)
        direct_true_L1, peer_true_L1 = _true_direct_peer_L1(data)
        direct_true_L2, peer_true_L2 = _true_direct_peer_L2(data)

        results = {
            "direct_node": direct,
            "peer_node": peer,
            "direct_mean": float(np.mean(direct)),
            "peer_mean": float(np.mean(peer)),
            "direct_true_L1": direct_true_L1,
            "peer_true_L1": peer_true_L1,
            "direct_true_L2": direct_true_L2,
            "peer_true_L2": peer_true_L2,
        }

        if X_cate is None:
            X_cate = data.X_raw
        X_cate = np.asarray(X_cate)
        if X_cate.ndim == 1:
            X_cate = X_cate.reshape(-1, 1)
        if X_cate.shape[0] != data.num:
            raise ValueError("X_cate must have the same number of rows as the sample size.")

        selected_kernel = kernel or self.kernel
        smoother = KERNEL_SMOOTHERS[selected_kernel]
        direct_hat_L2, meta_d = smoother(X_cate, direct, cv=self.cv, return_meta=True, random_state=self.seed)
        peer_hat_L2, meta_p = smoother(X_cate, peer, cv=self.cv, return_meta=True, random_state=self.seed)

        results.update({
            "direct_cate": direct_hat_L2,
            "peer_cate": peer_hat_L2,
            "kernel_params_direct": meta_d,
            "kernel_params_peer": meta_p,
            "selected_kernel": selected_kernel,
        })
        return results
