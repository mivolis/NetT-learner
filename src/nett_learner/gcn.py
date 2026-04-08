from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from .types import SimulationData
from .utils import get_device, set_seed
from .smoothing import KERNEL_SMOOTHERS
from .linear import _true_direct_peer_L1, _true_direct_peer_L2

class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        x = torch.spmm(norm_adj, x)
        return self.linear(x)

class GCNMLP(nn.Module):
    def __init__(
        self,
        x_dim: int,
        hidden_features: int = 32,
        out_features: int = 1,
        dropout: float = 0.2,
        add_zg: bool = True,
        add_interactionz: bool = True,
        add_interactiong: bool = True,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.add_zg = add_zg
        self.add_interactionz = add_interactionz
        self.add_interactiong = add_interactiong
        self.gcn1 = GCNLayer(x_dim, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.ln2 = nn.LayerNorm(hidden_features)
        self.proj_xz = nn.Linear(x_dim, hidden_features)
        self.proj_xg = nn.Linear(x_dim, hidden_features)

        head_in = hidden_features + 1
        if add_zg:
            head_in += 1
        if add_interactionz:
            head_in += hidden_features
        if add_interactiong:
            head_in += hidden_features
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_features),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor, Z: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        if Z.dim() == 1:
            Z = Z.unsqueeze(1)
        if G.dim() == 1:
            G = G.unsqueeze(1)

        h = self.gcn1(X, A_norm)
        h = self.ln1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gcn2(h, A_norm)
        h = self.ln2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        feats = [h, G]
        if self.add_zg:
            feats.append(Z * G)
        if self.add_interactionz:
            feats.append(self.proj_xz(X * Z))
        if self.add_interactiong:
            feats.append(self.proj_xg(X * G))
        return self.head(torch.cat(feats, dim=1))

def _adj_to_sparse_tensor(A_normal: np.ndarray, device: torch.device) -> torch.Tensor:
    A_coo = sp.coo_matrix(A_normal)
    idx = torch.LongTensor(np.vstack((A_coo.row, A_coo.col)))
    val = torch.FloatTensor(A_coo.data)
    return torch.sparse_coo_tensor(idx, val, size=A_coo.shape, device=device)

def _train_one_epoch(model, optimizer, X_t, Z_t, G_t, Y_t, A_norm, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(X_t, A_norm, Z_t, G_t).squeeze()
    loss = F.mse_loss(out[train_mask], Y_t[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())

def _evaluate(model, X_t, Z_t, G_t, Y_t, A_norm, train_mask, test_mask):
    model.eval()
    with torch.no_grad():
        pred = model(X_t, A_norm, Z_t, G_t).squeeze()
    y_train = Y_t[train_mask].detach().cpu().numpy()
    y_test = Y_t[test_mask].detach().cpu().numpy()
    p_train = pred[train_mask].detach().cpu().numpy()
    p_test = pred[test_mask].detach().cpu().numpy()
    train_r2 = 1.0 - np.sum((p_train - y_train) ** 2) / max(np.sum((y_train - y_train.mean()) ** 2), 1e-12)
    test_r2 = 1.0 - np.sum((p_test - y_test) ** 2) / max(np.sum((y_test - y_test.mean()) ** 2), 1e-12)
    train_mse = float(np.mean((p_train - y_train) ** 2))
    test_mse = float(np.mean((p_test - y_test) ** 2))
    return train_r2, test_r2, train_mse, test_mse, pred.detach().cpu().numpy()

@dataclass
class NetTGCN:
    device: torch.device | None = None
    hidden_features: int = 32
    dropout: float = 0.2
    lr: float = 1e-2
    weight_decay: float = 0.0
    epochs: int = 800
    seed: int = 0
    print_every: int = 200
    test_ratio: float = 0.2
    max_retries: int = 3
    kernel: str = "kr_rbf"
    cv: int = 5

    def _make_model(self, x_dim: int, *, add_interactiong: bool = True) -> GCNMLP:
        return GCNMLP(
            x_dim=x_dim,
            hidden_features=self.hidden_features,
            out_features=1,
            dropout=self.dropout,
            add_zg=True,
            add_interactionz=True,
            add_interactiong=add_interactiong,
        )

    def fit(self, data: SimulationData) -> "NetTGCN":
        set_seed(self.seed)
        self.device = self.device or get_device()
        self.A_norm_ = _adj_to_sparse_tensor(data.A_normal, self.device)
        self.A_matrix_ = data.A
        self.X_ = torch.tensor(data.X_raw, dtype=torch.float32, device=self.device)
        self.Z_ = torch.tensor(data.Z.astype(float), dtype=torch.float32, device=self.device)
        self.G_ = torch.tensor(data.G.astype(float), dtype=torch.float32, device=self.device)
        self.Y_ = torch.tensor(data.Y, dtype=torch.float32, device=self.device)

        num = data.num
        rng = np.random.default_rng(self.seed)
        all_idx = np.arange(num)
        idx1 = all_idx[data.Z > 0.5]
        idx0 = all_idx[data.Z <= 0.5]
        if len(idx1) == 0 or len(idx0) == 0:
            raise ValueError("Cannot do T-learner split: both treatment groups must be present.")

        n_test = int(round(self.test_ratio * num))
        if n_test < 2:
            n_test = 2
        n_test = min(n_test, num - 1)
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

        test_mask = torch.zeros(num, dtype=torch.bool, device=self.device)
        test_mask[test_idx] = True
        train_mask = ~test_mask

        z_bool = self.Z_ > 0.5
        train_mask_Z1 = train_mask & z_bool
        train_mask_Z0 = train_mask & ~z_bool
        test_mask_Z1 = test_mask & z_bool
        test_mask_Z0 = test_mask & ~z_bool

        if train_mask_Z1.sum().item() == 0 or train_mask_Z0.sum().item() == 0:
            raise ValueError("Train split missing a treatment group.")
        if test_mask_Z1.sum().item() == 0 or test_mask_Z0.sum().item() == 0:
            raise ValueError("Test split missing a treatment group.")

        retry_count = 0
        while retry_count < self.max_retries:
            torch.manual_seed(self.seed + retry_count)
            model_Z1 = self._make_model(data.k, add_interactiong=True).to(self.device)
            model_Z10 = self._make_model(data.k, add_interactiong=True).to(self.device)
            model_Z0 = self._make_model(data.k, add_interactiong=False).to(self.device)

            optimizer_Z1 = torch.optim.AdamW(model_Z1.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            optimizer_Z10 = torch.optim.AdamW(model_Z10.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            optimizer_Z0 = torch.optim.AdamW(model_Z0.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            best_val_r2_Z1 = -float("inf")
            best_val_r2_Z10 = -float("inf")
            best_val_r2_Z0 = -float("inf")
            best_model_Z1_state = None
            best_model_Z10_state = None
            best_model_Z0_state = None

            for epoch in range(self.epochs):
                _train_one_epoch(model_Z1, optimizer_Z1, self.X_, self.Z_, self.G_, self.Y_, self.A_norm_, train_mask_Z1)
                _train_one_epoch(model_Z10, optimizer_Z10, self.X_, self.Z_, self.G_, self.Y_, self.A_norm_, train_mask_Z0)
                _train_one_epoch(model_Z0, optimizer_Z0, self.X_, self.Z_, self.G_, self.Y_, self.A_norm_, train_mask_Z0)

                _, val_r2_Z1, _, _, _ = _evaluate(model_Z1, self.X_, self.Z_, self.G_, self.Y_, self.A_norm_, train_mask_Z1, test_mask_Z1)
                _, val_r2_Z0, _, _, _ = _evaluate(model_Z0, self.X_, self.Z_, self.G_, self.Y_, self.A_norm_, train_mask_Z0, test_mask_Z0)
                _, val_r2_Z10, _, _, _ = _evaluate(model_Z10, self.X_, self.Z_, self.G_, self.Y_, self.A_norm_, train_mask_Z0, test_mask_Z0)

                if val_r2_Z1 > best_val_r2_Z1:
                    best_val_r2_Z1 = val_r2_Z1
                    best_model_Z1_state = copy.deepcopy(model_Z1.state_dict())
                if val_r2_Z0 > best_val_r2_Z0:
                    best_val_r2_Z0 = val_r2_Z0
                    best_model_Z0_state = copy.deepcopy(model_Z0.state_dict())
                if val_r2_Z10 > best_val_r2_Z10:
                    best_val_r2_Z10 = val_r2_Z10
                    best_model_Z10_state = copy.deepcopy(model_Z10.state_dict())

            if best_val_r2_Z1 < 0.001 or best_val_r2_Z0 < 0.001:
                retry_count += 1
            else:
                break

        self.model_Z1_ = self._make_model(data.k, add_interactiong=True).to(self.device)
        self.model_Z10_ = self._make_model(data.k, add_interactiong=True).to(self.device)
        self.model_Z0_ = self._make_model(data.k, add_interactiong=False).to(self.device)
        self.model_Z1_.load_state_dict(best_model_Z1_state)
        self.model_Z10_.load_state_dict(best_model_Z10_state)
        self.model_Z0_.load_state_dict(best_model_Z0_state)
        self.model_Z1_.eval()
        self.model_Z10_.eval()
        self.model_Z0_.eval()
        return self

    @torch.no_grad()
    def direct_effects_nodewise(self, data: SimulationData) -> np.ndarray:
        num_nodes = data.X_raw.shape[0]
        X_t = torch.FloatTensor(data.X_raw).to(self.device)
        G_t = torch.FloatTensor(data.G).to(self.device)
        direct_pred = np.zeros(num_nodes, dtype=float)
        for i in range(num_nodes):
            Z1 = data.Z.copy(); Z1[i] = 1.0
            Z0 = data.Z.copy(); Z0[i] = 0.0
            Z1_t = torch.FloatTensor(Z1).to(self.device)
            Z0_t = torch.FloatTensor(Z0).to(self.device)
            y1_std = self.model_Z1_(X_t, self.A_norm_, Z1_t, G_t)[i].squeeze()
            y0_std = self.model_Z10_(X_t, self.A_norm_, Z0_t, G_t)[i].squeeze()
            y1 = y1_std.cpu().numpy() * data.Y_std + data.Y_mean
            y0 = y0_std.cpu().numpy() * data.Y_std + data.Y_mean
            direct_pred[i] = y1 - y0
        return direct_pred

    @torch.no_grad()
    def peer_effects_nodewise_control_surface(self, data: SimulationData) -> np.ndarray:
        num_nodes = data.X_raw.shape[0]
        X_t = torch.FloatTensor(data.X_raw).to(self.device)
        G_t = torch.FloatTensor(data.G).to(self.device)
        peer_pred = np.zeros(num_nodes, dtype=float)
        A_noself = data.A.copy()
        np.fill_diagonal(A_noself, 0)
        for i in range(num_nodes):
            neighbors = np.where(A_noself[i] > 0)[0]
            Z_obs = data.Z.copy()
            Z_obs[i] = 0.0
            Z_g0 = Z_obs.copy()
            if len(neighbors) > 0:
                Z_g0[neighbors] = 0.0
            Z_g0_t = torch.FloatTensor(Z_g0).to(self.device)
            G0 = data.G.copy(); G0[i] = 0.0
            G_0 = torch.FloatTensor(G0).to(self.device)
            y_g_obs_std = self.model_Z0_(X_t, self.A_norm_, Z_g0_t, G_t)[i].squeeze()
            y_g0_std = self.model_Z0_(X_t, self.A_norm_, Z_g0_t, G_0)[i].squeeze()
            y_g_obs = y_g_obs_std.cpu().numpy() * data.Y_std + data.Y_mean
            y_g0 = y_g0_std.cpu().numpy() * data.Y_std + data.Y_mean
            peer_pred[i] = y_g_obs - y_g0
        return peer_pred

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
