from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def _ensure_2d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def _ensure_1d(y):
    y = np.asarray(y, dtype=float)
    return y.reshape(-1)

def kernel_smoother_kr_rbf_cv(X, y, cv: int = 5, return_meta: bool = False, random_state: int = 0):
    X = _ensure_2d(X)
    y = _ensure_1d(y)
    base = KernelRidge(kernel="rbf")
    param_grid = {
        "alpha": [1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 5e-1, 1e-1],
        "gamma": [1e-3, 5e-2, 1e-2, 5e-1, 1e-1, 1.0],
    }
    gs = GridSearchCV(base, param_grid=param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X, y)
    yhat = gs.predict(X)
    meta = {"best_params": gs.best_params_, "best_val_mse": float(-gs.best_score_)}
    return (yhat, meta) if return_meta else yhat

def kernel_smoother_mlp_cv(X, y, cv: int = 5, return_meta: bool = False, random_state: int = 0):
    X = _ensure_2d(X)
    y = _ensure_1d(y)
    model = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPRegressor(
            random_state=random_state,
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
        )),
    ])
    param_grid = {
        "mlp__hidden_layer_sizes": [(64,), (64, 64), (128, 64)],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
    }
    cv_obj = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    gs = GridSearchCV(model, param_grid=param_grid, cv=cv_obj, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True)
    gs.fit(X, y)
    yhat = gs.predict(X)
    meta = {"best_params": gs.best_params_, "best_val_mse": float(-gs.best_score_)}
    return (yhat, meta) if return_meta else yhat

def kernel_smoother_knn_uniform(X, y, cv: int = 5, return_meta: bool = False, random_state: int = 0):
    X = _ensure_2d(X)
    y = _ensure_1d(y)
    model = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("knn", KNeighborsRegressor(weights="uniform")),
    ])
    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 15, 25, 35],
        "knn__p": [1, 2],
    }
    cv_obj = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    gs = GridSearchCV(model, param_grid=param_grid, cv=cv_obj, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True)
    gs.fit(X, y)
    yhat = gs.predict(X)
    meta = {"best_params": gs.best_params_, "best_val_mse": float(-gs.best_score_)}
    return (yhat, meta) if return_meta else yhat

KERNEL_SMOOTHERS = {
    "kr_rbf": kernel_smoother_kr_rbf_cv,
    "mlp": kernel_smoother_mlp_cv,
    "knn_u": kernel_smoother_knn_uniform,
}
