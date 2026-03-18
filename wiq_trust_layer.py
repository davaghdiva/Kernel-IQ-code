"""
Name    : wiq_trust_layer.py (not used in FRL paper)
Author  : William Smyth
Contact : drwss.academy@gmail.com
Date    : 26/03/2026
Desc    : pairwise "trust layer" for Kernel-IQ (WIQ) covariance matrices.

Design:
- Modify ONLY off-diagonal dependence.
- Work in correlation space.
- Preserve WIQ variances (diag of covariance).
- PSD-safe via Schur product: if G is PSD and R is PSD, then (G ⊙ R) is PSD.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def _robust_sigma_mad(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sig = 1.4826 * mad
    return float(max(sig, eps))

def compute_trust_features(
    df_returns_scale: pd.DataFrame,
    T_live: int,
    *,
    feature_set: str = "basic2",
    tail_c: float = 3.0,
    winsor_q: float = 0.01,
) -> np.ndarray:
    """
    Returns F of shape (N, d). Currently implements:
      - basic2: [log robust vol, tail exceedance rate]
    Cross-sectional winsorise + z-score.
    """
    if feature_set != "basic2":
        raise ValueError("Only feature_set='basic2' is implemented in this first pass.")

    X = df_returns_scale.values.astype(float)  # T_scale x N
    T, N = X.shape

    sig = np.array([_robust_sigma_mad(X[:, j]) for j in range(N)], dtype=float)
    log_sig = np.log(sig)

    thresh = tail_c * sig
    ex_rate = np.mean(np.abs(X) > thresh.reshape(1, -1), axis=0)

    F = np.vstack([log_sig, ex_rate]).T  # N x 2

    if winsor_q and 0 < winsor_q < 0.5:
        lo = np.quantile(F, winsor_q, axis=0)
        hi = np.quantile(F, 1.0 - winsor_q, axis=0)
        F = np.clip(F, lo.reshape(1, -1), hi.reshape(1, -1))

    mu = np.mean(F, axis=0, keepdims=True)
    sd = np.std(F, axis=0, keepdims=True)
    sd = np.clip(sd, 1e-12, None)
    return ((F - mu) / sd).astype(float)

def apply_trust_layer(
    Sigma_df: pd.DataFrame,
    F: np.ndarray,
    W: np.ndarray,
    *,
    lam: float,
    offdiag_only: bool = True,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Sigma' = D^{1/2} [ R ⊙ B ] D^{1/2}
    where B = (1-lam)*11^T + lam*G*, and G* is PSD with unit diagonal.
    If offdiag_only=True, force diag(R')=1 so diag(Sigma')==diag(Sigma).
    """
    Sigma = Sigma_df.values.astype(float)
    N = Sigma.shape[0]
    F = np.asarray(F, dtype=float)
    W = np.asarray(W, dtype=float)

    lam = float(lam)
    if lam <= 0.0:
        return Sigma_df.copy()

    d = np.clip(np.diag(Sigma).copy(), eps, None)
    s = np.sqrt(d)
    invs = 1.0 / s

    # cov -> corr
    R = (Sigma * invs).T * invs
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    # build PSD G* from U=FW
    U = F @ W
    G = U @ U.T
    gdiag = np.clip(np.diag(G).copy(), eps, None)
    invg = 1.0 / np.sqrt(gdiag)
    Gs = (G * invg).T * invg
    Gs = 0.5 * (Gs + Gs.T)
    np.fill_diagonal(Gs, 1.0)

    B = (1.0 - lam) * np.ones_like(Gs) + lam * Gs

    Rt = R * B
    Rt = 0.5 * (Rt + Rt.T)
    if offdiag_only:
        np.fill_diagonal(Rt, 1.0)

    Sigma_t = (Rt * s).T * s
    Sigma_t = 0.5 * (Sigma_t + Sigma_t.T)

    return pd.DataFrame(Sigma_t, index=Sigma_df.index, columns=Sigma_df.columns)
