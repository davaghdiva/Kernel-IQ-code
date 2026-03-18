"""
Name    : wiq_cov.py
Author  : William Smyth
Contact : drwss.academy@gmail.com
Date    : 26/03/2026
Desc    : Kernel-IQ (WIQ) covariance builders.

This module is self-contained so the WIQ project has no external estimator
dependencies.

Operational definition (rolling use-case)
----------------------------------------
Input to WIQ is a scaling window df_returns_scale of length T_scale (rows)
by N assets (columns).

At each time step:
  - Dependence is computed using only the most recent T_live rows within
    the scaling window.
  - Volatility scaling (units for delta and c) is computed using the
    full scaling window (either rolling SD or local EWMA on the scale window).
  - After centring and standardising the live window, each entry is mapped
    to a bounded, signed value according to:
      * dead-zone: abs(X) <= c        -> 0
      * left tail: X  < -delta_L   -> -eta_L
      * body:      -delta_L <= X <= delta_R -> signed body weight
      * right tail: X  >  delta_R  -> +eta_R
    (etas constrained to [0,1])
  - Temporal sequencing is applied via signed gamma to weight time rows
    (gamma > 0 emphasises recent; gamma < 0 emphasises early) in the Gram
    accumulator.
  - Dependence is then computed via Gram form:
        G = Z^T diag(a) Z
    and converted to correlation by diagonal normalisation.
  - Covariance is returned as Sigma = D * Corr * D, where D=diag(vols_long).

Primary API:
  - wishart_iq_corr_and_cov(...): returns (Corr_df, Cov_df)
  - wishart_iq_covariance(...)  : returns Cov_df

Notes
-----
- Centering is per-column (asset) and supports: mean, median, zero.
- Vol scaling is per-column and supports:
    * rolling_mT: SD from scale window
    * ewma: local EWMA SD from scale window (half-life fixed or proportional to T_live)
- 4-eta body plumbing supports:
    * equalised: eta_B_pos = eta_B_neg = eta_B (classic behaviour)
    * plusminus: eta_B_pos = clip(eta_B + delta_B), eta_B_neg = clip(eta_B - delta_B),
                with |delta_B| capped by eta_delta_max
    * direct: eta_B_pos and eta_B_neg provided explicitly
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd

def _prep_X(df_returns: pd.DataFrame, center_method: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Return (centred matrix R, center vector mu) from a returns DataFrame."""
    X = df_returns.values.astype(float)
    if center_method == "mean":
        mu = np.mean(X, axis=0, keepdims=True)
    elif center_method == "median":
        mu = np.median(X, axis=0, keepdims=True)
    elif center_method == "zero":
        mu = np.zeros((1, X.shape[1]), dtype=float)
    else:
        raise ValueError("center_method must be one of {'mean','median','zero'}")
    return (X - mu), mu.reshape(-1)

def _stds_from_R(R: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Column-wise standard deviations with numerical clipping."""
    stds = np.std(R, axis=0)
    return np.clip(stds, eps, None)

def _ewma_stds_from_R(R: np.ndarray, half_life: float, eps: float = 1e-12) -> np.ndarray:
    """Column-wise EWMA standard deviations from a (T x N) centred matrix.

    Uses a simple recursion on squared values:
        v_t = lam * v_{t-1} + (1-lam) * x_t^2

    half_life is in 'rows' of R.
    """
    if half_life is None or half_life <= 0:
        raise ValueError("EWMA half_life must be positive.")
    lam = math.exp(math.log(0.5) / float(half_life))
    x2 = R * R
    v = x2[0].copy()
    for t in range(1, x2.shape[0]):
        v = lam * v + (1.0 - lam) * x2[t]
    stds = np.sqrt(np.clip(v, eps * eps, None))
    return np.clip(stds, eps, None)

def _resolve_body_etas(
    *,
    eta_B: float,
    eta_mode: str = "3eta",
    eta_body_equalize: bool = True,
    eta_body_param: str = "plusminus",
    eta_delta_max: float = 0.20,
    delta_B: float | None = None,
    eta_B_pos: float | None = None,
    eta_B_neg: float | None = None,
) -> tuple[float, float, float | None]:
    """Resolve (eta_B_pos, eta_B_neg, delta_B_used)."""
    eta_B = float(eta_B)
    if eta_mode not in {"3eta", "4eta"}:
        raise ValueError("eta_mode must be one of {'3eta','4eta'}")
    if eta_mode == "3eta" or bool(eta_body_equalize):
        return eta_B, eta_B, None

    eta_body_param = str(eta_body_param)
    if eta_body_param == "plusminus":
        dmax = float(eta_delta_max)
        d = float(delta_B if delta_B is not None else 0.0)
        if dmax > 0:
            d = float(np.clip(d, -dmax, dmax))
        eta_pos = float(np.clip(eta_B + d, 0.0, 1.0))
        eta_neg = float(np.clip(eta_B - d, 0.0, 1.0))
        return eta_pos, eta_neg, d
    elif eta_body_param == "direct":
        if eta_B_pos is None or eta_B_neg is None:
            raise ValueError("direct body eta requires eta_B_pos and eta_B_neg")
        eta_pos = float(eta_B_pos)
        eta_neg = float(eta_B_neg)
        if not (0.0 <= eta_pos <= 1.0 and 0.0 <= eta_neg <= 1.0):
            raise ValueError("eta_B_pos and eta_B_neg must lie in [0,1]")
        return eta_pos, eta_neg, None
    else:
        raise ValueError("eta_body_param must be one of {'plusminus','direct'}")

def _temporal_factors_np(
    T: int,
    gamma: float,
    epsil: float,
    *,
    mode: str = "signed",
    gamma_max: float | None = None,
    eps_floor: float = 0.0,
) -> np.ndarray:
    """Time-decay factors a_t."""
    gamma = float(gamma)
    epsil = float(epsil)
    if gamma_max is not None and gamma_max > 0:
        gamma = float(np.clip(gamma, -gamma_max, gamma_max))

    t = np.arange(T, dtype=float)

    if mode == "raw":
        age = (T - 1 - t)
        age = np.maximum(age - epsil, 0.0)
        a = np.exp(-gamma * age)
    elif mode == "signed":
        g_abs = abs(gamma)
        is_recent = (gamma >= 0.0)
        age_choice = (T - 1 - t) if is_recent else t
        age = np.maximum(age_choice - epsil, 0.0)
        a = np.exp(-g_abs * age)
    else:
        raise ValueError("gamma mode must be 'raw' or 'signed'")

    if eps_floor and eps_floor > 0:
        a = float(eps_floor) + (1.0 - float(eps_floor)) * a

    return a

def _to_corr_np(G: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert symmetric accumulator G to correlation with safe diagonal handling."""
    G = 0.5 * (G + G.T)
    d = np.diag(G).copy()
    mask = d <= eps
    d_safe = d.copy()
    d_safe[mask] = 1.0
    inv_sqrt = 1.0 / np.sqrt(d_safe)
    C = (G * inv_sqrt).T * inv_sqrt
    if np.any(mask):
        idx = np.where(mask)[0]
        C[idx, :] = 0.0
        C[:, idx] = 0.0
        C[idx, idx] = 1.0
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)

def wishart_iq_corr_and_cov(
    df_returns_scale: pd.DataFrame,
    T_live: int,
    *,
    eta_L: float,
    eta_B: float,
    eta_R: float,
    delta_L: float,
    delta_R: float,
    gamma: float = 0.0,
    epsilon: float = 0.0,
    threshold_c: float = 0.05,
    center_method: str = "mean",
    gamma_mode: str = "signed",
    gamma_max: float | None = 0.10,
    a_floor: float = 0.0,
    a_mass_normalize: bool = True,
    vol_mode: str = "rolling_mT",
    ewma_halflife_mode: str = "fixed",
    ewma_halflife: float | None = None,
    ewma_halflife_factor: float = 1.0,
    eta_mode: str = "3eta",
    eta_body_equalize: bool = True,
    eta_body_param: str = "plusminus",
    eta_delta_max: float = 0.20,
    delta_B: float | None = None,
    eta_B_pos: float | None = None,
    eta_B_neg: float | None = None,
    debug_out: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """compute WIQ correlation and covariance from a scaling window.

    If debug_out is provided (dict), it is populated with diagnostic arrays:
      - vols_long, vols_short, centers
      - eta_B_pos_resolved, eta_B_neg_resolved, delta_B_used
    """
    if not isinstance(df_returns_scale, pd.DataFrame):
        raise TypeError("df_returns_scale must be a pandas DataFrame.")
    if T_live <= 1:
        raise ValueError("T_live must be >= 2.")
    if df_returns_scale.shape[0] < T_live:
        raise ValueError("df_returns_scale must have at least T_live rows.")

    # validate bounded etas
    eta_L = float(eta_L)
    eta_B = float(eta_B)
    eta_R = float(eta_R)
    if not (0.0 <= eta_L <= 1.0 and 0.0 <= eta_B <= 1.0 and 0.0 <= eta_R <= 1.0):
        raise ValueError("eta_L, eta_B, eta_R must lie in [0,1].")

    delta_L = float(delta_L)
    delta_R = float(delta_R)
    if delta_L <= 0 or delta_R <= 0:
        raise ValueError("delta_L and delta_R must be positive.")

    c = float(threshold_c)
    if c < 0:
        raise ValueError("threshold_c must be non-negative.")

    # split scaling + live dependence windows
    df_live = df_returns_scale.iloc[-T_live:]

    # 1) centre
    R_live, mu_live = _prep_X(df_live, center_method=center_method)  # T_live x N
    R_scl, _mu_scl = _prep_X(df_returns_scale, center_method=center_method)  # T_scale x N

    # 2) vols from scale window
    vol_mode = str(vol_mode)
    if vol_mode == "rolling_mT":
        vols_long = _stds_from_R(R_scl)
    elif vol_mode == "ewma":
        if ewma_halflife_mode == "proportional_to_T":
            hl = float(ewma_halflife_factor) * float(T_live)
        else:
            hl = float(ewma_halflife) if ewma_halflife is not None else float(T_live)
        vols_long = _ewma_stds_from_R(R_scl, half_life=hl)
    else:
        raise ValueError("vol_mode must be one of {'rolling_mT','ewma'}")

    # 2b) short vols from live window (diagnostic only)
    vols_short = _stds_from_R(R_live)

    # 3) standardise live window
    X = R_live / vols_long

    # 4) resolve body etas
    eta_B_pos_res, eta_B_neg_res, delta_B_used = _resolve_body_etas(
        eta_B=eta_B,
        eta_mode=eta_mode,
        eta_body_equalize=eta_body_equalize,
        eta_body_param=eta_body_param,
        eta_delta_max=eta_delta_max,
        delta_B=delta_B,
        eta_B_pos=eta_B_pos,
        eta_B_neg=eta_B_neg,
    )

    # 5) dead-zone and region mapping with signed replacement
    Z = np.zeros_like(X, dtype=float)
    dead = (np.abs(X) <= c)
    left = (~dead) & (X < -delta_L)
    right = (~dead) & (X > delta_R)
    body = (~dead) & (~left) & (~right)

    # region mass diagnostics (how much data is actually used by each channel)
    # note: masks are defined on the *marginal* standardized returns X.
    # dead-zone points contribute Z=0 and hence carry no evidence into the Gram form.
    tot = float(X.size)
    dead_ct = float(np.sum(dead))
    left_ct = float(np.sum(left))
    right_ct = float(np.sum(right))
    body_ct = float(np.sum(body))
    used_ct = tot - dead_ct

    # row-wise counts for later temporal-weighted summaries.
    dead_row = np.sum(dead, axis=1).astype(float)
    left_row = np.sum(left, axis=1).astype(float)
    right_row = np.sum(right, axis=1).astype(float)
    body_row = np.sum(body, axis=1).astype(float)

    if np.any(left):
        Z[left] = -eta_L
    if np.any(right):
        Z[right] = +eta_R
    if np.any(body):
        # split body by sign
        bp = body & (X > 0)
        bn = body & (X < 0)
        if np.any(bp):
            Z[bp] = +eta_B_pos_res
        if np.any(bn):
            Z[bn] = -eta_B_neg_res
        # if exactly zero (rare after scaling), remains 0

    # 6) temporal weights
    a = _temporal_factors_np(
        X.shape[0],
        gamma,
        epsilon,
        mode=gamma_mode,
        gamma_max=gamma_max,
        eps_floor=a_floor,
    )

    if a_mass_normalize:
        s = float(np.sum(a))
        if s > 0:
            a = a * (X.shape[0] / s)

    # 7) Gram form
    Zw = Z * np.sqrt(a).reshape(-1, 1)
    G = Zw.T @ Zw

    # 8) correlation
    C = _to_corr_np(G)

    # 9) covariance
    SD = np.diag(vols_long)
    Sigma = SD @ C @ SD

    if isinstance(debug_out, dict):
        # weighted region masses (post-normalization, if enabled)
        a_sum = float(np.sum(a))
        denom = (a_sum * float(X.shape[1])) if a_sum > 0 else 1.0
        dead_w = float(np.sum(a * dead_row) / denom)
        left_w = float(np.sum(a * left_row) / denom)
        right_w = float(np.sum(a * right_row) / denom)
        body_w = float(np.sum(a * body_row) / denom)

        region_mass_summary = {
            # unweighted, fraction of all (t,i) entries
            "mass_dead": float(dead_ct / tot) if tot > 0 else 0.0,
            "mass_left": float(left_ct / tot) if tot > 0 else 0.0,
            "mass_body": float(body_ct / tot) if tot > 0 else 0.0,
            "mass_right": float(right_ct / tot) if tot > 0 else 0.0,
            "mass_used": float(used_ct / tot) if tot > 0 else 0.0,
            # weighted by temporal factors a
            "mass_dead_w": dead_w,
            "mass_left_w": left_w,
            "mass_body_w": body_w,
            "mass_right_w": right_w,
            "mass_used_w": float(1.0 - dead_w),
        }

        debug_out.update(
            {
                "centers": mu_live,
                "vols_long": vols_long,
                "vols_short": vols_short,
                "eta_B_pos_resolved": float(eta_B_pos_res),
                "eta_B_neg_resolved": float(eta_B_neg_res),
                "delta_B_used": (None if delta_B_used is None else float(delta_B_used)),
                "vol_ratio_long_over_short": (vols_long / np.clip(vols_short, 1e-12, None)),
                "delta_eff_L": float(delta_L) * (vols_long / np.clip(vols_short, 1e-12, None)),
                "delta_eff_R": float(delta_R) * (vols_long / np.clip(vols_short, 1e-12, None)),
                "delta_eff_summary": (lambda ratio, dL, dR: {
                    "vol_ratio_min": float(np.min(ratio)),
                    "vol_ratio_mean": float(np.mean(ratio)),
                    "vol_ratio_median": float(np.median(ratio)),
                    "vol_ratio_max": float(np.max(ratio)),
                    "delta_eff_L_min": float(np.min(dL)),
                    "delta_eff_L_mean": float(np.mean(dL)),
                    "delta_eff_L_median": float(np.median(dL)),
                    "delta_eff_L_max": float(np.max(dL)),
                    "delta_eff_R_min": float(np.min(dR)),
                    "delta_eff_R_mean": float(np.mean(dR)),
                    "delta_eff_R_median": float(np.median(dR)),
                    "delta_eff_R_max": float(np.max(dR)),
                })(vols_long / np.clip(vols_short, 1e-12, None),
                   float(delta_L) * (vols_long / np.clip(vols_short, 1e-12, None)),
                   float(delta_R) * (vols_long / np.clip(vols_short, 1e-12, None))),

                "region_mass_summary": region_mass_summary,

            }
        )

    cols = df_returns_scale.columns
    C_df = pd.DataFrame(C, index=cols, columns=cols)
    S_df = pd.DataFrame(Sigma, index=cols, columns=cols)
    return C_df, S_df

def wishart_iq_covariance(
    df_returns_scale: pd.DataFrame,
    T_live: int,
    *,
    eta_L: float,
    eta_B: float,
    eta_R: float,
    delta_L: float,
    delta_R: float,
    gamma: float = 0.0,
    epsilon: float = 0.0,
    threshold_c: float = 0.05,
    center_method: str = "mean",
    gamma_mode: str = "signed",
    gamma_max: float | None = 0.10,
    a_floor: float = 0.0,
    a_mass_normalize: bool = True,
    vol_mode: str = "rolling_mT",
    ewma_halflife_mode: str = "fixed",
    ewma_halflife: float | None = None,
    ewma_halflife_factor: float = 1.0,
    eta_mode: str = "3eta",
    eta_body_equalize: bool = True,
    eta_body_param: str = "plusminus",
    eta_delta_max: float = 0.20,
    delta_B: float | None = None,
    eta_B_pos: float | None = None,
    eta_B_neg: float | None = None,
    debug_out: dict | None = None,
) -> pd.DataFrame:
    """convenience wrapper returning covariance only."""
    _C, S = wishart_iq_corr_and_cov(
        df_returns_scale=df_returns_scale,
        T_live=T_live,
        eta_L=eta_L,
        eta_B=eta_B,
        eta_R=eta_R,
        delta_L=delta_L,
        delta_R=delta_R,
        gamma=gamma,
        epsilon=epsilon,
        threshold_c=threshold_c,
        center_method=center_method,
        gamma_mode=gamma_mode,
        gamma_max=gamma_max,
        a_floor=a_floor,
        a_mass_normalize=a_mass_normalize,
        vol_mode=vol_mode,
        ewma_halflife_mode=ewma_halflife_mode,
        ewma_halflife=ewma_halflife,
        ewma_halflife_factor=ewma_halflife_factor,
        eta_mode=eta_mode,
        eta_body_equalize=eta_body_equalize,
        eta_body_param=eta_body_param,
        eta_delta_max=eta_delta_max,
        delta_B=delta_B,
        eta_B_pos=eta_B_pos,
        eta_B_neg=eta_B_neg,
        debug_out=debug_out,
    )
    return S
    