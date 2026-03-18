"""
Name    : mvo_utils.py
Author  : William Smyth
Contact : drwss.academy@gmail.com
Date    : 26/03/2026
Desc    : shared utilities for the WIQ project. This module 
holds the generic (estimator-agnostic) parts used across the
rolling experiments, Optuna parameter searches, and performance evaluation.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from defaults import DEFAULTS

def _script_dir() -> str:
    return os.path.abspath(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()

def _ensure_monthly_prices(df_prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df_prices.index, pd.DatetimeIndex):
        raise ValueError("Price CSV must have a 'Date' index column parsable to dates.")
    return df_prices.resample("M", label="right").last()

def _load_prices_subset(csv_path: str, n_assets: int | None = None) -> pd.DataFrame:
    path = csv_path if os.path.isabs(csv_path) else os.path.join(_script_dir(), csv_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find price file at {path}.")
    df = pd.read_csv(path, parse_dates=["Date"], dayfirst=False).set_index("Date")
    if n_assets is not None and n_assets > 0:
        df = df.iloc[:, :n_assets].copy()
    return _ensure_monthly_prices(df)

def _load_csv_series(csv_path: str, column: str | None = None) -> pd.Series:
    path = csv_path if os.path.isabs(csv_path) else os.path.join(_script_dir(), csv_path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    ser = df.iloc[:, 0] if column is None else df[column]
    ser = pd.to_numeric(ser, errors="coerce").dropna()
    ser.index = pd.to_datetime(ser.index)
    return ser.sort_index()

def _asof_align(series: pd.Series, idx: pd.DatetimeIndex, method: str = "ffill") -> pd.Series:
    union = series.index.union(idx).sort_values()
    s = series.reindex(union)
    if method == "ffill":
        s = s.ffill()
    elif method == "bfill":
        s = s.bfill()
    return s.reindex(idx)

def _load_rf_aligned(idx: pd.DatetimeIndex) -> pd.Series | None:
    """Load risk-free series and align it to the provided returns index.

    If DEFAULTS['rf_interpretation']=='annualized_yield', values are interpreted
    as annual yields (%) and converted to monthly effective rates.
    """
    try:
        rf = _load_csv_series(DEFAULTS['rf_csv'], column=DEFAULTS.get('rf_column'))
        if DEFAULTS.get('rf_interpretation') == "annualized_yield":
            y = rf / 100.0
            ppy = 12 if str(DEFAULTS.get('rf_frequency', 'M')).upper().startswith('M') else 252
            rf = (1.0 + y) ** (1.0 / ppy) - 1.0
        rf_aligned = _asof_align(rf, idx, method=DEFAULTS.get('rf_align_method', 'ffill'))
        return rf_aligned.rename("rf")
    except Exception:
        return None

def annualized_sharpe(ret: pd.Series, rf: pd.Series | float | None = None) -> float:
    if rf is None:
        ex = ret
    elif isinstance(rf, (int, float)):
        ex = ret - float(rf)
    else:
        ex = (ret - rf.reindex(ret.index)).dropna()
    mu = ex.mean()
    sd = ex.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((mu / sd) * np.sqrt(12.0))

@dataclass
class MVOSettings:
    min_w: float = 0.0
    max_w: float = 1.0
    cost_bps: float = 10.0

def _max_sharpe_objective(weights: np.ndarray, mean_excess: np.ndarray, cov_monthly: np.ndarray) -> float:
    mu_ann = 12.0 * float(np.dot(mean_excess, weights))
    sd_ann = float(np.sqrt(np.dot(weights.T, np.dot(cov_monthly * 12.0, weights))))
    if sd_ann == 0 or np.isnan(sd_ann):
        return 1e6
    return -mu_ann / sd_ann

def solve_max_sharpe(
    returns_excess_window: pd.DataFrame,
    cov_monthly: np.ndarray,
    prev_weights: np.ndarray | None,
    settings: MVOSettings,
) -> np.ndarray:
    """Solve long-only max Sharpe (monthly) by SLSQP, with optional turnover penalty."""
    from scipy.optimize import minimize

    p = returns_excess_window.shape[1]
    w0 = np.full(p, 1.0 / p, dtype=float)
    bounds = tuple((settings.min_w, settings.max_w) for _ in range(p))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    mean_excess = returns_excess_window.mean().values.astype(float)

    def objective(w: np.ndarray) -> float:
        base = _max_sharpe_objective(w, mean_excess, cov_monthly)
        if prev_weights is not None:
            l1_turnover = np.abs(w - prev_weights).sum()
            base += (settings.cost_bps / 10000.0) * l1_turnover
        return base

    res = minimize(objective, x0=w0, method="SLSQP", bounds=bounds, constraints=cons)
    w = res.x
    w[np.abs(w) < 1e-6] = 0.0
    s = w.sum()
    if s <= 0:
        w = np.full(p, 1.0 / p, dtype=float)
    else:
        w = w / s
    return w

def _bisection_method(f, a, b, tol=1e-5, max_iter=100):
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("The function must have different signs at a and b.")
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c
        if fc * fa < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    raise ValueError("Maximum iterations reached without convergence.")

class TransCost:
    """Self-financing transaction cost model (bps of traded notional)."""

    def __init__(self, c_bps: float):
        self.c = float(c_bps) / 10000.0
        self.tickers: list[str] | None = None

    def _init_cost_ratio(self, new_w: list[float], old_w: list[float]) -> float:
        e = [(-1 if n_w > o_w else +1) for o_w, n_w in zip(old_w, new_w)]
        return (
            (1 - self.c * sum(w * ei for w, ei in zip(old_w, e))) /
            (1 - self.c * sum(w * ei for w, ei in zip(new_w, e)))
        )

    def _cost_func(self, new_w, old_w, cost):
        e = [(-1 if n_w > o_w else +1) for o_w, n_w in zip(old_w, new_w)]
        return 1 - self.c * sum((old_w[i] - new_w[i] * cost) * e[i] for i in range(len(new_w))) - cost

    def get_cost(self, new_weights: dict, old_weights: dict) -> float:
        self.tickers = sorted(set(new_weights.keys()).union(old_weights.keys()))
        np_new = [new_weights.get(t, 0.0) for t in self.tickers]
        np_old = [old_weights.get(t, 0.0) for t in self.tickers]

        if sum(np_new) != 0:
            s = float(sum(np_new))
            np_new = [w / s for w in np_new]
        if sum(np_old) != 0:
            s = float(sum(np_old))
            np_old = [w / s for w in np_old]

        init_cost = self._init_cost_ratio(np_new, np_old)
        f = lambda c: self._cost_func(np_new, np_old, c)

        check = f(init_cost)
        if abs(check) < 1e-10:
            return 1 - init_cost

        # ensure a bracket; if none, fall back to endpoint with smaller residual.
        f0 = f(0.0)
        f1 = f(1.0)
        if f0 * f1 >= 0:
            c_star = 0.0 if abs(f0) <= abs(f1) else 1.0
            return 1 - c_star

        root = _bisection_method(f, 0.0, 1.0)
        return 1 - root

@dataclass
class BacktestConfig:
    lookback: int
    cost_bps: float = 10.0
    initial_value: float = 100000.0
