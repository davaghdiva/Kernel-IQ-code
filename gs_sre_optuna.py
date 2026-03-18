"""
Name    : gs_sre_optuna.py
Author  : William Smyth & Layla Abu Khalaf
Contact : drwss.academy@gmail.com
Date    : 26/12/2025
Desc    : Lightweight Optuna search for GS_opt (Gerber threshold)
          and SRE_opt (RMT2 alpha) using the same Max-Sharpe
          MVO setup as the main rolling experiment.
"""

import json
import os
import numpy as np
import pandas as pd
from defaults import DEFAULTS
from diq_mvo_optimizer import gerber_cov_stat, RMT2
from mvo_utils import (
    BacktestConfig,
    MVOSettings,
    _load_prices_subset,
    _ensure_monthly_prices,
    _load_rf_aligned,
    annualized_sharpe,
    solve_max_sharpe,
    TransCost,
)

try:
    import optuna
    _HAVE_OPTUNA = True
except Exception:
    optuna = None
    _HAVE_OPTUNA = False

def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

# generic backtests for GS/SRE using the same MVO machinery as the main experiment
def backtest_gs_mvo(
    prcs: pd.DataFrame,
    threshold: float,
    bt_cfg: BacktestConfig,
):
    """Monthly Max-Sharpe portfolio with Gerber covariance at fixed threshold."""
    prcs_m = _ensure_monthly_prices(prcs.copy())
    rets_m = prcs_m.pct_change().dropna()
    dates = rets_m.index
    symbols = prcs_m.columns.tolist()

    rf_m = _load_rf_aligned(dates)
    if rf_m is not None:
        rets_excess = rets_m.sub(rf_m, axis=0).dropna()
    else:
        rets_excess = rets_m.copy()

    tc = TransCost(c_bps=bt_cfg.cost_bps)
    values = []
    prev_weights = np.zeros(len(symbols), dtype=float)
    V = float(bt_cfg.initial_value)

    for t in range(bt_cfg.lookback, len(rets_m)):
        window_rets = rets_m.iloc[t - bt_cfg.lookback: t]
        window_excess = rets_excess.iloc[t - bt_cfg.lookback: t]
        r_next = rets_m.iloc[t].values.astype(float)

        cov_m, _ = gerber_cov_stat(window_rets.values, threshold=float(threshold))

        settings = MVOSettings(cost_bps=bt_cfg.cost_bps)
        w_new = solve_max_sharpe(window_excess, cov_m, prev_weights, settings)

        old_w = dict(zip(symbols, prev_weights))
        new_w = dict(zip(symbols, w_new))
        drag = tc.get_cost(new_weights=new_w, old_weights=old_w)
        V_after_cost = V * (1.0 - float(drag))

        port_ret = float(np.dot(w_new, r_next))
        V = V_after_cost * (1.0 + port_ret)

        prev_weights = w_new.copy()
        values.append((dates[t], V))

    df_values = pd.DataFrame(values, columns=["date", "GS_opt"]).set_index("date")
    ret_m = df_values["GS_opt"].pct_change().dropna()
    rf_eval = _load_rf_aligned(ret_m.index)
    if rf_eval is None:
        rf_eval = pd.Series(0.0, index=ret_m.index, name="rf")
    else:
        rf_eval = rf_eval.reindex(ret_m.index).ffill().rename("rf")

    return df_values, ret_m, rf_eval

def backtest_sre_mvo(
    prcs: pd.DataFrame,
    alpha: float,
    bt_cfg: BacktestConfig,
):
    """Monthly Max-Sharpe portfolio with SRE (RMT2) at fixed alpha."""
    prcs_m = _ensure_monthly_prices(prcs.copy())
    rets_m = prcs_m.pct_change().dropna()
    dates = rets_m.index
    symbols = prcs_m.columns.tolist()

    rf_m = _load_rf_aligned(dates)
    if rf_m is not None:
        rets_excess = rets_m.sub(rf_m, axis=0).dropna()
    else:
        rets_excess = rets_m.copy()

    tc = TransCost(c_bps=bt_cfg.cost_bps)
    values = []
    prev_weights = np.zeros(len(symbols), dtype=float)
    V = float(bt_cfg.initial_value)

    for t in range(bt_cfg.lookback, len(rets_m)):
        window_rets = rets_m.iloc[t - bt_cfg.lookback: t]
        window_excess = rets_excess.iloc[t - bt_cfg.lookback: t]
        r_next = rets_m.iloc[t].values.astype(float)

        # RMT2 signature: RMT2(rets, q=2, bWidth=0.01, alpha=0.1)
        cov_m, _ = RMT2(window_rets.values, alpha=float(alpha))

        settings = MVOSettings(cost_bps=bt_cfg.cost_bps)
        w_new = solve_max_sharpe(window_excess, cov_m, prev_weights, settings)

        old_w = dict(zip(symbols, prev_weights))
        new_w = dict(zip(symbols, w_new))
        drag = tc.get_cost(new_weights=new_w, old_weights=old_w)
        V_after_cost = V * (1.0 - float(drag))

        port_ret = float(np.dot(w_new, r_next))
        V = V_after_cost * (1.0 + port_ret)

        prev_weights = w_new.copy()
        values.append((dates[t], V))

    df_values = pd.DataFrame(values, columns=["date", "SRE_opt"]).set_index("date")
    ret_m = df_values["SRE_opt"].pct_change().dropna()
    rf_eval = _load_rf_aligned(ret_m.index)
    if rf_eval is None:
        rf_eval = pd.Series(0.0, index=ret_m.index, name="rf")
    else:
        rf_eval = rf_eval.reindex(ret_m.index).ffill().rename("rf")

    return df_values, ret_m, rf_eval

N_TRIALS = int(DEFAULTS.get("n_trials_gs_sre", 100))

def _optimize_gs(prcs: pd.DataFrame, bt_cfg: BacktestConfig):
    if not _HAVE_OPTUNA:
        raise RuntimeError("Optuna is not available but is required for GS_opt tuning.")

    seed = int(DEFAULTS.get("seed", 123))

    def objective(trial):
        thr = trial.suggest_float("threshold", 0.0, 1.0)
        _, ret_m, rf_m = backtest_gs_mvo(prcs, thr, bt_cfg)
        return annualized_sharpe(ret_m, rf_m)

    study = optuna.create_study(direction="maximize")
    study.sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    n_startup = max(5, int(0.2 * N_TRIALS))
    study.pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup)

    study.optimize(objective, n_trials=N_TRIALS)

    best_thr = float(study.best_params["threshold"])
    best_val = float(study.best_value)

    return {"threshold": best_thr}, best_val

def _optimize_sre(prcs: pd.DataFrame, bt_cfg: BacktestConfig):
    if not _HAVE_OPTUNA:
        raise RuntimeError("Optuna is not available but is required for SRE_opt tuning.")

    seed = int(DEFAULTS.get("seed", 123))

    def objective(trial):
        a = trial.suggest_float("alpha", 0.0, 1.0)
        _, ret_m, rf_m = backtest_sre_mvo(prcs, a, bt_cfg)
        return annualized_sharpe(ret_m, rf_m)

    study = optuna.create_study(direction="maximize")
    study.sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    n_startup = max(5, int(0.2 * N_TRIALS))
    study.pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup)

    study.optimize(objective, n_trials=N_TRIALS)

    best_alpha = float(study.best_params["alpha"])
    best_val = float(study.best_value)

    return {"alpha": best_alpha}, best_val

def main():
    # use the same training window and basic config as the main experiment
    prcs = _load_prices_subset(DEFAULTS["prices_csv"], n_assets=DEFAULTS["n_assets"])
    if DEFAULTS.get("begin_date") or DEFAULTS.get("end_date"):
        prcs = prcs.loc[DEFAULTS.get("begin_date"): DEFAULTS.get("end_date")]

    bt_cfg = BacktestConfig(
        lookback=int(DEFAULTS["lookback"]),
        cost_bps=float(DEFAULTS["cost_bps"]),
    )

    results = {}

    print(f"[gs_sre_optuna] Starting GS_opt tuning with {N_TRIALS} trials...")
    gs_params, gs_sharpe = _optimize_gs(prcs, bt_cfg)
    results["GS_opt"] = dict(params=gs_params, sharpe=gs_sharpe)

    print(f"[gs_sre_optuna] Starting SRE_opt tuning with {N_TRIALS} trials...")
    sre_params, sre_sharpe = _optimize_sre(prcs, bt_cfg)
    results["SRE_opt"] = dict(params=sre_params, sharpe=sre_sharpe)

    # output JSON files (one per method).
    export = DEFAULTS.get("export_json", {})
    gs_name = export.get("GS_opt", "gs_opt_params.json")
    sre_name = export.get("SRE_opt", "sre_opt_params.json")

    gs_path = os.path.join(_script_dir(), gs_name)
    sre_path = os.path.join(_script_dir(), sre_name)

    with open(gs_path, "w") as f:
        json.dump(gs_params, f, indent=2)
    with open(sre_path, "w") as f:
        json.dump(sre_params, f, indent=2)

    print(f"[gs_sre_optuna] GS_opt best Sharpe={gs_sharpe:.4f}; wrote {gs_path}")
    print(f"[gs_sre_optuna] SRE_opt best Sharpe={sre_sharpe:.4f}; wrote {sre_path}")

    print("\n[gs_sre_optuna] Summary:")
    for name, res in results.items():
        print(f"  {name}: Sharpe={res['sharpe']:.4f}, params={res['params']}")

if __name__ == "__main__":
    main()
