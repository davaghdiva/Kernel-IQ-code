"""
Name    : wiq_mvo.py
Author  : William Smyth
Contact : drwss.academy@gmail.com
Date    : 26/03/2026
Desc    : two-phase Optuna tuning for Kernel-IQ (WIQ) with optional pairwise trust layer.

Modes (defaults.py):
  DEFAULTS["tuning_mode"] in {"wiq","trust","auto"}

- "wiq":   Tune WIQ global parameters only. Writes wiq_params.json.
- "trust": Tune trust layer only on a frozen WIQ baseline loaded from wiq_params.json.
           Writes wiq_trust_params.json and wiq_params_trust.json.
- "auto":  Runs "wiq" then "trust" sequentially in one execution.

Notes:
- Backtest returns produced here are already EXCESS returns (returns - rf),
  so the objective uses annualized_sharpe(ret_ser) with rf=None.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from defaults import DEFAULTS
from mvo_utils import (
    BacktestConfig,
    MVOSettings,
    _load_prices_subset,
    _load_rf_aligned,
    annualized_sharpe,
    solve_max_sharpe,
)
from wiq_cov import wishart_iq_covariance
from wiq_trust_layer import compute_trust_features, apply_trust_layer

try:
    import optuna
    _HAVE_OPTUNA = True
except Exception:
    optuna = None
    _HAVE_OPTUNA = False

def _script_dir() -> str:
    return os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()

def _read_json(path_or_name: str) -> dict:
    path = path_or_name if os.path.isabs(path_or_name) else os.path.join(_script_dir(), path_or_name)
    with open(path, "r") as f:
        return json.load(f)

def _write_json(obj: dict, path_or_name: str) -> str:
    path = path_or_name if os.path.isabs(path_or_name) else os.path.join(_script_dir(), path_or_name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path

# -----------------------------
# windowing policies
# -----------------------------

def _slice_insample(prcs: pd.DataFrame) -> pd.DataFrame:
    b = DEFAULTS.get("begin_date")
    e = DEFAULTS.get("end_date")
    if b or e:
        return prcs.loc[b:e]
    return prcs

def _effective_T_live(n_assets: int) -> int:
    mode = str(DEFAULTS.get("wiq_T_mode", "fixed"))
    if mode == "q_ratio":
        q = float(DEFAULTS.get("wiq_q", 2.0))
        return int(np.ceil(q * float(n_assets)))
    return int(DEFAULTS["lookback"])

def _effective_T_scale(T_live: int) -> int:
    m = int(DEFAULTS.get("wiq_m", 3))
    return int(m * int(T_live))

def _body_asymmetry_active() -> bool:
    return (
        str(DEFAULTS.get("wiq_eta_mode", "3eta")) == "4eta"
        and (not bool(DEFAULTS.get("wiq_eta_body_equalize", True)))
        and str(DEFAULTS.get("wiq_eta_body_param", "plusminus")) == "plusminus"
    )

# allowed keys for passing into wishart_iq_covariance
WIQ_ALLOWED = {
    "eta_L","eta_B","eta_R",
    "delta_L","delta_R",
    "gamma","epsilon","threshold_c",
    "center_method","gamma_mode","gamma_max",
    "a_floor","a_mass_normalize",
    "vol_mode","ewma_halflife_mode","ewma_halflife","ewma_halflife_factor",
    "eta_mode","eta_body_equalize","eta_body_param","eta_delta_max",
    "delta_B","eta_B_pos","eta_B_neg",
}

# -----------------------------
# Backtest (objective engine)
# -----------------------------

def backtest_wiq_mvo(
    prcs: pd.DataFrame,
    wiq_params: dict,
    bt_cfg: BacktestConfig,
    trust_params: dict | None = None,
    diagnostics_path: str | None = None,
) -> pd.Series:
    """
    Monthly rebalanced Max-Sharpe backtest.

    Returns a Series of monthly EXCESS portfolio returns (already net of RF).
    """
    prcs = prcs.copy()
    rets = prcs.pct_change().dropna(axis=0)
    prcs = prcs.iloc[1:]  # align

    rf_series = _load_rf_aligned(rets.index)
    rets_excess = rets.sub(rf_series, axis=0) if rf_series is not None else rets

    nT, p = rets_excess.shape
    T_live = _effective_T_live(p)
    T_scale = _effective_T_scale(T_live)

    if nT < T_scale + 2:
        return pd.Series(dtype=float)

    settings = MVOSettings(min_w=0.0, max_w=1.0, cost_bps=float(bt_cfg.cost_bps))
    prev_w: np.ndarray | None = None

    # extract trust params (if any)
    trust_lambda = float(trust_params.get("trust_lambda", 0.0)) if trust_params else 0.0
    trust_W = trust_params.get("trust_W", None) if trust_params else None
    feature_set = str(trust_params.get("trust_feature_set", DEFAULTS.get("trust_feature_set", "basic2"))) if trust_params else str(DEFAULTS.get("trust_feature_set","basic2"))
    tail_c = float(trust_params.get("trust_tail_c", DEFAULTS.get("trust_tail_c", 3.0))) if trust_params else float(DEFAULTS.get("trust_tail_c", 3.0))

    trust_W_arr = None
    if trust_W is not None:
        trust_W_arr = np.array(trust_W, dtype=float)

    wiq_call = {k: v for k, v in wiq_params.items() if k in WIQ_ALLOWED}

    port_rets = []
    port_idx = []

    diag_rows: list[dict[str, Any]] = []

    for t in range(T_scale, nT):
        sub_excess_scale = rets_excess.iloc[t - T_scale : t]
        live_excess = sub_excess_scale.iloc[-T_live:]

        dbg: dict[str, Any] = {}
        cov_df = wishart_iq_covariance(sub_excess_scale, T_live, **wiq_call, debug_out=dbg)

        # diagnostics: effective delta in short-window volatility units
        # plus region mass summaries (dead/body/left/right) to quantify evidence usage.
        if diagnostics_path is not None and isinstance(dbg.get("delta_eff_summary", None), dict):
            end_date = rets_excess.index[t]
            s = dbg["delta_eff_summary"]
            rm = dbg.get("region_mass_summary", {}) if isinstance(dbg.get("region_mass_summary", None), dict) else {}
            diag_rows.append({
                "date": str(end_date),
                "T_live": int(T_live),
                "T_scale": int(T_scale),
                "wiq_m": int(DEFAULTS.get("wiq_m", 3)),
                "delta_L": float(wiq_call.get("delta_L")),
                "delta_R": float(wiq_call.get("delta_R")),
                **s,
                **rm,
            })

        # apply trust layer ONLY if enabled and meaningful
        if trust_lambda > 0.0 and trust_W_arr is not None:
            F = compute_trust_features(
                sub_excess_scale,
                T_live,
                feature_set=feature_set,
                tail_c=tail_c,
            )
            cov_df = apply_trust_layer(
                cov_df,
                F,
                trust_W_arr,
                lam=trust_lambda,
                offdiag_only=True,
            )

        w = solve_max_sharpe(
            returns_excess_window=live_excess,
            cov_monthly=cov_df.values.astype(float),
            prev_weights=prev_w,
            settings=settings,
        )
        prev_w = w.copy()

        # realised NEXT-period excess return (already excess)
        r_tp1 = float(np.dot(w, rets_excess.iloc[t].values.astype(float)))
        port_rets.append(r_tp1)
        port_idx.append(rets_excess.index[t])

    if diagnostics_path is not None and len(diag_rows) > 0:
        diag_df = pd.DataFrame(diag_rows)
        write_header = not os.path.exists(diagnostics_path)
        diag_df.to_csv(diagnostics_path, mode=("w" if write_header else "a"), header=write_header, index=False)

    return pd.Series(port_rets, index=pd.DatetimeIndex(port_idx), name="wiq_port_excess")

# -----------------------------
# Sampling spaces
# -----------------------------

def _wiq_base_structural() -> dict[str, Any]:
    return dict(
        threshold_c=float(DEFAULTS.get("wiq_c", 0.05)),
        epsilon=float(DEFAULTS.get("wiq_epsilon", 0.0)),
        center_method=str(DEFAULTS.get("wiq_center_method", "mean")),
        gamma_mode=str(DEFAULTS.get("gamma_mode", "signed")),
        gamma_max=float(DEFAULTS.get("u_gamma", 0.10)),
        a_floor=float(DEFAULTS.get("a_floor", 0.0)),
        a_mass_normalize=True,
        vol_mode=str(DEFAULTS.get("wiq_vol_mode", "rolling_mT")),
        ewma_halflife_mode=str(DEFAULTS.get("wiq_ewma_halflife_mode", "fixed")),
        ewma_halflife=(None if DEFAULTS.get("wiq_ewma_halflife", None) is None else float(DEFAULTS.get("wiq_ewma_halflife"))),
        ewma_halflife_factor=float(DEFAULTS.get("wiq_ewma_halflife_factor", 1.0)),
        eta_mode=str(DEFAULTS.get("wiq_eta_mode", "3eta")),
        eta_body_equalize=bool(DEFAULTS.get("wiq_eta_body_equalize", True)),
        eta_body_param=str(DEFAULTS.get("wiq_eta_body_param", "plusminus")),
        eta_delta_max=float(DEFAULTS.get("wiq_eta_delta_max", 0.20)),
        eta_B_pos=float(DEFAULTS.get("wiq_eta_B_pos", 0.50)),
        eta_B_neg=float(DEFAULTS.get("wiq_eta_B_neg", 0.50)),
    )

def _sample_wiq(trial: "optuna.trial.Trial") -> dict[str, Any]:
    eta_L = trial.suggest_float("eta_L", 0.0, 1.0)
    eta_B = trial.suggest_float("eta_B", 0.0, 1.0)
    eta_R = trial.suggest_float("eta_R", 0.0, 1.0)
    delta_L = trial.suggest_float("delta_L", 1.0, 2.0, step=0.2)
    delta_R = trial.suggest_float("delta_R", 1.0, 2.0, step=0.2)

    u = float(DEFAULTS.get("u_gamma", 0.10))
    gamma = trial.suggest_float("gamma", -u, u, step=0.02)

    wiq = dict(
        eta_L=float(eta_L),
        eta_B=float(eta_B),
        eta_R=float(eta_R),
        delta_L=float(delta_L),
        delta_R=float(delta_R),
        gamma=float(gamma),
        **_wiq_base_structural(),
    )
    if _body_asymmetry_active():
        dmax = float(DEFAULTS.get("wiq_eta_delta_max", 0.20))
        wiq["delta_B"] = float(trial.suggest_float("delta_B", -dmax, dmax, step=0.02))
    return wiq

def _sample_trust(trial: "optuna.trial.Trial") -> dict[str, Any]:
    lam_max = float(DEFAULTS.get("trust_lambda_max", 0.50))
    lam = trial.suggest_float("trust_lambda", 0.0, lam_max)

    r = int(DEFAULTS.get("trust_rank", 2))
    feature_set = str(DEFAULTS.get("trust_feature_set", "basic2"))
    tail_c = float(DEFAULTS.get("trust_tail_c", 3.0))
    W_bound = float(DEFAULTS.get("trust_W_bound", 1.0))

    # basic2 => d=2
    d = 2
    W = [[float(trial.suggest_float(f"W_{i}_{j}", -W_bound, W_bound)) for j in range(r)] for i in range(d)]

    return dict(
        trust_lambda=float(lam),
        trust_feature_set=feature_set,
        trust_rank=r,
        trust_W=W,
        trust_tail_c=tail_c,
    )

# -----------------------------
# Tuning routines
# -----------------------------

def _objective_sharpe_from_excess(ret_excess: pd.Series) -> float:
    # IMPORTANT: ret_excess is ALREADY excess returns, so rf=None.
    return float(annualized_sharpe(ret_excess, rf=None))

def tune_wiq_only(prcs: pd.DataFrame, bt_cfg: BacktestConfig, *, n_trials: int, seed: int) -> tuple[dict, float]:
    if not _HAVE_OPTUNA:
        raise RuntimeError("Optuna is required for tuning.")

    def objective(trial: "optuna.trial.Trial") -> float:
        wiq = _sample_wiq(trial)
        ret_ex = backtest_wiq_mvo(prcs, wiq, bt_cfg, trust_params=None)
        return _objective_sharpe_from_excess(ret_ex)

    study = optuna.create_study(direction="maximize")
    study.sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    study.pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, int(0.2 * n_trials)))

    # enqueue default-ish seed trial
    seed_trial = {
        "eta_L": float(DEFAULTS.get("wiq_eta_L", 0.30)),
        "eta_B": float(DEFAULTS.get("wiq_eta_B", 0.50)),
        "eta_R": float(DEFAULTS.get("wiq_eta_R", 0.30)),
        "delta_L": float(DEFAULTS.get("wiq_delta_L", 2.0)),
        "delta_R": float(DEFAULTS.get("wiq_delta_R", 2.0)),
        "gamma": float(DEFAULTS.get("wiq_gamma", 0.0)),
    }
    if _body_asymmetry_active():
        seed_trial["delta_B"] = 0.0
    study.enqueue_trial(seed_trial)

    study.optimize(objective, n_trials=n_trials)

    best_wiq = _sample_wiq(study.best_trial)
    return best_wiq, float(study.best_value)

def tune_trust_only(prcs: pd.DataFrame, wiq_frozen: dict, bt_cfg: BacktestConfig, *, n_trials: int, seed: int) -> tuple[dict, float]:
    if not _HAVE_OPTUNA:
        raise RuntimeError("Optuna is required for tuning.")

    def objective(trial: "optuna.trial.Trial") -> float:
        trust = _sample_trust(trial)
        ret_ex = backtest_wiq_mvo(prcs, wiq_frozen, bt_cfg, trust_params=trust)
        return _objective_sharpe_from_excess(ret_ex)

    study = optuna.create_study(direction="maximize")
    study.sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    study.pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, int(0.2 * n_trials)))

    # seed trust trial: lambda=0 (recover WIQ)
    r = int(DEFAULTS.get("trust_rank", 2))
    seed_trust = {"trust_lambda": 0.0}
    # provide a neutral W seed (still fine)
    for i in range(2):
        for j in range(r):
            seed_trust[f"W_{i}_{j}"] = 0.0
    study.enqueue_trial(seed_trust)

    study.optimize(objective, n_trials=n_trials)

    best_trust = _sample_trust(study.best_trial)
    return best_trust, float(study.best_value)

# -----------------------------
# Main entry
# -----------------------------

def main() -> None:
    if not _HAVE_OPTUNA:
        raise RuntimeError("Optuna is not available but is required for wiq_mvo tuning.")

    prcs = _load_prices_subset(DEFAULTS["prices_csv"], n_assets=DEFAULTS.get("n_assets"))
    prcs = _slice_insample(prcs)

    bt_cfg = BacktestConfig(
        lookback=int(DEFAULTS["lookback"]),
        cost_bps=float(DEFAULTS.get("cost_bps", 10.0)),
    )

    mode = str(DEFAULTS.get("tuning_mode", "auto")).lower().strip()
    n_trials = int(DEFAULTS.get("n_trials_wiq", 300))
    seed = int(DEFAULTS.get("seed", 260370))

    base_json  = str(DEFAULTS.get("wiq_base_params_json", "wiq_params.json"))
    trust_json = str(DEFAULTS.get("wiq_trust_params_json", "wiq_trust_params.json"))
    merged_json = str(DEFAULTS.get("wiq_params_trust_json", "wiq_params_trust.json"))

    if mode not in {"wiq","trust","auto"}:
        raise ValueError("defaults.tuning_mode must be one of {'wiq','trust','auto'}")

    # -----------------
    # Phase 1: WIQ
    # -----------------
    wiq_frozen = None
    wrote_is_diag = False
    if mode in {"wiq","auto"}:
        print(f"[wiq_mvo] Phase 1 (WIQ) tuning, trials={n_trials} ...")
        best_wiq, best_val = tune_wiq_only(prcs, bt_cfg, n_trials=n_trials, seed=seed)
        best_wiq["best_insample_sharpe"] = float(best_val)

        _write_json(best_wiq, base_json)
        print(f"[wiq_mvo] Wrote baseline WIQ params to: {os.path.join(_script_dir(), base_json)}")
        print(f"[wiq_mvo] Best in-sample Sharpe (WIQ) = {best_val:.4f}")

        wiq_frozen = {k: v for k, v in best_wiq.items() if k in WIQ_ALLOWED}

        # optional: write clean in-sample diagnostics for the frozen best WIQ parameters (NOT Optuna trial-mix).
        # this runs one backtest pass on the in-sample slice with diagnostics enabled.
        if bool(DEFAULTS.get("wiq_write_is_diagnostics", True)) and (not wrote_is_diag):
            is_name = str(DEFAULTS.get("wiq_is_diagnostics_csv", "wiq_is_best_diagnostics.csv"))
            is_path = is_name if os.path.isabs(is_name) else os.path.join(_script_dir(), is_name)
            try:
                if os.path.exists(is_path):
                    os.remove(is_path)  # ensure a clean file (avoid appending across runs)
            except Exception:
                pass
            _ = backtest_wiq_mvo(prcs, best_wiq, bt_cfg, trust_params=None, diagnostics_path=is_path)
            wrote_is_diag = True
            print(f"[wiq_mvo] Wrote IS diagnostics (frozen WIQ) to: {is_path}")

    # -----------------
    # Phase 2: TRUST
    # -----------------
    if mode in {"trust","auto"}:
        if wiq_frozen is None:
            # trust-only mode: load baseline from disk
            base_loaded = _read_json(base_json)
            wiq_frozen = {k: v for k, v in base_loaded.items() if k in WIQ_ALLOWED}

            if bool(DEFAULTS.get("wiq_write_is_diagnostics", True)) and (not wrote_is_diag):
                is_name = str(DEFAULTS.get("wiq_is_diagnostics_csv", "wiq_is_best_diagnostics.csv"))
                is_path = is_name if os.path.isabs(is_name) else os.path.join(_script_dir(), is_name)
                try:
                    if os.path.exists(is_path):
                        os.remove(is_path)
                except Exception:
                    pass
                _ = backtest_wiq_mvo(prcs, base_loaded, bt_cfg, trust_params=None, diagnostics_path=is_path)
                wrote_is_diag = True
                print(f"[wiq_mvo] Wrote IS diagnostics (frozen WIQ) to: {is_path}")

        print(f"[wiq_mvo] Phase 2 (TRUST) tuning on frozen WIQ, trials={n_trials} ...")
        best_trust, best_val = tune_trust_only(prcs, wiq_frozen, bt_cfg, n_trials=n_trials, seed=seed)
        best_trust["best_insample_sharpe"] = float(best_val)
        best_trust["wiq_base_params_json"] = base_json

        _write_json(best_trust, trust_json)

        # merged JSON = baseline WIQ file overlaid with trust params
        base_loaded = _read_json(base_json)
        merged = dict(base_loaded)
        merged.update(best_trust)
        merged["use_wiq_trust"] = True
        _write_json(merged, merged_json)

        print(f"[wiq_mvo] Wrote trust params to: {os.path.join(_script_dir(), trust_json)}")
        print(f"[wiq_mvo] Wrote merged WIQ_TRUST params to: {os.path.join(_script_dir(), merged_json)}")
        print(f"[wiq_mvo] Best in-sample Sharpe (TRUST overlay) = {best_val:.4f}")

    print("[wiq_mvo] Done.")


if __name__ == "__main__":
    main()
