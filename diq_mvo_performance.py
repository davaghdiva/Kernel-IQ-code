"""
Name    : diq_mvo_performance.py
Author  : William Smyth & Layla Abu Khalaf (adapted from https://github.com/yinsenm/gerber)
Contact : drwss.academy@gmail.com
Date    : 26/12/2025
Desc    : performance of MVO for MaxSR: evaluates WIQ and alternatives.
"""
import os
import sys
import numpy as np
import pandas as pd
from defaults import DEFAULTS as _D
from scipy.stats import norm

BASE_DIR = os.path.abspath(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
OOS_DIR  = os.path.join(BASE_DIR, "OOS_results")

STRATEGY_LABEL = 'maxSharpe'

def _load_csv_series(csv_path, column=None):
    path = csv_path if os.path.isabs(csv_path) else os.path.join(BASE_DIR, csv_path)
    df  = pd.read_csv(path, index_col=0, parse_dates=True)
    ser = df.iloc[:, 0] if column is None else df[column]
    ser = pd.to_numeric(ser, errors="coerce").dropna()
    ser.index = pd.to_datetime(ser.index)
    return ser.sort_index().rename("rf")

def _asof_align(series, idx, method="ffill"):
    union = series.index.union(idx).sort_values()
    s     = series.reindex(union)
    if method == "ffill":
        s = s.ffill()
    elif method == "bfill":
        s = s.bfill()
    return s.reindex(idx).rename(series.name)

def _load_rfr_aligned_to_returns(csv_path, returns_index,
                                 frequency      ="M",
                                 column         =None,
                                 align_method   ="ffill",
                                 interpretation ="monthly_effective"):
    rf = _load_csv_series(csv_path, column)
    if interpretation == "annualized_yield":
        y   = rf / 100.0
        ppy = 12 if str(frequency).upper().startswith("M") else 252
        rf  = (1.0 + y) ** (1.0 / ppy) - 1.0
    rf_aligned = _asof_align(rf, returns_index, method=align_method)
    return rf_aligned.rename("rf")

def _discover_cov_functions(oos_dir: str) -> list[str]:
    import glob
    covs = []
    for fp in glob.glob(os.path.join(oos_dir, "*_value.csv")):
        name = os.path.basename(fp).replace("_value.csv","")
        if name:
            covs.append(name)
    covs = sorted(set(covs))
    if not covs:
        raise RuntimeError(f"No *_value.csv files found under '{oos_dir}'. Run diq_mvo.py first.")
    return covs

def _resolve_rf(returns_index: pd.DatetimeIndex) -> pd.Series | None:
    try:
        path = _D['rf_csv']
        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)
        return _load_rfr_aligned_to_returns(
            path, returns_index,
            frequency      = _D.get('rf_frequency','M'),
            column         = _D.get('rf_column', None),
            align_method   = _D.get('rf_align_method','ffill'),
            interpretation = _D.get('rf_interpretation','monthly_effective')
        )
    except Exception as e:
        print(f"[perf] WARNING: defaults RF load failed: {e}")
        return None

def evaluate_port_performance(filename: str) -> pd.DataFrame:
    """given the path to {cov}_value.csv, compute monthly performance metrics for each column (port)."""
    prcs = pd.read_csv(filename, parse_dates=['date']).set_index(['date']).resample("M", label="right").last()
    if prcs.empty:
        raise ValueError(f"No data in {filename}")
    rets = prcs.pct_change().dropna()
    rf = _resolve_rf(rets.index)
    df_port_performances = pd.DataFrame(index=[
        'Sharpe Ratio',
        'Annualized Return (%)',
        'Cumulative Return (%)',
        'Annualized STD (%)',
        'Maximum Drawdown (%)',
        'Monthly 95% VaR (%)',
        'Sortino Ratio',
        'Calmar Ratio'  
    ])

    for port_name in prcs.columns:
        prc = prcs[port_name]
        ret = rets[port_name]

        ann_ret = 12 * ret.mean()
        ann_std = np.sqrt(12) * ret.std()
        cumulative_ret = (1 + ret).prod() - 1

        VaR2     = ret.quantile(0.05)
        rollmax  = prc.cummax()
        drawdown = prc / rollmax - 1.0
        max_dd   = drawdown.min()

        if rf is not None:
            ex_ret = (ret - rf).dropna()
        else:
            ex_ret = ret
        ex_std = ex_ret.std()
        sharpe = (ex_ret.mean() / ex_std) * np.sqrt(12) if ex_std != 0 else np.nan

        # sortino
        if rf is not None:
            base = ex_ret  
        else:
            base = ret     
        downside = base[base < 0]
        downside_std = downside.std()
        sortino = (base.mean() * np.sqrt(12)) / downside_std if downside_std != 0 else np.nan

        # calmar
        calmar = (ann_ret) / abs(max_dd) if max_dd != 0 else np.nan

        df_port_performances[port_name] = [
            sharpe,
            ann_ret * 100,
            cumulative_ret * 100,
            ann_std * 100,
            max_dd * 100,
            VaR2 * 100,
            sortino,
            calmar,
        ]

    return df_port_performances.round(2)

def main():
    prefix = OOS_DIR
    cov_function_list = _discover_cov_functions(oos_dir=prefix)

    ports_dict     = {f"${k}$": os.path.join(prefix, f"{k}_value.csv")    for k in cov_function_list}
    turnovers_dict = {f"${k}$": os.path.join(prefix, f"{k}_turnover.csv") for k in cov_function_list}

    df_ports_list = []
    for key, value in ports_dict.items():
        print(f"Evaluating {key} performance ...")
        df_port_perform = evaluate_port_performance(value)
        # attach MultiIndex columns as (port_name, covariance_key)
        df_port_perform.columns = pd.MultiIndex.from_tuples(
            [(col, key) for col in df_port_perform.columns]
        )
        df_ports_list.append(df_port_perform)

    for i, (key, value) in enumerate(turnovers_dict.items()):
        try:
            df_turnover = pd.read_csv(value, parse_dates=["date"]).set_index("date").resample("M").mean()
            df_ports_list[i].loc["Turnover"] = [round(val, 2) for val in df_turnover.mean() * 12]
        except Exception as e:
            print(f"[perf] WARNING: could not read turnover for {key}: {e}")

    df_port_performance = pd.concat(df_ports_list, axis=1)

    ports_columns = [(STRATEGY_LABEL, cov) for cov in ports_dict.keys()]
    os.makedirs(prefix, exist_ok=True)
    df_sel = df_port_performance[ports_columns]
    df_sel.to_latex(os.path.join(prefix, "performance.tex"))
    df_sel.to_csv(os.path.join(prefix, "performance.csv"))

    print(f"Exports written under {prefix} (performance.csv, performance.tex)")

if __name__ == '__main__':
    main()
