"""
Name    : wiq_sr_tests.py
Author  : William Smyth & Layla Abu Khalaf
Contact : drwss.academy@gmail.com
Date    : 26/12/2025
Desc    : Sharpe ratio testing for WIQ project.

Philosophy:
- Treat WIQ as the benchmark.
- Ask whether ANY competitor delivers a Sharpe ratio at least as high as
  the benchmark, i.e.

    H0: SR_comp >= SR_bench  vs  H1: SR_comp < SR_bench

  which is equivalent to testing

    H0: SR_bench - SR_comp <= 0  vs  H1: SR_bench - SR_comp > 0.

- Implement this both for:
    * full-sample Sharpe ratios, using a moving-block bootstrap; and
    * rolling 36-month Sharpe ratios, using Newey–West HAC SEs and a
      moving-block bootstrap for the mean difference.

Features:
- Loads OOS after-cost EXCESS returns from *_value.csv in OOS_results/.
- Computes full-sample annualised Sharpe ratios.
- Tests full-sample mean excess-return differences (naive + Newey–West,
  two-sided; mainly descriptive).
- Tests full-sample Sharpe ratio differences via moving-block bootstrap,
  reporting:
    * SR_bench, SR_comp, SR_diff = SR_bench - SR_comp;
    * two-sided CI and p-value for SR_diff;
    * one-sided p-value for H0: SR_comp >= SR_bench.
- Computes 36-month rolling Sharpe ratios on:
    * monthly grid (step = 1),
    * thinned grid (e.g. every 3 months, step = 3).
- For each rolling grid, tests the mean difference in rolling Sharpe:
    * naive SE / t / p (two-sided);
    * HAC Newey–West SE / t (two-sided) and one-sided p-value for
      H0: mean_diff <= 0;
    * moving-block bootstrap CI for mean_diff and one-sided p-value.
- Writes results to CSV files in SR_test_results/:
    * full_sample_sharpes.csv
    * pairwise_full_sample_results.csv
    * pairwise_rolling_results.csv

Benchmarks:
- By default, WIQ vs all other methods.

To adjust robustness settings, change:
- ROLLING_STEPS
- NW_LAGS
- BLOCK_LENGTHS_MEAN
- BLOCK_LENGTH_SR
"""

import os
import math
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OOS_DIR  = os.path.join(BASE_DIR, "OOS_results")

OUTPUT_DIR = os.path.join(BASE_DIR, "SR_test_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STRATEGIES = ["WIQ","WIQ_TRUST","GS1","GS2","GS3","GS_opt","LS1","LS2","LS3",
              "LS4","LS5","NLS6","NLS7","NLS8","HC","CRE","SRE","SRE_opt"]

# risk-free configuration
RF_CSV        = os.path.join(BASE_DIR, "DGS3MO_monthly_rf.csv")
RF_DATE_COL   = "DATE"
RF_VALUE_COL  = "DGS3MO"          # monthly-effective rf return
RF_ALIGN_METH = "ffill"           # how to align rf to returns index

# rolling Sharpe parameters
ROLLING_WINDOW = 36               # window length in months
ROLLING_STEPS  = [1, 3]           # 1 = monthly grid, 3 = quarterly grid

# Newey–West truncation lags for rolling Sharpe mean tests
NW_LAGS = [6]      

# moving-block bootstrap block lengths for mean(D_t) (rolling Sharpe)
BLOCK_LENGTHS_MEAN = [12] 

# block length for full-sample Sharpe-difference bootstrap
BLOCK_LENGTH_SR = 12

# number of bootstrap replications
N_BOOT = 5000

# benchmark comparison sets
# WIQ vs all others
BENCHMARKS = {
    "WIQ": lambda names: [s for s in names if s != "WIQ"],
}

def load_rf_series(index: pd.DatetimeIndex) -> pd.Series:
    """
    Load the monthly risk-free series and align it to the given index
    using RF_ALIGN_METH.
    Interprets RF_VALUE_COL as a monthly-effective return.
    """
    if not os.path.exists(RF_CSV):
        raise FileNotFoundError(f"Risk-free CSV not found at {RF_CSV}")
    rf_df = pd.read_csv(RF_CSV, parse_dates=[RF_DATE_COL])
    rf_df = rf_df.set_index(RF_DATE_COL).sort_index()

    if RF_VALUE_COL not in rf_df.columns:
        raise KeyError(f"Column '{RF_VALUE_COL}' not found in {RF_CSV}")

    rf = rf_df[RF_VALUE_COL].astype(float)
    rf_aligned = rf.reindex(index, method=RF_ALIGN_METH)
    return rf_aligned

def load_portfolio_excess_returns(strategy: str) -> pd.Series:
    """
    From OOS_results/{strategy}_value.csv:
        - Resample to month-end (month-end label).
        - Compute monthly returns (pct_change).
        - Subtract monthly risk-free (excess returns).

    Returns a pd.Series of monthly EXCESS returns for the 'maxSharpe' portfolio.
    """
    fname = os.path.join(OOS_DIR, f"{strategy}_value.csv")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Value file for {strategy} not found: {fname}")

    df = pd.read_csv(fname, parse_dates=["date"]).set_index("date").sort_index()

    if "maxSharpe" not in df.columns:
        raise KeyError(
            f"'maxSharpe' column not found in {fname}. "
            f"Columns available: {list(df.columns)}"
        )

    # monthly values (end-of-month)
    prc = df["maxSharpe"].resample("ME", label="right").last()
    rets = prc.pct_change().dropna()

    rf = load_rf_series(rets.index)
    mask = rf.notna()
    rets = rets[mask]
    rf   = rf[mask]

    ex_rets = rets - rf
    ex_rets.name = strategy
    return ex_rets

def load_all_excess_returns() -> pd.DataFrame:
    """
    Load EXCESS returns for all strategies in STRATEGIES for which a
    *_value.csv exists. Returns a DataFrame with aligned monthly returns
    (inner join on dates).
    """
    series_dict = {}
    for strat in STRATEGIES:
        try:
            s = load_portfolio_excess_returns(strat)
            series_dict[strat] = s
            print(f"[load] Loaded excess returns for {strat} ({len(s)} months)")
        except FileNotFoundError as e:
            print(f"[load] Skipping {strat}: {e}")
        except Exception as e:
            print(f"[load] ERROR loading {strat}: {e}")

    if not series_dict:
        raise RuntimeError("No strategies loaded. Check STRATEGIES and OOS_results.")

    df = pd.concat(series_dict.values(), axis=1, join="inner").sort_index()
    print(f"[load] Final aligned excess-return panel shape: {df.shape}")
    return df

def compute_full_sample_sharpe(ex_ret: pd.Series) -> float:
    mu = ex_ret.mean()
    sigma = ex_ret.std(ddof=1)
    if sigma == 0:
        return np.nan
    return float(np.sqrt(12.0) * mu / sigma)

def compute_rolling_sharpe(ex_ret: pd.Series, window: int, step: int = 1) -> pd.Series:
    mu = ex_ret.rolling(window).mean()
    sigma = ex_ret.rolling(window).std(ddof=1)
    sr_monthly = mu / sigma
    sr_annual  = np.sqrt(12.0) * sr_monthly
    sr_annual = sr_annual.dropna()
    if step > 1:
        sr_annual = sr_annual.iloc[::step]
    return sr_annual

def newey_west_se(x: np.ndarray, max_lag: int) -> float:
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    if T <= 1:
        return np.nan

    x_mean = x.mean()
    u = x - x_mean

    gamma0 = np.mean(u * u)
    var_hat = gamma0

    for lag in range(1, max_lag + 1):
        cov = np.mean(u[lag:] * u[:-lag])
        weight = 1.0 - lag / (max_lag + 1.0)
        var_hat += 2.0 * weight * cov

    se = math.sqrt(var_hat / T)
    return se

def moving_block_bootstrap_mean(x: np.ndarray, block_length: int,
                                n_boot: int, random_state: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    if T <= 1:
        return np.array([np.nan])

    if block_length <= 0 or block_length > T:
        block_length = T

    n_blocks = math.ceil(T / block_length)
    start_indices = np.arange(0, T - block_length + 1)

    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.choice(start_indices, size=n_blocks, replace=True)
        pieces = [x[s:s+block_length] for s in starts]
        x_star = np.concatenate(pieces, axis=0)[:T]
        boot_means[b] = x_star.mean()

    return boot_means

def moving_block_bootstrap_sharpe_diff(
    pair_returns: np.ndarray,
    block_length: int,
    n_boot: int,
    random_state: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    x = np.asarray(pair_returns, dtype=float)
    T = x.shape[0]
    if T <= 1:
        return np.array([np.nan])

    if block_length <= 0 or block_length > T:
        block_length = T

    n_blocks = math.ceil(T / block_length)
    start_indices = np.arange(0, T - block_length + 1)

    boot_diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.choice(start_indices, size=n_blocks, replace=True)
        pieces = [x[s:s+block_length, :] for s in starts]
        x_star = np.concatenate(pieces, axis=0)[:T, :]

        mu = x_star.mean(axis=0)
        sigma = x_star.std(axis=0, ddof=1)
        sr = np.where(sigma == 0.0, np.nan, np.sqrt(12.0) * mu / sigma)
        boot_diffs[b] = sr[0] - sr[1]

    return boot_diffs

def test_full_sample_mean_diff(ex_df: pd.DataFrame,
                               bench: str, comp: str,
                               nw_lag: int) -> dict:
    series = ex_df[[bench, comp]].dropna()
    d = series[bench] - series[comp]
    d_np = d.to_numpy()

    T = len(d_np)
    mean_diff = float(d_np.mean())
    se_naive = float(d_np.std(ddof=1) / math.sqrt(T)) if T > 1 else np.nan
    se_nw = float(newey_west_se(d_np, nw_lag)) if T > 1 else np.nan

    t_naive = mean_diff / se_naive if se_naive not in (0, np.nan) else np.nan
    t_nw    = mean_diff / se_nw    if se_nw    not in (0, np.nan) else np.nan

    p_naive = 2 * (1 - norm.cdf(abs(t_naive))) if not np.isnan(t_naive) else np.nan
    p_nw    = 2 * (1 - norm.cdf(abs(t_nw)))    if not np.isnan(t_nw)    else np.nan

    return {
        "T": T,
        "mean_diff": mean_diff,
        "se_naive": se_naive,
        "se_nw": se_nw,
        "t_naive": t_naive,
        "t_nw": t_nw,
        "p_naive": p_naive,
        "p_nw": p_nw,
    }

def test_full_sample_sharpe_diff_bootstrap(ex_df: pd.DataFrame,
                                           bench: str, comp: str,
                                           block_length: int = BLOCK_LENGTH_SR,
                                           n_boot: int = N_BOOT,
                                           random_state: int | None = 12345) -> dict:
    series = ex_df[[bench, comp]].dropna()
    R = series.to_numpy()
    T = R.shape[0]

    mu = R.mean(axis=0)
    sigma = R.std(axis=0, ddof=1)
    sr = np.where(sigma == 0.0, np.nan, np.sqrt(12.0) * mu / sigma)
    sr_bench, sr_comp = float(sr[0]), float(sr[1])
    sr_diff = float(sr_bench - sr_comp)

    boot_diffs = moving_block_bootstrap_sharpe_diff(
        R, block_length=block_length, n_boot=n_boot, random_state=random_state
    )

    lower = float(np.quantile(boot_diffs, 0.025))
    upper = float(np.quantile(boot_diffs, 0.975))

    p_lower = (boot_diffs <= 0.0).mean()
    p_upper = (boot_diffs >= 0.0).mean()
    p_two = float(2 * min(p_lower, p_upper))
    p_one = float(p_lower)

    return {
        "T": T,
        "SR_bench": sr_bench,
        "SR_comp": sr_comp,
        "SR_diff": sr_diff,
        "boot_ci_lower": lower,
        "boot_ci_upper": upper,
        "boot_p_two": p_two,
        "boot_p_one": p_one,
    }

def test_rolling_sharpe_diff(
    rolling_sr_bench: pd.Series,
    rolling_sr_comp: pd.Series,
    label_bench: str,
    label_comp: str,
    nw_lag: int,
    block_length: int,
    random_state: int | None = 12345,
) -> dict:
    df = pd.concat(
        [rolling_sr_bench.rename(label_bench),
         rolling_sr_comp .rename(label_comp)],
        axis=1,
        join="inner",
    ).dropna()

    d = df[label_bench] - df[label_comp]
    d_np = d.to_numpy()
    T = len(d_np)
    mean_diff = float(d_np.mean())

    if T <= 1:
        return {
            "T": T,
            "mean_diff": mean_diff,
            "se_naive": np.nan,
            "se_nw": np.nan,
            "t_naive": np.nan,
            "t_nw": np.nan,
            "p_naive_two": np.nan,
            "p_nw_two": np.nan,
            "p_nw_one": np.nan,
            "boot_ci_lower": np.nan,
            "boot_ci_upper": np.nan,
            "p_boot_one": np.nan,
        }

    se_naive = float(d_np.std(ddof=1) / math.sqrt(T))
    se_nw    = float(newey_west_se(d_np, nw_lag))

    t_naive = mean_diff / se_naive if se_naive not in (0, np.nan) else np.nan
    t_nw    = mean_diff / se_nw    if se_nw    not in (0, np.nan) else np.nan

    p_naive_two = 2 * (1 - norm.cdf(abs(t_naive))) if not np.isnan(t_naive) else np.nan
    p_nw_two    = 2 * (1 - norm.cdf(abs(t_nw)))    if not np.isnan(t_nw)    else np.nan
    p_nw_one    = 1 - norm.cdf(t_nw) if not np.isnan(t_nw) else np.nan

    boot_means = moving_block_bootstrap_mean(
        d_np, block_length, N_BOOT, random_state=random_state
    )
    lower = float(np.quantile(boot_means, 0.025))
    upper = float(np.quantile(boot_means, 0.975))
    p_boot_one = float((boot_means <= 0.0).mean())

    return {
        "T": T,
        "mean_diff": mean_diff,
        "se_naive": se_naive,
        "se_nw": se_nw,
        "t_naive": t_naive,
        "t_nw": t_nw,
        "p_naive_two": p_naive_two,
        "p_nw_two": p_nw_two,
        "p_nw_one": p_nw_one,
        "boot_ci_lower": lower,
        "boot_ci_upper": upper,
        "p_boot_one": p_boot_one,
    }

def main():
    # direct console output to file as well
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(OUTPUT_DIR, f"sr_console_log_{timestamp}.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    print(f"\n[Logging console output to: {log_path}]\n")
    
    ex_df = load_all_excess_returns()
    loaded_strategies = list(ex_df.columns)
    print(f"[main] Strategies loaded: {loaded_strategies}")

    # 2. full-sample Sharpe ratios
    print("\n=== Full-sample OOS excess-return Sharpe ratios (annualised) ===")
    full_sharpes = {}
    for strat in loaded_strategies:
        sr = compute_full_sample_sharpe(ex_df[strat])
        full_sharpes[strat] = sr
        print(f"{strat:5s}: Sharpe = {sr:6.3f}")
    print()

    # save full-sample Sharpe ratios to CSV
    fs_df = pd.DataFrame({
        "strategy": list(full_sharpes.keys()),
        "full_sample_sharpe": list(full_sharpes.values()),
    })
    fs_df.to_csv(os.path.join(OUTPUT_DIR, "full_sample_sharpes.csv"), index=False)

    # 3. rolling Sharpe series for each step
    rolling_srs = {}
    for step in ROLLING_STEPS:
        print(f"[main] Computing {ROLLING_WINDOW}-month rolling Sharpe with step={step}...")
        step_dict = {}
        for strat in loaded_strategies:
            rs = compute_rolling_sharpe(ex_df[strat], ROLLING_WINDOW, step=step)
            step_dict[strat] = rs
            print(f"[roll] step={step}, {strat:5s}: {len(rs)} rolling Sharpe points")
        rolling_srs[step] = step_dict

    # containers to accumulate pairwise results for saving
    pair_full_rows = []
    pair_roll_rows = []

    # 4. tests for each benchmark vs its comparators
    for bench, comp_fn in BENCHMARKS.items():
        if bench not in loaded_strategies:
            print(f"\n[warn] Benchmark {bench} not loaded; skipping.")
            continue

        comps = [c for c in comp_fn(loaded_strategies) if c in loaded_strategies]
        if not comps:
            print(f"\n[warn] No comparators for benchmark {bench}; skipping.")
            continue

        print(f"\n================ Benchmark: {bench} =================")
        print("Comparators:", ", ".join(comps))

        for comp in comps:
            print(f"\n---- {bench} vs {comp} ----")

            baseline_nw_lag = NW_LAGS[0]

            # 4a. full-sample mean excess-return difference
            res_full = test_full_sample_mean_diff(ex_df, bench, comp, nw_lag=baseline_nw_lag)
            print("Full-sample EXCESS return difference (bench - comp):")
            print(f"  T           = {res_full['T']}")
            print(f"  mean_diff   = {res_full['mean_diff']:.6f}")
            print(f"  se_naive    = {res_full['se_naive']:.6f}, "
                  f"t_naive = {res_full['t_naive']:.3f}, "
                  f"p_naive(two-sided) = {res_full['p_naive']:.3g}")
            print(f"  se_NW(q={baseline_nw_lag}) = {res_full['se_nw']:.6f}, "
                  f"t_NW = {res_full['t_nw']:.3f}, "
                  f"p_NW(two-sided) = {res_full['p_nw']:.3g}")

            # 4b. full-sample Sharpe difference via block bootstrap
            res_sr = test_full_sample_sharpe_diff_bootstrap(ex_df, bench, comp,
                                                            block_length=BLOCK_LENGTH_SR)
            print("\nFull-sample Sharpe ratio difference (bench - comp):")
            print(f"  T           = {res_sr['T']}")
            print(f"  SR_bench    = {res_sr['SR_bench']:.3f}")
            print(f"  SR_comp     = {res_sr['SR_comp']:.3f}")
            print(f"  SR_diff     = {res_sr['SR_diff']:.3f}")
            print(f"  Boot CI (95%, two-sided) ~ [{res_sr['boot_ci_lower']:.3f}, "
                  f"{res_sr['boot_ci_upper']:.3f}]")
            print(f"  Boot p_two  (H0: SR_diff = 0)      = {res_sr['boot_p_two']:.3g}")
            print(f"  Boot p_one  (H0: SR_comp >= bench) = {res_sr['boot_p_one']:.3g}")

            # store pairwise full-sample results
            pair_full_rows.append({
                "benchmark": bench,
                "comparator": comp,
                "T": res_full["T"],
                "mean_diff": res_full["mean_diff"],
                "se_naive": res_full["se_naive"],
                "t_naive": res_full["t_naive"],
                "p_naive_two": res_full["p_naive"],
                "se_nw": res_full["se_nw"],
                "t_nw": res_full["t_nw"],
                "p_nw_two": res_full["p_nw"],
                "SR_bench": res_sr["SR_bench"],
                "SR_comp": res_sr["SR_comp"],
                "SR_diff": res_sr["SR_diff"],
                "boot_ci_lower": res_sr["boot_ci_lower"],
                "boot_ci_upper": res_sr["boot_ci_upper"],
                "boot_p_two": res_sr["boot_p_two"],
                "boot_p_one": res_sr["boot_p_one"],
            })

            # 4c. rolling Sharpe differences
            for step in ROLLING_STEPS:
                rs_bench = rolling_srs[step][bench]
                rs_comp  = rolling_srs[step][comp]
                print(f"\nRolling {ROLLING_WINDOW}-month Sharpe difference "
                      f"(bench - comp), step={step}:")
                for nw_lag in NW_LAGS:
                    for blk in BLOCK_LENGTHS_MEAN:
                        res_roll = test_rolling_sharpe_diff(
                            rs_bench, rs_comp,
                            bench, comp,
                            nw_lag=nw_lag,
                            block_length=blk,
                            random_state=12345,
                        )
                        print(
                            f"  [q={nw_lag}, L={blk}] "
                            f"T={res_roll['T']}, "
                            f"mean_diff={res_roll['mean_diff']:.6f}, "
                            f"se_naive={res_roll['se_naive']:.6f}, "
                            f"t_naive={res_roll['t_naive']:.3f}, "
                            f"p_naive(two)={res_roll['p_naive_two']:.3g}, "
                            f"se_NW={res_roll['se_nw']:.6f}, "
                            f"t_NW={res_roll['t_nw']:.3f}, "
                            f"p_NW(two)={res_roll['p_nw_two']:.3g}, "
                            f"p_NW(one H0: mean<=0)={res_roll['p_nw_one']:.3g}, "
                            f"BootCI=[{res_roll['boot_ci_lower']:.6f}, "
                            f"{res_roll['boot_ci_upper']:.6f}], "
                            f"p_boot_one(H0: mean<=0)={res_roll['p_boot_one']:.3g}"
                        )

                        # store pairwise rolling results
                        pair_roll_rows.append({
                            "benchmark": bench,
                            "comparator": comp,
                            "step": step,
                            "nw_lag": nw_lag,
                            "block_length": blk,
                            "T": res_roll["T"],
                            "mean_diff": res_roll["mean_diff"],
                            "se_naive": res_roll["se_naive"],
                            "t_naive": res_roll["t_naive"],
                            "p_naive_two": res_roll["p_naive_two"],
                            "se_nw": res_roll["se_nw"],
                            "t_nw": res_roll["t_nw"],
                            "p_nw_two": res_roll["p_nw_two"],
                            "p_nw_one": res_roll["p_nw_one"],
                            "boot_ci_lower": res_roll["boot_ci_lower"],
                            "boot_ci_upper": res_roll["boot_ci_upper"],
                            "p_boot_one": res_roll["p_boot_one"],
                        })

    # write pairwise results to CSV
    if pair_full_rows:
        pf_df = pd.DataFrame(pair_full_rows)
        pf_df.to_csv(os.path.join(OUTPUT_DIR, "pairwise_full_sample_results.csv"),
                     index=False)

    if pair_roll_rows:
        pr_df = pd.DataFrame(pair_roll_rows)
        pr_df.to_csv(os.path.join(OUTPUT_DIR, "pairwise_rolling_results.csv"),
                     index=False)

    print(f"\n[main] CSV results written to: {OUTPUT_DIR}")
    print("\n[main] Done.")

    # restore stdout
    sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()
