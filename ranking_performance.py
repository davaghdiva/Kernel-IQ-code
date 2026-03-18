"""
Name    : ranking_performance.py
Author  : William Smyth & Layla Abu Khalaf
Contact : drwss.academy@gmail.com
Date    : 26/12/2025
Desc    : aggregating performance across methods 
          and metrics to obtain an overall winner.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "OOS_results" / "performance.csv"  
OUT_DIR = SCRIPT_DIR / "ranking_results"
OUT_DIR.mkdir(exist_ok=True)

def load_and_flatten(path: str | Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Not a MultiIndex header")
    except Exception:
        df = pd.read_csv(path, header=0, index_col=0)

    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        level1 = df.columns.get_level_values(1)
        if len(set(level1)) > 1 and (len(set(level0)) == 1 or len(set(level0)) < len(set(level1))):
            df.columns = [str(x) for x in level1]
        else:
            df.columns = [str(x) for x in level0]
    else:
        df.columns = [str(c) for c in df.columns]

    df.columns = [c.replace("$", "").strip() for c in df.columns]
    return df

HIGHER_BETTER = [
    "annualized return", "cumulative return",
    "sharpe", "sortino", "calmar"
]

LOWER_BETTER = [
    "annualized std", "std", "stdev", "vol", "volatility", "turnover",
    # risk magnitudes (we take abs() before ranking, so lower is better)
    "drawdown", "max drawdown", "maximum drawdown",
    "var", "value at risk", "cvar", "expected shortfall", "es",
]

# strings that should be treated as magnitudes for ranking
RISK_MAGNITUDE_KEYWORDS = [
    "drawdown", "max drawdown", "maximum drawdown",
    "var", "value at risk", "cvar", "expected shortfall", "es",
]

def _row_matches_any(name: str, keywords) -> bool:
    m = str(name).lower()
    return any(k in m for k in keywords)

def make_risk_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """
    drawdown and VaR/ES rows are converted to magnitudes (abs),
    so that 'lower is better' is always correct for these rows.
    """
    df2 = df.copy()
    for row in df.index:
        if _row_matches_any(row, RISK_MAGNITUDE_KEYWORDS):
            df2.loc[row] = df2.loc[row].abs()
    return df2

def metric_direction(metric: str):
    """
    returns:
      True  -> ascending=True  (lower values rank better)
      False -> ascending=False (higher values rank better)
      None  -> unknown metric name (not ranked)
    uses substring matching on lowercase names.
    """
    m = metric.lower()
    if any(k in m for k in LOWER_BETTER):
        return True
    if any(k in m for k in HIGHER_BETTER):
        return False
    return None

def rank_metrics(df: pd.DataFrame):
    ranks_rows = {}
    unknown_metrics = []

    for row in df.index:
        asc = metric_direction(str(row))
        if asc is None:
            unknown_metrics.append(str(row))
            continue
        r = df.loc[row].rank(ascending=asc, method="min")
        ranks_rows[row] = r

    ranks_df = pd.DataFrame(ranks_rows).T

    if ranks_df.empty:
        warn_path = OUT_DIR / "unmapped_metrics.txt"
        with open(warn_path, "w", encoding="utf-8") as f:
            f.write("No metrics were recognized by the ranking rules.\n")
            f.write("Metrics present in performance.csv:\n")
            for m in df.index:
                f.write(f" - {m}\n")
        raise RuntimeError(f"No rankable metrics found. See details in {warn_path}")

    # aggregate rank
    agg = ranks_df.sum(axis=0).sort_values()
    tidy = ranks_df.copy()
    tidy.loc["AGGREGATE_RANK_SUM"] = agg
    method_table = (
        tidy.T.copy()
            .rename(columns={"AGGREGATE_RANK_SUM": "Aggregate Rank (Lower=Better)"})
            .sort_values("Aggregate Rank (Lower=Better)")
    )

    # warnings if any metrics were unmapped
    if unknown_metrics:
        warn_path = OUT_DIR / "unmapped_metrics.txt"
        with open(warn_path, "w", encoding="utf-8") as f:
            f.write("The following metrics were present but not ranked (no rule matched):\n")
            for m in unknown_metrics:
                f.write(f" - {m}\n")
            f.write(
                "\nAdd appropriate keywords to HIGHER_BETTER or LOWER_BETTER "
                "to include them next time.\n"
            )
        print(f"[Warning] {len(unknown_metrics)} metric(s) not ranked. See {warn_path}")

    return ranks_df, method_table

def main():
    df = load_and_flatten(INPUT_FILE)

    # apply magnitude standardisation only for ranking
    df_for_ranking = make_risk_magnitudes(df)

    ranks_df, method_table = rank_metrics(df_for_ranking)

    df.to_csv(OUT_DIR / "parsed_metrics.csv") 
    ranks_df.to_csv(OUT_DIR / "per_metric_ranks.csv")
    method_table.to_csv(OUT_DIR / "overall_ranking.csv")

    print("\n=== Ranking complete ===")
    print(f"Files saved under {OUT_DIR.resolve()}")
    print("\nTop 5 methods:")
    print(method_table.head(5)["Aggregate Rank (Lower=Better)"])
    print("\nBottom 3 methods:")
    print(method_table.tail(3)["Aggregate Rank (Lower=Better)"])

if __name__ == "__main__":
    main()
