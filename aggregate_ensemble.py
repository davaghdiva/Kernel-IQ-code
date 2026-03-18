"""
Name    : aggregate_ensemble.py
Author  : William Smyth
Contact : drwss.academy@gmail.com
Date    : 26/03/2026
Desc    : aggregating results across the ensemble of random seeds.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ENSEMBLE_ROOT = Path("ENSEMBLE_RUNS")
OUT_DIR = ENSEMBLE_ROOT / "_ensemble_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _find_seed_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.glob("seed_*") if p.is_dir()])

def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def _infer_method_col(df: pd.DataFrame) -> str:
    # common names first
    for c in ["Method", "method", "Estimator", "Strategy", "Unnamed: 0"]:
        if c in df.columns:
            return c
    # fall back to first column
    return df.columns[0]

def _infer_rank_col(df: pd.DataFrame) -> str:
    # prefer "Aggregate Rank" style columns
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if "rank" in lc:
            # put aggregate/overall first
            score = 0
            if "aggregate" in lc or "overall" in lc or "total" in lc:
                score -= 10
            candidates.append((score, c))
    if not candidates:
        raise RuntimeError(f"Could not infer a rank column from: {list(df.columns)}")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def main() -> None:
    seed_dirs = _find_seed_dirs(ENSEMBLE_ROOT)
    if not seed_dirs:
        raise RuntimeError("No ENSEMBLE_RUNS/seed_* folders found.")

    # -----------------------
    # 1) Aggregate ranks table
    # -----------------------
    rank_frames = []
    for sd in seed_dirs:
        seed = int(sd.name.replace("seed_", ""))
        path = sd / "ranking_results" / "overall_ranking.csv"
        if not path.exists():
            print(f"[warn] Missing {path}")
            continue

        df = _read_csv(path)
        mcol = _infer_method_col(df)
        rcol = _infer_rank_col(df)

        df = df.rename(columns={mcol: "Method", rcol: "AggregateRank"})
        df = df[["Method", "AggregateRank"]]
        df = _coerce_numeric(df, ["AggregateRank"])
        df["seed"] = seed
        rank_frames.append(df)

    if not rank_frames:
        raise RuntimeError("No overall_ranking.csv files were found/read successfully.")

    ranks = pd.concat(rank_frames, ignore_index=True)

    # % best (ties handled by counting all tied best as best for that seed)
    best_by_seed = (
        ranks.assign(min_rank=ranks.groupby("seed")["AggregateRank"].transform("min"))
             .assign(is_best=lambda x: x["AggregateRank"] == x["min_rank"])
    )
    p_best = (
        best_by_seed[best_by_seed["is_best"]]
        .groupby("Method")["seed"]
        .nunique()
        .div(best_by_seed["seed"].nunique())
        .mul(100.0)
        .rename("P_best_%")
        .reset_index()
    )

    # summary stats for ranks
    rank_summary = (
        ranks.groupby("Method")["AggregateRank"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            n="count",
        )
        .reset_index()
    )
    rank_summary = rank_summary.merge(p_best, on="Method", how="left")
    rank_summary = rank_summary.sort_values(["mean", "median"], ascending=True)

    rank_summary.to_csv(OUT_DIR / "ensemble_rank_summary.csv", index=False)

    # ---------------------------
    # 2) Aggregate parsed metrics
    # ---------------------------
    metric_frames = []
    for sd in seed_dirs:
        seed = int(sd.name.replace("seed_", ""))
        path = sd / "ranking_results" / "parsed_metrics.csv"
        if not path.exists():
            print(f"[warn] Missing {path}")
            continue

        df = _read_csv(path)
        mcol = _infer_method_col(df)
        df = df.rename(columns={mcol: "Method"})
        df["seed"] = seed

        # attempt to make all non-Method/seed columns numeric where possible
        metric_cols = [c for c in df.columns if c not in ["Method", "seed"]]
        for c in metric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        metric_frames.append(df)

    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True)

        metric_cols = [c for c in metrics.columns if c not in ["Method", "seed"]]

        # build a wide summary: (Method x metric) with mean/median/std/q25/q75
        rows = []
        for method, g in metrics.groupby("Method"):
            for c in metric_cols:
                s = g[c].dropna()
                if s.empty:
                    continue
                rows.append({
                    "Method": method,
                    "Metric": c,
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                    "q25": float(s.quantile(0.25)),
                    "q75": float(s.quantile(0.75)),
                    "n": int(s.shape[0]),
                })

        metric_summary = pd.DataFrame(rows)
        metric_summary.to_csv(OUT_DIR / "ensemble_metric_summary_long.csv", index=False)

        # convenience: extract the Sharpe-like columns if present
        sharpe_like = [c for c in metric_cols if "sharpe" in c.lower() or c.lower() == "sr"]
        if sharpe_like:
            sharpe_rows = metric_summary[metric_summary["Metric"].isin(sharpe_like)].copy()
            sharpe_rows.to_csv(OUT_DIR / "ensemble_sharpe_like_metrics.csv", index=False)

    else:
        print("[warn] No parsed_metrics.csv files found. Metric summary not created.")

    # -----------------------
    # 3) Optional: per-metric ranks
    # -----------------------
    pmr_frames = []
    for sd in seed_dirs:
        seed = int(sd.name.replace("seed_", ""))
        path = sd / "ranking_results" / "per_metric_ranks.csv"
        if not path.exists():
            continue
        df = _read_csv(path)
        mcol = _infer_method_col(df)
        df = df.rename(columns={mcol: "Method"})
        df["seed"] = seed

        # numeric conversion for rank columns
        rank_cols = [c for c in df.columns if c not in ["Method", "seed"]]
        for c in rank_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        pmr_frames.append(df)

    if pmr_frames:
        pmr = pd.concat(pmr_frames, ignore_index=True)
        rank_cols = [c for c in pmr.columns if c not in ["Method", "seed"]]

        rows = []
        for method, g in pmr.groupby("Method"):
            for c in rank_cols:
                s = g[c].dropna()
                if s.empty:
                    continue
                rows.append({
                    "Method": method,
                    "MetricRank": c,
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                    "q25": float(s.quantile(0.25)),
                    "q75": float(s.quantile(0.75)),
                    "n": int(s.shape[0]),
                })

        pmr_summary = pd.DataFrame(rows)
        pmr_summary.to_csv(OUT_DIR / "ensemble_per_metric_rank_summary_long.csv", index=False)

    print(f"\nDone. Outputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
