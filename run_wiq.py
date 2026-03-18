"""
Name    : run_wiq.py
Author  : William Smyth
Contact : drwss.academy@gmail.com
Date    : 26/03/2026
Desc    : convenience runner to execute the WIQ project in sequence.

pipeline:
1. gs_sre_optuna.py
   Optional: in-sample tuning for baseline methods GS_opt (Gerber threshold)
   and SRE_opt (RMT2 alpha).

2. wiq_mvo.py
   In-sample WIQ parameter search (Optuna) and export to JSON.

3. diq_mvo.py
   Out-of-sample (Part 2) rolling Max-Sharpe backtest across covariance
   estimators (WIQ, shrinkage, and RMT baselines), producing OOS portfolio value
   paths, returns, transaction costs, and turnover.

4. diq_mvo_performance.py
   Compute performance summary tables from the OOS outputs.

5. ranking_performance.py
   Aggregate ranking across metrics.

6. wiq_sr_tests.py
   Sharpe ratio inference and robustness checks with WIQ as the benchmark.
"""

from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OOS_DIR = BASE_DIR / "OOS_results"
FIXED_CACHE_DIR = BASE_DIR / "OOS_fixed_cache"

FIXED_METHODS = [
    "HC",
    "LS1", "LS2", "LS3", "LS4", "LS5",
    "NLS6", "NLS7", "NLS8",
    "CRE", "SRE",
    "GS1", "GS2", "GS3",
]


def run_step(script_name: str, extra_env: dict[str, str] | None = None) -> None:
    script_path = BASE_DIR / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Could not find {script_name} in {BASE_DIR}")
    print(f"\n[runner] Starting {script_name} ...\n")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")
    print(f"\n[runner] Completed {script_name}.\n")


def _method_output_paths(method_name: str) -> list[Path]:
    return [
        OOS_DIR / f"{method_name}_value.csv",
        OOS_DIR / f"{method_name}_return.csv",
        OOS_DIR / f"{method_name}_trans.csv",
        OOS_DIR / f"{method_name}_turnover.csv",
    ]


def _clear_oos_dir() -> None:
    if OOS_DIR.exists():
        shutil.rmtree(OOS_DIR)
    OOS_DIR.mkdir(parents=True, exist_ok=True)


def _cache_exists() -> bool:
    return all((FIXED_CACHE_DIR / f"{method}_value.csv").exists() for method in FIXED_METHODS)


def _copy_fixed_outputs_to_cache() -> None:
    FIXED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for method in FIXED_METHODS:
        for src in _method_output_paths(method):
            if src.exists():
                shutil.copy2(src, FIXED_CACHE_DIR / src.name)
    print(f"[runner] Fixed-method cache refreshed in {FIXED_CACHE_DIR}")


def _preload_fixed_cache_into_oos() -> None:
    OOS_DIR.mkdir(parents=True, exist_ok=True)
    for method in FIXED_METHODS:
        for suffix in ("value", "return", "trans", "turnover"):
            src = FIXED_CACHE_DIR / f"{method}_{suffix}.csv"
            if not src.exists():
                raise FileNotFoundError(f"Missing cached fixed-method file: {src}")
            shutil.copy2(src, OOS_DIR / src.name)
    print(f"[runner] Preloaded fixed-method cache into {OOS_DIR}")


def main() -> None:
    use_fixed_cache = os.environ.get("WIQ_USE_FIXED_CACHE", "0").strip() == "1"

    run_step("gs_sre_optuna.py")
    run_step("wiq_mvo.py")

    if use_fixed_cache:
        if _cache_exists():
            _clear_oos_dir()
            _preload_fixed_cache_into_oos()
            run_step("diq_mvo.py", extra_env={"DIQ_RUN_MODE": "seed_varying_only"})
        else:
            _clear_oos_dir()
            run_step("diq_mvo.py", extra_env={"DIQ_RUN_MODE": "all"})
            _copy_fixed_outputs_to_cache()
    else:
        run_step("diq_mvo.py", extra_env={"DIQ_RUN_MODE": "all"})

    for script in [
        "diq_mvo_performance.py",
        "ranking_performance.py",
        "wiq_sr_tests.py",
    ]:
        run_step(script)

    print("\n[runner] All stages completed successfully.")

if __name__ == "__main__":
    main()
