"""
Name    : run_ensemble.py
Author  : William Smyth
Contact : drwss.academy@gmail.com
Date    : 26/03/2026
Desc    : runner for the [random seed ensemble] Kernel-IQ (WIQ) project.
"""

import os
import re
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULTS_PY = PROJECT_DIR / "defaults.py"
PIPELINE_CMD = [sys.executable, "run_wiq.py"]

# output folders project writes to
HEAVY_DIRS = ["OOS_results", "ranking_results", "SR_test_results"]

# Tier-1 keep list (relative to project root)
TIER1_FILES = [
    "ranking_results/overall_ranking.csv",
    "ranking_results/parsed_metrics.csv",
    "ranking_results/per_metric_ranks.csv",
    "OOS_results/performance.csv",
    "OOS_results/wiq_oos_delta_diagnostics.csv",  # optional, keep if present
    "SR_test_results/full_sample_sharpes.csv",
    "SR_test_results/pairwise_full_sample_results.csv",
    "SR_test_results/pairwise_rolling_results.csv",
    "wiq_params.json",
    "gs_opt_params.json",
    "sre_opt_params.json",
    "wiq_is_best_diagnostics.csv",
]

# optional, keep any console logs if present
KEEP_CONSOLE_LOGS = True

ENSEMBLE_ROOT = PROJECT_DIR / "ENSEMBLE_RUNS"
FIXED_CACHE_DIR = PROJECT_DIR / "OOS_fixed_cache"


def patch_seed_in_defaults(defaults_text: str, new_seed: int) -> str:
    """
    Patches a line like:
        seed            = 260370,
    to:
        seed            = <new_seed>,
    """
    pat = r"(^\s*seed\s*=\s*)\d+(\s*,\s*$)"
    if not re.search(pat, defaults_text, flags=re.MULTILINE):
        raise RuntimeError(
            "Could not find a line matching 'seed = <int>,' in defaults.py. "
            "Please ensure DEFAULTS contains a line like: seed = 777,"
        )
    return re.sub(pat, rf"\g<1>{new_seed}\g<2>", defaults_text, flags=re.MULTILINE, count=1)


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tier1_outputs(seed_dir: Path) -> List[str]:
    missing = []

    for rel in TIER1_FILES:
        src = PROJECT_DIR / rel
        if src.exists():
            safe_copy(src, seed_dir / rel)
        else:
            missing.append(rel)

    if KEEP_CONSOLE_LOGS:
        sr_root = PROJECT_DIR / "SR_test_results"
        if sr_root.exists():
            sr_logs = list(sr_root.glob("sr_console_log_*.txt"))
            for log in sr_logs:
                rel_dst = Path("SR_test_results") / log.name
                safe_copy(log, seed_dir / rel_dst)

    for p in PROJECT_DIR.glob("*params*.json"):
        safe_copy(p, seed_dir / p.name)

    for p in PROJECT_DIR.glob("*diagnostic*.csv"):
        safe_copy(p, seed_dir / p.name)

    return missing


def delete_heavy_dirs() -> None:
    for d in HEAVY_DIRS:
        p = PROJECT_DIR / d
        if p.exists() and p.is_dir():
            shutil.rmtree(p)


def run_one_seed(seed: int, retain_full: bool) -> None:
    print(f"\n=== Running seed={seed}, retain_full={retain_full} ===")

    original_text = DEFAULTS_PY.read_text(encoding="utf-8")
    patched_text = patch_seed_in_defaults(original_text, seed)
    DEFAULTS_PY.write_text(patched_text, encoding="utf-8")

    try:
        env = os.environ.copy()
        env["WIQ_USE_FIXED_CACHE"] = "1"

        subprocess.run(PIPELINE_CMD, cwd=str(PROJECT_DIR), check=True, env=env)

        seed_dir = ENSEMBLE_ROOT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "seed": seed,
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "pipeline_cmd": PIPELINE_CMD,
            "retain_full": retain_full,
            "used_fixed_cache": True,
            "fixed_cache_dir": str(FIXED_CACHE_DIR),
        }
        (seed_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if retain_full:
            for d in HEAVY_DIRS:
                src_dir = PROJECT_DIR / d
                if src_dir.exists():
                    dst_dir = seed_dir / d
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    shutil.copytree(src_dir, dst_dir)
        else:
            missing = copy_tier1_outputs(seed_dir)
            if missing:
                (seed_dir / "missing_files.txt").write_text("\n".join(missing), encoding="utf-8")
                print(f"[warn] Some Tier-1 files missing for seed={seed}. See missing_files.txt")

            delete_heavy_dirs()

        print(f"[ok] Seed {seed} saved to {seed_dir}")

    finally:
        DEFAULTS_PY.write_text(original_text, encoding="utf-8")


def main():
    ENSEMBLE_ROOT.mkdir(exist_ok=True)

    seeds_phase_a = list(range(1000, 1050))

    for seed in seeds_phase_a:
        run_one_seed(seed=seed, retain_full=False)

    print("\nPhase A complete.")
    print("The fixed deterministic baselines were cached once in OOS_fixed_cache, and later seeds only reran WIQ, GS_opt, and SRE_opt.")
    print("Next: run aggregate_ensemble.py to choose median/10th/90th seeds.")
    print("Then rerun those three seeds with retain_full=True using Phase B below.\n")

    # phase B example (uncomment after having chosen seeds)
    # seeds_phase_b = [?]
    # for seed in seeds_phase_b:
    #     run_one_seed(seed=seed, retain_full=True)

if __name__ == "__main__":
    main()
