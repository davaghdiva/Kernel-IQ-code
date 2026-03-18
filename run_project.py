"""
Name    : run_project.py
Author  : William Smyth
Contact : drwss.academy@gmail.com
Date    : 26/03/2026
Desc    : Main runner for the kernel-IQ project at ensemble level.
"""

from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parent

REQUIRED_SCRIPTS = [
    "run_ensemble.py",
    "aggregate_ensemble.py",
]

def check_scripts() -> None:
    missing = [name for name in REQUIRED_SCRIPTS if not (PROJECT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required script(s): {', '.join(missing)}")

def run_step(script_name: str) -> None:
    print(f"\n[run_project] Starting {script_name} ...\n")
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=str(PROJECT_DIR)
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")
    print(f"\n[run_project] Completed {script_name}.\n")

def main() -> None:
    check_scripts()
    run_step("run_ensemble.py")
    run_step("aggregate_ensemble.py")
    print("[run_project] Full project completed successfully.")

if __name__ == "__main__":
    main()