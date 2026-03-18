# Kernel-IQ FRL Replication Repository

This repository contains the replication code and supporting material for the Kernel-IQ covariance estimation experiments.

The repository is structured to support two complementary use cases:

1. **Full reproducibility:** users can run the code from scratch, including the full multi-seed ensemble experiments used in the study.
2. **Immediate inspection of results:** completed outputs are included so that users can inspect the reported results without repeating the full computational workload.

The Python scripts included here correspond closely to the code used to generate the results reported in the manuscript. The repository has been kept intentionally simple so that the computational workflow is transparent while preserving the original execution environment of the project.

---

# Repository Layout

- `*.py` in the repository root  
  Core project scripts used to run the experiments. These are kept at the repository root so that the existing import structure remains unchanged.

- `prices_multi_asset_master.csv`  
  Asset return data used by the portfolio optimisation experiments.

- `DGS3MO_monthly_rf.csv`  
  Risk-free rate series used in the Sharpe ratio calculations.

- `ENSEMBLE_RUNS.zip`  
  Archive containing the outputs from the 50-seed ensemble experiments reported in the paper. Extracting this archive reproduces the original experiment directory structure, including the per-seed outputs and the aggregated ensemble summaries used in the analysis.

---

# Running the Experiments

The code can be executed using a standard Python environment.

Create a fresh environment and install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Most users will not need to rerun the experiments, as the full ensemble outputs are already provided in `ENSEMBLE_RUNS.zip`.

To reproduce the full project workflow from scratch, run:

```bash
python run_project.py
```

This executes the ensemble stage followed by the aggregation stage.

If required, these stages can also be run separately:

```bash
python run_ensemble.py
python aggregate_ensemble.py
```

The underlying WIQ pipeline can also be run on its own using:

```bash
python run_wiq.py
```

---

# Included Results

Because the full ensemble experiment may require a substantial amount of compute time depending on hardware, this repository includes completed outputs corresponding to the experiments reported in the manuscript.

These files allow users to inspect the numerical results, tables, and performance statistics without rerunning the full experiment pipeline.

---

# Citation

If you use this code or build upon it in academic work, please cite the associated paper:

**William S. Smyth**  
*Kernel–IQ: Marginal Squeezing for Scalable Covariance Estimation*  
Finance Research Letters.

Citation metadata is provided in the file `CITATION.cff`.

---

# Reproducibility

The repository is organised so that the complete workflow from raw data to aggregated experiment outputs can be reproduced directly from the scripts provided.

---

# Notes on Repository Structure

The Python scripts are intentionally located in the repository root rather than within a `src/` directory. This preserves the original local import structure used during development and avoids unnecessary refactoring that could affect the execution of the code.

The goal of this repository is to provide a clear and faithful computational record of the experiments underlying the paper.
