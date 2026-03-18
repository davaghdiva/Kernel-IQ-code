"""
Name    : defaults.py
Author  : William Smyth
Contact : drwss.academy@gmail.com
Date    : 26/03/2026
Desc    : central parameter configuration for the Kernel-IQ (WIQ) project.
"""
DEFAULTS = dict(
    # data universe
    n_assets   = 10,

    # rolling experiment settings
    lookback   = 20,          # dependence lookback T (rows)
    wiq_m      = 3,           # scaling window multiplier m (scaling window length mT)

    # transaction/optimisation settings
    cost_bps   = 10.0,        # turnover penalty (bps) in optimiser objective

    # Optuna settings
    # - gs_sre_optuna.py uses n_trials_gs_sre
    # - wiq_mvo.py uses n_trials_wiq
    seed            = 260370,
    n_trials_gs_sre = 100,
    n_trials_wiq    = 500,

    # in-sample window (tuning) and out-of-sample window for Part 2
    begin_date = "1987-12",
    end_date   = "2004-12",
    part2_begin_date = None,     # None => end_date + 1 month
    part2_end_date   = "2024-12",

    # data files
    prices_csv = "./prices_multi_asset_master.csv",
    rf_csv     = "./DGS3MO_monthly_rf.csv",

    # risk-free settings
    rf_frequency      = "M",                 # used only if interpretation == "annualized_yield"
    rf_align_method   = "ffill",             # {"ffill","bfill"}
    rf_interpretation = "monthly_effective", # {"monthly_effective","annualized_yield"}
    rf_column         = None,                # None => first column

    # temporal sequencing (shared convention)
    gamma_mode = "signed",   # {"signed","raw"}; sign selects which edge is favoured when "signed"
    u_gamma    = 0.10,       # upper bound for abs(gamma)
    a_floor    = 0.00,       # small floor in temporal weights (0 disables)

    # ------------------------
    # Kernel-IQ (WIQ) defaults
    # ------------------------
    wiq_eta_L    = 0.30,
    wiq_eta_B    = 0.50,
    wiq_eta_R    = 0.30,
    wiq_delta_L  = 2.0,
    wiq_delta_R  = 2.0,
    wiq_c        = 0.05,
    wiq_gamma    = 0.0,
    wiq_epsilon  = 0.0,

    # WIQ Optuna export
    wiq_params_json = "wiq_params.json",
    wiq_oos_diagnostics_csv = "wiq_oos_delta_diagnostics.csv",

    # ------------------------
    # WIQ_TRUST (trust layer) # unused in the FRL paper
    # ------------------------
    tuning_mode = "wiq", # "auto" | "wiq" | "trust"
    wiq_base_params_json  = "wiq_params.json",
    wiq_trust_params_json = "wiq_trust_params.json",
    wiq_params_trust_json = "wiq_params_trust.json",

    # baseline default: do NOT force trust everywhere. WIQ_TRUST uses merged JSON.
    use_wiq_trust = False,

    trust_rank = 2,
    trust_feature_set = "basic2",
    trust_lambda_max = 0.50,
    trust_W_bound = 1.0,
    trust_tail_c = 3.0,
)

# ====================================
# Kernel-IQ canonical specifications
# ====================================
# select: "prod" (stable baseline) or "research" (potentially adaptive + skew)
wiq_spec = DEFAULTS.get("wiq_spec", "research")  # "prod"|"research"

WIQ_SPEC_PROD = dict(
    # T policy
    wiq_T_mode="q_ratio",      # "fixed"|"q_ratio"
    wiq_q=2.0,                 # T_live = ceil(wiq_q * N)

    # scale window
    wiq_m=int(DEFAULTS.get("wiq_m", 3)),  # keep existing unless overridden

    # vol scaling
    wiq_vol_mode="rolling_mT", # "rolling_mT"|"ewma"

    # centering
    wiq_center_method="mean",  # "mean"|"median"|"zero"

    # ata policy
    wiq_eta_mode="4eta",            # "3eta"|"4eta"
    wiq_eta_body_equalize=True,     # True => eta_B_pos = eta_B_neg = eta_B
    wiq_eta_body_param="plusminus", # "plusminus"|"direct" (ignored when equalize=True)
    wiq_eta_delta_max=0.20,         # ignored when equalize=True

    # defaults for direct mode (only used if wiq_eta_body_param == "direct")
    wiq_eta_B_pos=0.50,
    wiq_eta_B_neg=0.50,
)

WIQ_SPEC_RESEARCH = dict(
    # T policy
    wiq_T_mode="q_ratio",
    wiq_q=2.0,

    # scale window (slightly longer to support EWMA inside the scale window)
    wiq_m=max(int(DEFAULTS.get("wiq_m", 3)), 4),

    # vol scaling (local EWMA inside the scale window)
    wiq_vol_mode="rolling_mT", #"rolling_mT" or "ewma"
    wiq_ewma_halflife_mode="proportional_to_T",  # "fixed"|"proportional_to_T"
    wiq_ewma_halflife_factor=1.0,                # half-life = factor * T_live
    wiq_ewma_halflife=None,                      # used only if halflife_mode == "fixed"

    # centering
    wiq_center_method="mean", #"mean"|"median"|"zero"

    # eta policy (controlled learnable asymmetry)
    wiq_eta_mode="4eta",
    wiq_eta_body_equalize=True,     #True amounts to 3eta
    wiq_eta_body_param="plusminus", #"plusminus"|"?"|"direct"
    wiq_eta_delta_max=0.20,

    # defaults for direct mode (only used if wiq_eta_body_param == "direct")
    wiq_eta_B_pos=0.50,
    wiq_eta_B_neg=0.50,
)

WIQ_SPECS = {"prod": WIQ_SPEC_PROD, "research": WIQ_SPEC_RESEARCH}

if wiq_spec not in WIQ_SPECS:
    raise ValueError(f"wiq_spec must be one of {list(WIQ_SPECS)}, got: {wiq_spec!r}")

# apply the chosen WIQ spec into DEFAULTS
DEFAULTS.update(WIQ_SPECS[wiq_spec])
