"""
before-go-live/better-alpha/gate2_run.py
=========================================
Gate 2 automated AUC + IC test for one Phase 1A feature family.
See 00-OVERVIEW.md §Gate 2 for the full gating criteria.

Usage (from repo root):
    python before-go-live/better-alpha/gate2_run.py --family db_regime
    python before-go-live/better-alpha/gate2_run.py --family pead
    python before-go-live/better-alpha/gate2_run.py --family earnings

Multi-seed mode (recommended — gives variance estimate):
    python before-go-live/better-alpha/gate2_run.py --family db_regime --n-seeds 3

Holdout-adjusted baseline (run once before any family tests):
    python before-go-live/better-alpha/gate2_run.py --holdout-baseline

Steps per run:
  1. Reads AUC baseline from baseline_v1_auc_holdout.txt (preferred)
     or baseline_v1_auc.txt (fallback, full-history baseline).
  2. Runs run_ml_pipeline.py with --enable-{family} (holdout filter
     is applied automatically since holdout_config.txt exists).
     With --n-seeds N, repeats with N different seeds and averages.
  3. Reads new per-ticker AUC from ml_state.json.
  4. IC gate: DEFERRED until n_obs >= 10 in alpha_ic_results.csv.
     (IC comes from live DB signals, not from the retrained model —
     it cannot change from a pipeline retraining run.)
  5. Evaluates Gate 2 criteria from 00-OVERVIEW.md.
  6. Appends a row to gate2_results.csv.
  7. Prints PASS / FAIL with full criteria breakdown.

Gate 2 PASS (ALL hard criteria must be met):
  ✓ delta_auc > +0.003 (primary gate — always active)
  ✓ delta_auc_std ≤ +0.010 (AUC consistency must not worsen)
  ✓ No ticker drops below AUC 0.50 that was previously above 0.53
  ✓ delta_ic > +0.003 [deferred if n_obs < 10 — becomes Gate 3 IC check]
  ✓ delta_icir ≥ -0.05 [deferred if n_obs < 10]

AUC baseline note:
  If baseline_v1_auc_holdout.txt exists, it is used as the comparison
  target (holdout-filtered, apples-to-apples with Gate 2 runs).
  Otherwise falls back to baseline_v1_auc.txt (full history, slightly
  conservative — see gate2_audit.md for the methodology analysis).

On PASS: update PROJECT-STATE.md and set the flag to True in
         run_ml_pipeline.py. Wait for Gate 3 (2 Saturday live runs).
On FAIL: keep the flag False. Investigate the failing criterion.
         Do NOT proceed to the next family without understanding why.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
ML_LAB = os.path.join(ROOT, 'ml_quant_finance_research', 'ml_research', 'stock_ml_lab')

BASELINE_FILE          = os.path.join(HERE, 'baseline_v1_auc.txt')
BASELINE_HOLDOUT_FILE  = os.path.join(HERE, 'baseline_v1_auc_holdout.txt')
GATE2_CSV              = os.path.join(HERE, 'gate2_results.csv')
ML_STATE_PATH          = os.path.join(ROOT, 'shared', 'state', 'ml_state.json')
IC_RESULTS_PATH        = os.path.join(ROOT, 'backtests', 'alpha_ic_results.csv')

FAMILY_FLAG_MAP = {
    'db_regime': 'ENABLE_DB_REGIME_FEATURES',
    'pead':      'ENABLE_PEAD_FEATURES',
    'earnings':  'ENABLE_EARNINGS_CALENDAR_FEATURES',
    'crosssectional': 'ENABLE_CROSSSECTIONAL_FEATURES',
    'acceleration':   'ENABLE_ACCELERATION_FEATURES',
    'target_refinement': 'ENABLE_ALPHA_TARGET',
    'stationary_only': 'ENABLE_STATIONARY_ONLY',
    'regularized_models': 'ENABLE_REGULARIZED_MODELS',
    'short_volume': 'ENABLE_SHORT_VOLUME_FEATURES',
    'combo_d': 'COMBO_D (Stationary + Regularized)',
}
CLI_FLAG_MAP = {
    'db_regime': '--enable-db-regime',
    'pead':      '--enable-pead',
    'earnings':  '--enable-earnings',
    'crosssectional': '--enable-crosssectional',
    'acceleration':   '--enable-acceleration',
    'target_refinement': '--enable-alpha-target',
    'stationary_only': '--enable-stationary-only',
    'regularized_models': ['--enable-regularized-models'],
    'short_volume': ['--enable-short-volume'],
    'combo_d': ['--enable-stationary-only', '--enable-regularized-models'],
}

# Gate 2 thresholds (from 00-OVERVIEW.md)
THRESHOLD_DELTA_AUC  = 0.003
THRESHOLD_DELTA_STD  = 0.010   # must not increase by more than this
THRESHOLD_DELTA_ICIR = -0.05   # must not decrease by more than this
THRESHOLD_DELTA_IC   = 0.003
MIN_OBS_FOR_IC_GATE  = 10      # IC gate deferred below this n_obs

# Seeds for multi-seed variance estimation
DEFAULT_SEEDS = [42, 123, 7]


# ─────────────────────────────────────────────────────────────────────────────
def parse_baseline(prefer_holdout: bool = True):
    """Read baseline_v1_auc*.txt → dict.

    If prefer_holdout is True and baseline_v1_auc_holdout.txt exists, use it.
    Otherwise fall back to baseline_v1_auc.txt.
    Returns (result_dict, source_file).
    """
    chosen = None
    if prefer_holdout and os.path.exists(BASELINE_HOLDOUT_FILE):
        chosen = BASELINE_HOLDOUT_FILE
    elif os.path.exists(BASELINE_FILE):
        chosen = BASELINE_FILE
    else:
        print(f"ERROR: Neither {BASELINE_HOLDOUT_FILE} nor {BASELINE_FILE} found.")
        print("Run gate0_baseline.py first, then optionally --holdout-baseline.")
        sys.exit(1)

    result = {}
    with open(chosen, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            k, _, v = line.partition('=')
            try:
                result[k.strip()] = float(v.strip())
            except ValueError:
                result[k.strip()] = v.strip()
    return result, chosen


def already_tested(family: str) -> bool:
    """Return True if gate2_results.csv already has a row for this family."""
    if not os.path.exists(GATE2_CSV) or os.path.getsize(GATE2_CSV) < 10:
        return False
    df = pd.read_csv(GATE2_CSV)
    return family in df.get('family', pd.Series(dtype=str)).values


def run_pipeline(cli_flags: list):
    """Invoke run_ml_pipeline.py with the given CLI flags.

    Returns the process return code.
    """
    cmd = [sys.executable, 'run_ml_pipeline.py'] + cli_flags
    flags_str = ' '.join(cli_flags) if cli_flags else '(no flags — pure baseline)'
    print(f"\n[Gate 2] Launching ML pipeline with {flags_str}...")
    print(f"         CWD: {ML_LAB}")
    print(f"         This takes ~45–60 minutes. Logs stream below.\n")
    proc = subprocess.run(cmd, cwd=ML_LAB)
    if proc.returncode != 0:
        print(f"\nERROR: run_ml_pipeline.py exited with code {proc.returncode}.")
        print("       Review the output above for errors, then re-run gate2_run.py.")
        sys.exit(1)
    print("[Gate 2] Pipeline complete.\n")


def read_aucs_from_state():
    """Return (aucs_list, below_050_count) from ml_state.json."""
    if not os.path.exists(ML_STATE_PATH):
        print(f"ERROR: {ML_STATE_PATH} not found after pipeline run.")
        sys.exit(1)
    with open(ML_STATE_PATH, encoding='utf-8') as f:
        state = json.load(f)
    signals = state.get('model_signals', {})
    aucs = [v['auc'] for v in signals.values() if isinstance(v, dict) and 'auc' in v]
    if not aucs:
        print("ERROR: No AUC values found in ml_state.json model_signals.")
        sys.exit(1)
    below = sum(1 for a in aucs if a < 0.50)
    return aucs, below


def read_ic_obs_count(horizon: int = 21) -> int:
    """Read just the n_obs for ml_alpha IC from alpha_ic_results.csv.

    We only use this to decide whether IC gate is deferred — we never
    read "before/after" IC from this file because the IC is computed
    from live DB signals that don't change on pipeline retraining runs.
    """
    if not os.path.exists(IC_RESULTS_PATH):
        return 0
    df = pd.read_csv(IC_RESULTS_PATH)
    row = df[(df['model'] == 'ml_alpha') & (df['horizon'] == horizon)]
    if row.empty:
        return 0
    n_obs = row.iloc[0].get('n_obs')
    return int(n_obs) if not pd.isna(n_obs) else 0


def run_pipeline_multi_seed(cli_flags: list, seeds: list):
    """Run the pipeline once per seed, collect per-run AUC stats.

    Returns (mean_auc, std_auc, all_aucs_per_seed, n_tickers, below_050).
    """
    all_mean_aucs = []
    all_std_aucs  = []
    all_n_tickers = []
    all_below     = []

    for i, seed in enumerate(seeds):
        print(f"\n{'-' * 60}")
        print(f" Seed run {i+1}/{len(seeds)} — seed={seed}")
        print(f"{'-' * 60}")
        seed_flags = cli_flags + ['--seed', str(seed)]
        run_pipeline(seed_flags)
        aucs, below = read_aucs_from_state()
        all_mean_aucs.append(float(np.mean(aucs)))
        all_std_aucs.append(float(np.std(aucs)))
        all_n_tickers.append(len(aucs))
        all_below.append(below)

    mean_of_means = float(np.mean(all_mean_aucs))
    std_of_means  = float(np.std(all_mean_aucs, ddof=0))  # population std of seed means
    n_tickers     = int(np.median(all_n_tickers))
    below_050     = int(np.max(all_below))  # worst case
    avg_within_std = float(np.mean(all_std_aucs))

    print(f"\n{'=' * 60}")
    print(f" Multi-seed summary ({len(seeds)} seeds: {seeds})")
    print(f"   Per-seed AUCs: {[f'{a:.4f}' for a in all_mean_aucs]}")
    print(f"   Mean of means: {mean_of_means:.4f}")
    print(f"   Seed-to-seed std: {std_of_means:.4f}")
    print(f"   Avg within-seed std: {avg_within_std:.4f}")
    print(f"   Noise floor (seed std): ±{std_of_means:.4f}")
    print(f"{'=' * 60}")

    return mean_of_means, avg_within_std, all_mean_aucs, n_tickers, below_050, std_of_means


def append_result(row: dict):
    """Append one row to gate2_results.csv."""
    df_new = pd.DataFrame([row])
    write_header = (not os.path.exists(GATE2_CSV)) or os.path.getsize(GATE2_CSV) < 10
    df_new.to_csv(GATE2_CSV, mode='a', header=write_header, index=False)
    print(f"Result appended -> {GATE2_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
def run_holdout_baseline():
    """Run pipeline with holdout filter but NO feature flags.

    Records the holdout-adjusted baseline AUC for fair Gate 2 comparisons.
    """
    print("\n" + "=" * 60)
    print(" HOLDOUT-ADJUSTED BASELINE")
    print(" Running pipeline with holdout filter, no feature flags.")
    print("=" * 60)

    if os.path.exists(BASELINE_HOLDOUT_FILE):
        print(f"\nWARNING: {BASELINE_HOLDOUT_FILE} already exists.")
        print("Rename it first if you want to re-record.")
        sys.exit(1)

    # Read original baseline for reference
    if not os.path.exists(BASELINE_FILE):
        print(f"ERROR: {BASELINE_FILE} not found. Run gate0_baseline.py first.")
        sys.exit(1)

    # Run with 3 seeds for variance estimate
    seeds = DEFAULT_SEEDS
    mean_auc, avg_std, per_seed_aucs, n_tickers, _, seed_std = \
        run_pipeline_multi_seed([], seeds)

    # Also read IC (just n_obs — value is from live DB, same as before)
    n_obs_ic = read_ic_obs_count(horizon=21)

    # Read the original full-history baseline for comparison
    orig_baseline, _ = parse_baseline(prefer_holdout=False)
    orig_auc = float(orig_baseline.get('mean_auc_best_of_3', 0))
    holdout_penalty = mean_auc - orig_auc

    lines = [
        f"# Holdout-adjusted baseline — recorded {date.today().isoformat()}",
        f"# Source: {len(seeds)}-seed average (seeds={seeds})",
        f"# Pipeline run with holdout filter, no Phase 1 feature flags.",
        f"# Compare: full-history baseline AUC = {orig_auc:.4f}",
        f"# Holdout penalty (holdout - full) = {holdout_penalty:+.4f}",
        f"n_tickers={n_tickers}",
        f"mean_auc_holdout={mean_auc:.4f}",
        f"std_auc_holdout={avg_std:.4f}",
        f"seed_std={seed_std:.4f}",
        f"per_seed_aucs={','.join(f'{a:.4f}' for a in per_seed_aucs)}",
        f"seeds_used={','.join(str(s) for s in seeds)}",
        f"n_obs_ic={n_obs_ic}",
        f"holdout_penalty={holdout_penalty:+.4f}",
    ]
    out = "\n".join(lines) + "\n"

    with open(BASELINE_HOLDOUT_FILE, 'w', encoding='utf-8') as f:
        f.write(out)

    print(f"\n{'=' * 60}")
    print(f" HOLDOUT BASELINE RECORDED")
    print(f"{'=' * 60}")
    print(out)
    print(f"Written to: {BASELINE_HOLDOUT_FILE}")
    print(f"\nHoldout penalty: {holdout_penalty:+.4f}")
    print(f"Seed-to-seed noise floor: ±{seed_std:.4f}")
    print(f"\nFuture Gate 2 runs will automatically use this as the comparison target.")
    print(f"Any delta_auc within ±{seed_std:.4f} is statistically indistinguishable from noise.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Gate 2: test one Phase 1A feature family (AUC + IC vs baseline).'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--family',
        choices=['db_regime', 'pead', 'earnings', 'crosssectional', 'acceleration',
                 'target_refinement', 'stationary_only', 'regularized_models', 'short_volume', 'combo_d'],
        help="Feature family to test"
    )
    group.add_argument(
        '--holdout-baseline', action='store_true', dest='holdout_baseline',
        help='Record a holdout-adjusted baseline (no feature flags, holdout filter on, 3-seed average).',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-run even if this family already has a row in gate2_results.csv.',
    )
    parser.add_argument(
        '--n-seeds', type=int, default=1, dest='n_seeds',
        help='Number of random seeds to average over (default: 1 = single run with seed 42). '
             'Use 3 for proper variance estimation.',
    )
    args = parser.parse_args()

    # ── Mode: holdout baseline ────────────────────────────────────────────────
    if args.holdout_baseline:
        run_holdout_baseline()
        return

    family    = args.family
    flag_name = FAMILY_FLAG_MAP[family]
    n_seeds   = max(1, args.n_seeds)
    seeds     = DEFAULT_SEEDS[:n_seeds] if n_seeds <= len(DEFAULT_SEEDS) else \
                DEFAULT_SEEDS + list(range(1000, 1000 + n_seeds - len(DEFAULT_SEEDS)))

    # ── Guard: already tested? ────────────────────────────────────────────────
    if already_tested(family) and not args.force:
        print(f"WARNING: '{family}' already has a row in gate2_results.csv.")
        print("         Use --force to overwrite. Exiting without running.")
        sys.exit(0)

    print("\n" + "=" * 60)
    print(f" GATE 2 — Testing family: {family}")
    print(f" Flag:    {flag_name}")
    print(f" Seeds:   {seeds} ({n_seeds} run{'s' if n_seeds > 1 else ''})")
    print("=" * 60)

    # ── 1. Baseline ───────────────────────────────────────────────────────────
    baseline, baseline_src = parse_baseline(prefer_holdout=True)

    # Support both full-history and holdout baseline key names
    auc_before = float(baseline.get('mean_auc_holdout',
                       baseline.get('mean_auc_best_of_3', 0)))
    std_before = float(baseline.get('std_auc_holdout',
                       baseline.get('std_auc_best_of_3', 0)))
    n_tix_base = int(float(baseline.get('n_tickers', 126)))

    # Noise floor from holdout baseline (if available)
    seed_std_baseline = float(baseline.get('seed_std', 0))
    holdout_penalty   = float(baseline.get('holdout_penalty', 0))
    using_holdout_baseline = ('mean_auc_holdout' in baseline)

    print(f"\nBaseline: AUC={auc_before:.4f} ± {std_before:.4f}  "
          f"n_tickers={n_tix_base}")
    print(f"  Source: {os.path.basename(baseline_src)}")
    if using_holdout_baseline:
        print(f"  (Holdout-adjusted baseline — apples-to-apples comparison)")
        print(f"  Seed-to-seed noise floor: ±{seed_std_baseline:.4f}")
    else:
        print(f"  (Full-history baseline — comparison slightly conservative)")
        print(f"  NOTE: Run --holdout-baseline first for a fair comparison.\n")

    # ── 2. Run pipeline with family enabled ───────────────────────────────────
    cli_flags = CLI_FLAG_MAP[family]
    if isinstance(cli_flags, str):
        cli_flags = [cli_flags]

    if n_seeds == 1:
        run_pipeline(cli_flags)
        aucs_after, below_050 = read_aucs_from_state()
        auc_after  = float(np.mean(aucs_after))
        std_after  = float(np.std(aucs_after))
        n_tickers  = len(aucs_after)
        seed_std_after = 0.0
    else:
        auc_after, std_after, per_seed_aucs, n_tickers, below_050, seed_std_after = \
            run_pipeline_multi_seed(cli_flags, seeds)

    delta_auc = auc_after - auc_before
    delta_std = std_after - std_before

    print(f"\nNew AUC:  {auc_after:.4f} ± {std_after:.4f}  "
          f"n_tickers={n_tickers}  n_below_0.50={below_050}")
    print(f"Delta:    AUC {delta_auc:+.4f}  std {delta_std:+.4f}")
    if n_seeds > 1:
        print(f"Seed std: ±{seed_std_after:.4f} (run-to-run noise)")

    # ── 3. IC gate — honest reporting ─────────────────────────────────────────
    # IC comes from live DB signals (alpha_eval.py reads the signals/price_targets
    # tables). Retraining the ML pipeline does NOT regenerate signals in the DB,
    # so the IC file is unchanged. We only check n_obs to decide deferral.
    n_obs_ic = read_ic_obs_count(horizon=21)
    ic_sufficient = (n_obs_ic >= MIN_OBS_FOR_IC_GATE)

    print(f"\nIC (ml_alpha 21d): n_obs={n_obs_ic}")
    if not ic_sufficient:
        print(f"  IC gate DEFERRED (n_obs={n_obs_ic} < {MIN_OBS_FOR_IC_GATE}).")
        print(f"  IC cannot be evaluated from pipeline retraining — it requires")
        print(f"  live signal accumulation via Saturday pipeline runs.")
        print(f"  Re-check after {MIN_OBS_FOR_IC_GATE}+ Saturday runs.")
    else:
        # If we ever reach sufficient IC observations, read the actual values
        df = pd.read_csv(IC_RESULTS_PATH)
        row = df[(df['model'] == 'ml_alpha') & (df['horizon'] == 21)]
        if not row.empty:
            r = row.iloc[0]
            ic_val  = float(r['mean_ic']) if not pd.isna(r.get('mean_ic')) else None
            icir_val = float(r['icir']) if not pd.isna(r.get('icir')) else None
            print(f"  IC={ic_val}  ICIR={icir_val}")
        else:
            ic_val = icir_val = None

    # ── 4. Evaluate criteria ──────────────────────────────────────────────────
    print("\n--- Gate 2 Criteria ---")

    # C1: AUC improvement (primary — always active)
    c1 = delta_auc > THRESHOLD_DELTA_AUC
    print(f"  {'[PASS]' if c1 else '[FAIL]'} delta_auc > +{THRESHOLD_DELTA_AUC:.3f}:  "
          f"{delta_auc:+.4f}  ({'PASS' if c1 else 'FAIL'})")

    # Statistical significance annotation
    combined_noise = max(seed_std_baseline, seed_std_after)
    if combined_noise > 0:
        ratio = abs(delta_auc) / combined_noise if combined_noise > 0 else float('inf')
        if ratio < 1.0:
            print(f"    [!] delta is within noise floor (|delta|/seed_std = {ratio:.1f}x) "
                  f"— NOT statistically distinguishable from baseline")
        elif ratio < 2.0:
            print(f"    [!] delta is marginal (|delta|/seed_std = {ratio:.1f}x) "
                  f"— weak evidence")
        else:
            print(f"    [+] delta exceeds noise floor (|delta|/seed_std = {ratio:.1f}x)")

    # C2: AUC variance stability
    c2 = delta_std <= THRESHOLD_DELTA_STD
    print(f"  {'[PASS]' if c2 else '[FAIL]'} delta_std <= +{THRESHOLD_DELTA_STD:.3f}:   "
          f"{delta_std:+.4f}  ({'PASS' if c2 else 'FAIL'})")

    # C3: No ticker regressions below 0.50
    c3 = (below_050 == 0)
    print(f"  {'[PASS]' if c3 else '[FAIL]'} no ticker AUC < 0.50:  "
          f"n_below={below_050}  ({'PASS' if c3 else 'FAIL - investigate these tickers'})")

    # C4+C5: IC criteria (deferred if n_obs too low)
    if ic_sufficient:
        ic_before_val  = float(baseline.get('mean_ic_ml_alpha_21d', 0))
        icir_before_val = float(baseline.get('icir_ml_alpha_21d', 0))
        delta_ic   = (ic_val   - ic_before_val)   if ic_val   is not None else None
        delta_icir = (icir_val - icir_before_val) if icir_val is not None else None
        c4 = delta_ic > THRESHOLD_DELTA_IC if delta_ic is not None else False
        c5 = delta_icir >= THRESHOLD_DELTA_ICIR if delta_icir is not None else False
        print(f"  {'[PASS]' if c4 else '[FAIL]'} delta_ic > +{THRESHOLD_DELTA_IC:.3f}:   "
              f"{delta_ic:+.4f}  ({'PASS' if c4 else 'FAIL'})")
        print(f"  {'[PASS]' if c5 else '[FAIL]'} delta_icir >= {THRESHOLD_DELTA_ICIR:.2f}:  "
              f"{delta_icir:+.4f}  ({'PASS' if c5 else 'FAIL'})")
        hard = [c1, c2, c3, c4, c5]
    else:
        c4 = c5 = None
        delta_ic = delta_icir = None
        print(f"  [-] IC criteria (C4+C5): DEFERRED (n_obs={n_obs_ic} < {MIN_OBS_FOR_IC_GATE})")
        hard = [c1, c2, c3]

    passed = all(hard)

    # ── 5. Verdict ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  GATE 2 VERDICT: {'[PASS]' if passed else '[FAIL]'}")
    print(f"{'=' * 60}")

    if passed:
        print(f"\nNext steps:")
        print(f"  1. Set {flag_name} = True in run_ml_pipeline.py (line ~69).")
        print(f"  2. Run the next two Saturday live pipeline runs with this flag enabled.")
        print(f"  3. After each Saturday run, check alpha_eval.py IC for ml_alpha.")
        print(f"     -> Gate 3 requires live IC within +/-0.008 of walk-forward IC.")
        print(f"  4. Once Gate 3 passes, proceed to the next feature family.")
        if c4 is None:
            print(f"\n  (!) IC gate was deferred. Revisit delta_ic after accumulating")
            print(f"     >={MIN_OBS_FOR_IC_GATE} ml_alpha signal observations in the live DB.")
            print(f"     If IC then fails (+0.003 threshold), revert the flag.")
    else:
        print(f"\nNext steps:")
        print(f"  * Keep {flag_name} = False (it was not changed by this script).")
        print(f"  * Investigate the failing criterion (see above).")
        print(f"  * Do NOT proceed to the next family until this is understood.")
        if combined_noise > 0 and abs(delta_auc) < 2 * combined_noise:
            print(f"\n  NOTE: The AUC delta ({delta_auc:+.4f}) is within 2x the seed-to-seed")
            print(f"  noise floor (±{combined_noise:.4f}). This FAIL may be a false negative —")
            print(f"  the test lacks statistical power at this sample size to detect the")
            print(f"  feature's effect. Consider re-running with --n-seeds 5 or more data.")

    # ── 6. Record ─────────────────────────────────────────────────────────────
    notes_parts = [
        f"n_tickers={n_tickers}",
        f"n_below_0.50={below_050}",
        f"IC_gate={'active' if ic_sufficient else f'deferred(n_obs={n_obs_ic}<{MIN_OBS_FOR_IC_GATE})'}",
        f"holdout_filtered=True",
        f"baseline_from={os.path.basename(baseline_src)}",
        f"n_seeds={n_seeds}",
        f"seeds={seeds}",
    ]
    if combined_noise > 0:
        notes_parts.append(f"noise_floor=±{combined_noise:.4f}")
    if n_seeds > 1:
        notes_parts.append(f"seed_std_after=±{seed_std_after:.4f}")

    row = {
        'date_tested':     date.today().isoformat(),
        'family':          family,
        'flag_name':       flag_name,
        'n_seeds':         n_seeds,
        'mean_ic_before':  'DEFERRED' if not ic_sufficient else f"{float(baseline.get('mean_ic_ml_alpha_21d', 0)):.4f}",
        'mean_ic_after':   'DEFERRED' if not ic_sufficient else (f"{ic_val:.4f}" if ic_val is not None else 'N/A'),
        'delta_ic':        'DEFERRED' if not ic_sufficient else (f"{delta_ic:+.4f}" if delta_ic is not None else 'N/A'),
        'icir_before':     'DEFERRED' if not ic_sufficient else f"{float(baseline.get('icir_ml_alpha_21d', 0)):.4f}",
        'icir_after':      'DEFERRED' if not ic_sufficient else (f"{icir_val:.4f}" if icir_val is not None else 'N/A'),
        'mean_auc_before': f"{auc_before:.4f}",
        'mean_auc_after':  f"{auc_after:.4f}",
        'delta_auc':       f"{delta_auc:+.4f}",
        'seed_std':        f"{seed_std_after:.4f}" if n_seeds > 1 else 'N/A',
        'noise_floor':     f"±{combined_noise:.4f}" if combined_noise > 0 else 'N/A',
        'pass':            'PASS' if passed else 'FAIL',
        'notes':           '; '.join(notes_parts),
    }
    print()
    append_result(row)


if __name__ == '__main__':
    main()
