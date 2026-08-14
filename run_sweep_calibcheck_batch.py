"""
Run Sweep Calibcheck Batch

Generates a THIRD, genuinely fresh sweep trace set (traces/run_sweep_calibcheck_N.jsonl)
-- never used in training (features_sweep.csv) or in calibrating run_hybrid's
PRESSURE_WINDOW_MULTIPLIER/PRESSURE_RATIO_THRESHOLD (features_sweep_holdout.csv).
Exists to answer one honest question: does the hybrid's calibration, tuned
entirely against the first held-out set's own 12-point curve, actually
generalize to sweep data it has never seen in any capacity -- or was that
calibration fit to that one dataset's particular characteristics?
"""
