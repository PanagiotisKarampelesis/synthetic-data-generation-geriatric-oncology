# Starting points for a thesis

## Reproduce first

Run the reference pipeline, inspect `optimal_policy.csv`, `rewards.csv`, `transitions.csv` and the five plotted curves. Trace each result back to the corresponding notebook execution count. Read the implementation notes before interpreting treatment-policy differences.

## Investigate model assumptions

1. **Action coverage.** Quantify state–action coverage. Current fixed-dose data support only one observed action per state; the other action uses an assumed transition vector and a state-average reward. Compare behavior-policy designs with observed dose switches and explicit action-conditioned simulators.
2. **Terminal outcomes.** Add an absorbing death state and justify reward timing. Compare policy rankings and cumulative rewards against the existing continuing eight-state process.
3. **State sufficiency.** Study age, cumulative exposure, clinical streak and toxicity carryover as additional state variables or history features. Compare state aggregation and sample requirements.
4. **Reward estimation.** Examine annual QoL measurement, lookahead, censoring and the reuse of a later measurement for multiple earlier steps. Compare alternative estimators without presenting them as the original analysis.
5. **Uncertainty and robustness.** Vary simulation seeds, transition priors, reward fallbacks, discount rates and horizons. Distinguish individual-trajectory dispersion from uncertainty in estimated policy performance.
6. **Population weighting.** Compare equally weighted starting states with baseline-frequency weighting and patient-level evaluation on held-out trajectories.

These are research directions, not validated fixes. State changes, new death handling and treatment switches can change results. Keep a versioned reference and report the differences rather than tuning extensions to preserve the published curves.
