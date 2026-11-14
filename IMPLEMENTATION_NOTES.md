# Implementation notes

These notes summarize the conventions used in the code accompanying DOI: 10.3389/frai.2026.1760912.

## Model conventions

- **Clinical profile:** Project documentation labels 0 as favorable (Neg) and 1 as unfavorable (Pos). The paper gives the reverse coding; 
- **Dose assignment:** Each generated patient trajectory retains its baseline dose.
- **Frailty transitions:** Full-dose modifiers lower deterioration probability and increase recovery probability.
- **QoL storage:** Values are retained at full floating-point precision.


## Generator timing and parameters

Baseline (`visit=0`) stores a pretreatment QoL draw and zero QALYs. Cycle k starts at (k−1) × 0.25 years. Its record combines start-of-cycle age, frailty, profile, dose and exposure with that cycle's toxicity and QALYs; `cum_qaly` includes the current cycle.

Each cycle:
1. Retain 80% of prior toxicity damage.
2. Draw QoL from the current state and remaining damage.
3. Draw toxicity death; if no death occurs, draw severe toxicity and apply its immediate and carryover effects.
4. Credit QoL × 0.25 and store the record; retain measured `QOL` every fourth cycle.
5. Stop on death. Otherwise update exposure, clinical profile, its streak and frailty, then advance age by 0.25 years.

Death records receive a full cycle's reward (an end-of-cycle convention); no half-cycle correction is applied. Baseline and cycle 1 share age and state variables but have different QoL draws. The 15,298 adjacent record pairs include this initialization boundary.

| Mechanism | Rule |
|---|---|
| Cohort defaults | 500 patients, 40 cycles, seed 42, 0.25 years/cycle |
| Baseline age | Uniform integer 71–95 inclusive |
| Baseline frailty | P(frail)=0.7 |
| Baseline clinical profile | Uniform integer 0 or 1 |
| Baseline dose | Fit: full; frail: P(full)=0.5 |
| Base QoL draws | Fit U(0.6,0.9); frail U(0.4,0.7), newly drawn each cycle |
| Age penalty | 0.015 * max(age-75,0) |
| QoL noise | Normal mean 0, SD 0.03 |
| Dose QoL penalty | Full: 0.040 fit / 0.060 frail; reduced: 0.006 |
| QoL efficacy term | Only when clinical_test=1: +0.015 full / +0.008 reduced |
| Severe-toxicity hit | 0.15 fit / 0.22 frail |
| Carryover damage | Add 80% of the toxicity hit; retain 80% of existing damage next cycle |
| Exposure | +1 per surviving full-dose cycle; reduced branch decays exposure by 0.10 to a minimum of zero |
| Exposure contribution to P(test=1) | min(0.05*exposure,0.25) for nonnegative exposure |
| P(next test=1), current test=1 | 0.40*(0.92 full / 1.03 reduced) + exposure contribution |
| P(next test=1), current test=0 | 0.20*(0.95 full / 1.05 reduced) + exposure contribution |
| Clinical probability bounds | Clip to [0.05,0.99] |
| Fit-to-frail base | 0.014 age<=75; 0.022 age<=80; 0.036 otherwise |
| Fit-to-frail modifiers | Full: -0.003; if updated test streak>=2: +0.003 full / +0.008 reduced |
| Frail-to-fit base | 0.010 age<=75; 0.008 age<=80; 0.006 otherwise |
| Frail-to-fit modifiers | +0.004 full dose; +0.003 if updated clinical_test=0 |
| Severe-toxicity probability | Base 0.12 age<65 / 0.17 otherwise; multiply by 1 full / 0.55 reduced; sampled conditional on no toxicity death |
| Toxicity-death probability | Base 0.0015 age<65 / 0.0145 otherwise; multiply by 1 full / 0.50 reduced |


Final QoL is clipped to [0,1]; the uniform intervals above describe only its base draw. All default patients use the age≥65 toxicity branch. Background and cancer-related mortality are absent. Reduced dose is neither treatment interruption nor zero toxicity; with fixed reduced dosing and zero initial exposure, its exposure-decay branch remains at zero.

## MDP estimation and simulation

- **States and actions:** eight combinations of frailty, clinical indicator and current dose; actions are reduced or full dose. There is no separate hold/stop action.
- **Rewards:** future measured `QOL` (next row or lookahead up to three visits) × 0.25. Missing action-specific observations use the state's average measured QoL; states without measurements use a default. Rewards do not directly average `qaly_quarter`.
- **Transitions:** observed vectors use Dirichlet smoothing with alpha=0.5 across all eight states. Unobserved actions use 0.7 at the current state and 0.3/7 at every other state.
- **Dose selection:** At each decision step, the MDP evaluates the patient's current state and selects either full or reduced dose.
- **Terminal outcomes:** death terminates generated patient records.
- **State information:** age, exposure, clinical streak and toxicity carryover affect the generator but are omitted from the eight-state representation; the aggregated process is not guaranteed to be exactly Markov.
- **Measurement availability:** quarterly latent QoL is recoverable from `qaly_quarter/0.25`, although measured `QOL` is stored annually. Analyses assuming annual observations must distinguish those measurements from latent quarterly outcomes.

## Figure and reproducibility

Each group equally weights its selected starting states. Pooled variance is `mean(within_state_variance + state_mean**2) - group_mean**2`. The band is ±1 pooled SD, not a confidence interval or weighting by observed cohort frequencies.

The figure shows 21 quarterly rewards at cycle-start times 0–60 months, covering 5.25 years. The generator covers 40 quarterly cycles (10 years). See the README for group definitions and output files.

Reference checks establish numerical reproducibility. Interpretation of policy differences remains conditional on the conventions and limitations above. Model extensions should be evaluated separately against the preserved reference workflow.
