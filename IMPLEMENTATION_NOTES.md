# Implementation notes and publication discrepancies

These notes describe the supplied implementation as executed, not a corrected model. The numerical generator is preserved. The paper reference is DOI 10.3389/frai.2026.1760912, especially sections 3.1.1–3.2 (printed pages 8–9). Supplementary Tables S1/S2 were not included in the review materials and have not been checked.

## 1. Explicit differences from published prose

| Topic | Published description | Supplied implementation |
|---|---|---|
| Clinical-profile coding (§3.1.2) | 0 unfavorable, 1 favorable | Supplied README labels 0 favorable, 1 unfavorable. Code contains no authoritative semantic mapping that resolves all model terms. |
| Dose adaptation (§3.1.1/3.1.4) | Dose may be reduced/held with probability 0.20 after an unfavorable clinical profile | Dose is assigned once and never changed. |
| Full-dose frailty effect (§3.1.3) | Full dose increases fit-to-frail risk; reduced dose favors recovery | Full dose subtracts 0.003 from deterioration probability and adds 0.004 to recovery probability. |
| QoL storage (§3.1.5) | Rounded to two decimals | Full floating-point precision is retained. |

Do not silently reverse clinical-profile values or change transition signs to align the code with the prose. First establish the convention used by the original downstream MDP. For example, `clinical_test == 0` increases recovery probability, while `clinical_test == 1` activates the positive QoL efficacy term. Full-dose exposure increases the probability of `clinical_test == 1` through the cumulative-exposure term. These mechanisms require a consistent scientific interpretation; a label edit alone does not establish one.

Documentation of these differences does not amend the published article. Coauthors should determine whether a publication clarification is appropriate after examining the original analysis pipeline.

## 2. Cycle timing and event order

The following is an explicit description of the existing implementation's timing, not evidence that this convention was specified in the paper.

Baseline (`visit=0`) stores a pretreatment QoL draw and zero QALYs. For cycle k>=1, recorded age is baseline_age + 0.25*(k-1). The cycle spans [(k-1)*0.25, k*0.25] years. Frailty, clinical profile, dose and exposure are cycle-start quantities; toxicity and QALYs describe the cycle. `cum_qaly` includes the current cycle. Thus rows combine start-of-cycle state and cycle outcomes.

Within each cycle:

1. Multiply prior toxicity damage by 0.8.
2. Draw QoL using current frailty, age, dose, clinical profile and remaining damage.
3. Draw toxicity death. Only if no death occurs, draw nonfatal severe toxicity; subtract its immediate QoL hit and accumulate future damage.
4. Credit `QoL * 0.25` and store the record (retaining `QOL` only every fourth cycle).
5. Stop if death occurred. Otherwise update exposure, draw the next clinical profile, update its streak, draw the next frailty state using the updated profile and current age, then increase age by 0.25.

The full-cycle reward on the death record is equivalent to an end-of-cycle death convention. A half-cycle correction would change rewards and is not applied. Baseline and cycle 1 share state variables and age, but have different QoL draws. A downstream transition estimator must account for that initialization boundary explicitly. The reported 15,298 adjacent record pairs include those baseline-to-cycle-1 pairs.

## 3. Parameter inventory (as implemented)

These values document software behavior; this review does not validate their clinical calibration or literature provenance.

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

All default patients are older than 65, so their toxicity probabilities use only the older-age branch. Background and cancer-related mortality are absent. Reduced dose is not equivalent to zero exposure or zero toxicity. Under fixed initial dosing, the exposure-recovery branch always starts from zero and never produces a declining positive exposure trajectory.

## 4. Data availability and downstream modeling limits

- `QOL` missingness is scheduled, but the latent quarterly QoL can be recovered as `qaly_quarter/0.25` for every follow-up cycle. Models claiming access only to annual QoL must not use quarterly rewards as contemporaneously available QoL measurements.
- There are no within-patient dose switches. With states including current dose, observed trajectories do not directly support both actions at each state. Smoothing does not establish the empirical effect of an unobserved switching action.
- Age, cumulative exposure, clinical streak and carryover damage affect transitions or rewards but are absent from the paper's eight-state representation. The aggregated state process is not guaranteed to be exactly Markov.
- Death appears as a flag on the last record, not as an explicit absorbing-state row. The supplied MDP has now been checked; see section 5 for terminal handling.
- The two QoL uniform intervals describe the base draw, not bounds on final QoL after age, treatment, noise and damage terms. Final values are clipped to [0,1].

These are model limitations, distinct from numerical reproduction. Section 5 records the subsequent MDP inspection; this complete package also checks its numerical outputs against the original notebook functions.

## 5. Findings after inspecting the original MDP notebook

The MDP notebook was subsequently supplied and inspected. The following findings extend the initial generator-only review. Supplementary tables remain unavailable.

- The notebook uses clinical-profile labels Neg/Pos. Generator comments explicitly associate test 1 with disease activity/bad test and test 0 with good tests. The reference workflow retains this convention.
- Rewards use a future measured `QOL` (next row or lookahead up to three visits), multiplied by 0.25. When no action-specific reward observations exist, the reward is the current state's average measured QoL ×0.25; if a state has no QoL observations, a hard-coded default is used.
- Observed transition vectors use alpha=0.5 Dirichlet smoothing across all eight states. Completely unobserved actions use a distinct fallback: 0.7 probability at the current state and 0.3/7 at each other state. The claimed switching-action benefits therefore depend on these assumptions.
- The dose component of the next state is not constrained to equal the action in fallback/smoothed transitions.
- There is no absorbing death state in the eight-state MDP or trajectory simulator. Death flags terminate data trajectories, but simulated MDP trajectories accrue rewards for their entire requested horizon.
- The original state setup and `clinical_test_based` policy must both be adapted for a clinical-code inversion. In a separate controlled test, inverting the CSV test bit while retaining semantic state-ID order and adapting that policy produced identical R, P, values, policies and all simulated trajectory summaries. This optional transformation is not applied in this reference package.
- In [43] pools variance across starting states with equal weights: E[within-state variance + mean squared] - pooled mean squared. With 500 simulations per state this is the SD of an equally weighted mixture, not a confidence interval or weighting by the actual baseline cohort.
- In [44] executes 21 steps, despite a comment mentioning 40. The plotting function displays x=0..4 years from a series indexed at 0..5 years. Its values are per-cycle QALYs, not cumulative QALYs.

The clean pipeline preserves these numerical assumptions to reproduce the original analysis. Its figure changes only layout, visible tick labels and unit wording. Numerical reproducibility does not establish clinical validity or resolve the published method discrepancies.
