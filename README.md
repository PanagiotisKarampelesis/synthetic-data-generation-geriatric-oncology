# Synthetic patient data and MDP modeling for geriatric oncology

Focused reproducibility workflow for the final generator and the paper figure produced by **In [43] (aggregation), In [57] / In [59] (final plot)**, associated with:

Karampelesis P, Denazis S, Koufopavlou O and Zacharaki EI (2026). *From mechanistic models to artificial intelligence: exploring the potential of digital twins in geriatric oncology.* Frontiers in Artificial Intelligence 9:1760912. https://doi.org/10.3389/frai.2026.1760912

This package includes synthetic data generation, an eight-state MDP, policy optimization, trajectory simulation and the baseline-cohort figure. It preserves the original numerical workflow, including its modeling assumptions. **Known discrepancies with the paper and limitations are documented in [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md).** This is a proof of concept, not a clinically validated treatment system.

## Quick start

Use Python 3.10 or later (verified here with Python 3.12.14). From the repository root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell alternative: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-reproducible.txt
python scripts/run_mdp.py
```

The last command reads the bundled `Fixed_patient_data.csv` and writes results to `generated/mdp/`. No notebook session or hidden kernel variables are required. The repository cited by the article is https://github.com/PanagiotisKarampelesis/synthetic-data-generation-geriatric-oncology.

## Outputs

| File in `generated/mdp/` | Contents |
|---|---|
| `figure_baseline_groups.png` | Frail/fit mean QALY-per-cycle curves and ±1 SD bands |
| `figure_curves.csv` | Underlying curve means and standard deviations |
| `optimal_policy.csv` | State definitions, values and optimal actions |
| `rewards.csv` | All 16 state–action rewards |
| `transitions.csv` | All 128 state–action–next-state probabilities |
| `mdp_arrays.npz` | Numerical R, P, value and policy arrays |
| `policy_summary.csv` | Final cumulative reward summary by policy |
| `run_metadata.json` | Settings, versions, timing and interpretation limits |

A verified example run is included in `results/reference/`. New runs write to `generated/`, which is excluded from Git.

## Reproduce the reference data

```bash
python scripts/generate_synthetic_data.py --n_patients 500 --n_visits 40 --seed 42
```

Output: `generated/Fixed_patient_data.csv`. The CLI protects the root reference CSV. To use a separately generated dataset:

```bash
python scripts/run_mdp.py --input generated/Fixed_patient_data.csv --output-dir generated/mdp_regenerated
```

Programmatic use:

```python
from src.data_generation import simulate_data
from src.pipeline import run_analysis

df = simulate_data(save_csv=False)
mdp, policy, trajectories = run_analysis('Fixed_patient_data.csv', 'generated/mdp')
```

## What is reproduced

- Generator: 500 patients, 40 quarterly cycles, seed 42, 15,798 rows and 15,298 adjacent within-patient record pairs; 2,108 nonfatal severe-toxicity events and 212 toxicity deaths.
- MDP: states `(frailty, clinical_test, current_dose)`; two actions; reward lookahead 3; transition smoothing alpha 0.5; annual discount 0.97; value iteration tolerance 0.001.
- Trajectories: 500 simulations per starting state/policy, seed 123; 21 reward cycles, as in both executed In [44] and In [59] calls.
- In original state-ID order, optimal actions: `[0, 1, 0, 1, 1, 0, 1, 0]`.
- All-start-state mean final cumulative QALYs, rounded as in the saved notebook: optimal **2.127**, always reduced **1.843**, always full **1.634**. These summaries use all eight starting states equally; the figure's Fit group uses only fit/full-dose starting states.

The figure displays optimal, reduced-dose and full-dose policies in both panels, including the previously hidden Fit reduced-dose curve. It shows all 21 points over 0–60 months, matching the supplied reference image. The MDP and trajectory calculations are unchanged. Verified numerical outputs and a newly rendered figure are in `results/reference/`. The exploratory notebook is excluded from the current version to avoid conflicting generators and repeated definitions.

The generator's 40-cycle horizon is distinct from the trajectory plot's 21-cycle run. The notebook's “10 years / 40 visits” comment above In [44] was inconsistent with the actual call. Twenty-one quarterly rewards cover 5.25 years; plotted cycle-start indices 0–20 are labeled 0–60 months.

## Clinical-profile coding

This reference release retains **0 = favorable (Neg), 1 = unfavorable (Pos)**, consistent with the original generator comments. Paper §3.1.2 describes the reverse coding. A coordinated relabeling can preserve the numerical MDP, but it is not applied here. Do not mix a recoded CSV with this legacy MDP without adapting its state mapping and test-based policy.

## Variables

| Column | Meaning |
|---|---|
| `patient_id` | Synthetic ID |
| `age` | Baseline/start-of-cycle age |
| `visit` | 0 baseline; 1–40 quarterly cycle index |
| `frailty` | 0 fit; 1 frail |
| `clinical_test` | Legacy code described above |
| `full_dose` | 0 reduced-dose branch; 1 full-dose branch; fixed within each generated trajectory |
| `cum_dose_cat` | Numeric cumulative exposure at cycle start, not a category |
| `QOL` | Utility proxy, retained at baseline and every fourth cycle |
| `qaly_quarter` | Simulated cycle QoL × 0.25, zero at baseline |
| `cum_qaly` | Cumulative simulated QALYs |
| `severe_tox` | Nonfatal severe-toxicity event |
| `death_tox` | Toxicity death; last record of a terminated trajectory |

The MDP estimates rewards from measured `QOL` with lookahead/fallback; it does **not** directly average `qaly_quarter`. Reduced dose is not a separately modeled stop/hold action.

## Files and notebook provenance

| Path | Role / source |
|---|---|
| `src/data_generation.py` | Supplied standalone generator, preserved byte-for-byte |
| `src/mdp.py` | `SimpleQOLMDP` extracted from In [12] |
| `src/trajectories.py` | Simulation and policy functions from In [16] |
| `src/plotting.py` | Group definitions from In [25], aggregation from In [43], final plot from In [57] |
| `src/pipeline.py` | Explicit orchestration equivalent to In [15]/In [59], plus exports |
| `notebooks/reproduce_mdp.ipynb` | Four code cells: generate data, fit/simulate, display the figure (plus setup) |
| `docs/FIGURE_PROVENANCE.md` | Generator comparison and exact original-cell dependency map |
| `tests/` | Generator and MDP numerical regression checks |
| `docs/verification*.json` | Results of validation performed on this package |

The bracketed numbers are execution counts, not positional cell numbers. Only the last generator, In [4], matches the reference CSV. The first three produce 17,030, 16,887 and 15,266 rows respectively; In [4] produces the matching 15,798. The clean notebook has no repeated generator definitions or alternative plots. Only optimal/full/reduced policies are simulated, and only the baseline-cohort plot is produced.

## Verify

```bash
python -m unittest discover -s tests -v
```

The tests compare every generated CSV column, source hashes, and MDP R/P/value/policy arrays against verified reference outputs; figure means/SDs match their reference CSV within 1e-12. The reference outputs were independently checked against the original notebook functions before removing the exploratory notebook. `requirements-reproducible.txt` pins the versions tested during this review; it does not claim the original publication environment.

## Starting a thesis extension

See [docs/THESIS_STARTING_POINTS.md](docs/THESIS_STARTING_POINTS.md). Preserve this reference workflow before changing the model. Evaluate extensions separately, with sensitivity analysis and an explicit comparison against the reference. No software license is currently included; contact the authors about reuse and redistribution permissions.
