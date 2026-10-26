# Synthetic patient data and MDP modeling for geriatric oncology

A Python project for generating synthetic longitudinal patient data, fitting a Markov decision process (MDP), and comparing dose policies in geriatric oncology. Associated publication:

Karampelesis P, Denazis S, Koufopavlou O and Zacharaki EI (2026). *From mechanistic models to artificial intelligence: exploring the potential of digital twins in geriatric oncology.* Frontiers in Artificial Intelligence 9:1760912. https://doi.org/10.3389/frai.2026.1760912

This package includes synthetic data generation, an eight-state MDP, policy optimization, trajectory simulation and the baseline-cohort figure. **Known discrepancies with the paper and limitations are documented in [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md).** This is a proof of concept, not a clinically validated treatment system.

## Citation

If you find this work useful in your research, please cite our paper:

Karampelesis P, Denazis S, Koufopavlou O, Zacharaki EI. [From mechanistic models to artificial intelligence: exploring the potential of digital twins in geriatric oncology](https://doi.org/10.3389/frai.2026.1760912). Frontiers in Artificial Intelligence. 2026 Jul 23;9:1760912.

DOI: [10.3389/frai.2026.1760912](https://doi.org/10.3389/frai.2026.1760912).

Use **Cite this repository** in the GitHub sidebar to copy the paper citation in APA or BibTeX format.

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

The last command reads the bundled `Fixed_patient_data.csv` and writes results to `generated/mdp/`. Data generation is a separate command; the default analysis uses the bundled dataset. The repository cited by the article is https://github.com/PanagiotisKarampelesis/synthetic-data-generation-geriatric-oncology.

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

## Default configuration and results

- Generator: 500 patients, 40 quarterly cycles, seed 42, 15,798 rows and 15,298 adjacent within-patient record pairs; 2,108 nonfatal severe-toxicity events and 212 toxicity deaths.
- MDP: states `(frailty, clinical_test, current_dose)`; two actions; reward lookahead 3; transition smoothing alpha 0.5; annual discount 0.97; value iteration tolerance 0.001.
- Trajectories: 500 simulations per starting state/policy, seed 123; 21 quarterly reward cycles.
- In state-ID order, optimal actions: `[0, 1, 0, 1, 1, 0, 1, 0]`.
- All-start-state mean final cumulative QALYs, rounded to three decimals: optimal **2.127**, always reduced **1.843**, always full **1.634**. These summaries use all eight starting states equally; the figure's Fit group uses only fit/full-dose starting states.

The figure compares optimal, reduced-dose and full-dose policies for Frail and Fit baseline groups. Each curve reports mean QALYs per three-month cycle; shading is ±1 pooled standard deviation. The default plot contains 21 points with cycle-start labels from 0 to 60 months.

The generator uses 40 cycles (10 years), while the policy simulations use 21 reward cycles (5.25 years). The final plotted point marks the start of the last reward cycle, at month 60.

## How the analysis works

1. Load longitudinal patient records from a CSV file.
2. Map `(frailty, clinical_test, current_dose)` to eight states, with reduced and full dose as the two actions.
3. Estimate quarterly rewards from measured `QOL` using lookahead and fallback rules, and estimate transition probabilities with smoothing.
4. Apply value iteration to obtain a policy maximizing expected discounted reward.
5. Simulate the optimal policy, always-reduced dose and always-full dose from each starting state.
6. Aggregate trajectories into Frail and Fit baseline groups and export the figure, curve data, model arrays and policy summaries.

See [docs/FIGURE_GUIDE.md](docs/FIGURE_GUIDE.md) for group definitions and interpretation of the plot.

## Clinical-profile coding

The implementation uses **0 = favorable (Neg), 1 = unfavorable (Pos)**. Paper §3.1.2 describes the reverse coding. A coordinated relabeling can preserve the numerical MDP, but it is not applied here. Do not mix a recoded CSV with this MDP without adapting all code that interprets the clinical indicator.

## Variables

| Column | Meaning |
|---|---|
| `patient_id` | Synthetic ID |
| `age` | Baseline/start-of-cycle age |
| `visit` | 0 baseline; 1–40 quarterly cycle index |
| `frailty` | 0 fit; 1 frail |
| `clinical_test` | Clinical status indicator; coding described above |
| `full_dose` | 0 reduced-dose branch; 1 full-dose branch; fixed within each generated trajectory |
| `cum_dose_cat` | Numeric cumulative exposure at cycle start, not a category |
| `QOL` | Utility proxy, retained at baseline and every fourth cycle |
| `qaly_quarter` | Simulated cycle QoL × 0.25, zero at baseline |
| `cum_qaly` | Cumulative simulated QALYs |
| `severe_tox` | Nonfatal severe-toxicity event |
| `death_tox` | Toxicity death; last record of a terminated trajectory |

The MDP estimates rewards from measured `QOL` with lookahead/fallback; it does **not** directly average `qaly_quarter`. Reduced dose is not a separately modeled stop/hold action.

## Project structure

| Path | Purpose |
|---|---|
| `src/data_generation.py` | Generate longitudinal synthetic patient records |
| `src/mdp.py` | Define states and actions, estimate rewards/transitions, solve the MDP |
| `src/trajectories.py` | Simulate and summarize the three dose policies |
| `src/plotting.py` | Define baseline groups, pool statistics and draw the figure |
| `src/pipeline.py` | Run the analysis and export results |
| `scripts/generate_synthetic_data.py` | Command-line entry point for data generation |
| `scripts/run_mdp.py` | Command-line entry point for modeling and plotting |
| `results/reference/` | Reference numerical results and figure |
| `tests/` | Data, model and figure regression checks |
| `IMPLEMENTATION_NOTES.md` | Model assumptions, limitations and publication discrepancies |

## Verify

```bash
python -m unittest discover -s tests -v
```

The tests compare every generated CSV column, source hashes, and MDP R/P/value/policy arrays against verified reference outputs; figure means/SDs match their reference CSV within 1e-12. `requirements-reproducible.txt` pins the versions used to validate this project.

## Starting a thesis extension

See [docs/THESIS_STARTING_POINTS.md](docs/THESIS_STARTING_POINTS.md). Preserve this reference workflow before changing the model. Evaluate extensions separately, with sensitivity analysis and an explicit comparison against the reference. No software license is currently included; contact the authors about reuse and redistribution permissions.
