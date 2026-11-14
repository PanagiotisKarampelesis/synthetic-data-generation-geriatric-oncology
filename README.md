# Synthetic patient data and MDP modeling for geriatric oncology

A Python research project for generating synthetic longitudinal patient data and comparing dosing policies using a Markov decision process (MDP).

The analysis compares three policies—state-dependent dosing, always full dose, and always reduced dose—and produces results and figures for baseline Fit and Frail groups.

## Publication

Karampelesis P, Denazis S, Koufopavlou O, Zacharaki EI (2026).
*From mechanistic models to artificial intelligence: exploring the potential of digital twins in geriatric oncology.*
Frontiers in Artificial Intelligence, 9:1760912.
https://doi.org/10.3389/frai.2026.1760912

Use **Cite this repository** on GitHub to obtain the citation.

This is a research proof of concept, not a clinically validated treatment tool. Model assumptions, implementation details, and known differences from the published description are documented in [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md).

## Quick start

Use Python 3.10 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproducible.txt
python scripts/run_mdp.py
```

On Windows, use `python` instead of `python3` and activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell.

The analysis reads the bundled `Fixed_patient_data.csv` and saves outputs in `generated/mdp/`. An example figure and numerical results are available in [results/reference/](results/reference/).

To generate a separate dataset and analyze it:

```bash
python scripts/generate_synthetic_data.py --n_patients 500 --n_visits 40 --seed 42
python scripts/run_mdp.py --input generated/Fixed_patient_data.csv --output-dir generated/mdp_regenerated
```

The generated CSV is saved in `generated/`; the bundled reference dataset is preserved. An optional interactive example is provided in [notebooks/reproduce_mdp.ipynb](notebooks/reproduce_mdp.ipynb).

## Analysis and outputs

The workflow maps frailty, clinical profile and current dose to eight states, estimates quarterly rewards and transition probabilities, and uses value iteration to select a dosing policy. It then simulates the three policies and exports:

| Output | Contents |
|---|---|
| `figure_baseline_groups.png` | Fit/Frail policy comparison |
| `figure_curves.csv` | Curve means and standard deviations |
| `optimal_policy.csv` | State values and selected actions |
| `rewards.csv`, `transitions.csv` | Estimated rewards and transition probabilities |
| `policy_summary.csv` | Final cumulative reward summaries |
| `mdp_arrays.npz`, `run_metadata.json` | Model arrays and execution settings |

The generator defaults to 500 patients and 40 quarterly cycles (10 years). Policy simulations use 500 trajectories per starting state and policy, over 21 reward cycles (5.25 years).

### Reading the figure

- Groups refer to **baseline** frailty. Frail includes both starting doses; Fit includes full-dose starting states. Both groups include both clinical profiles.
- Curves show mean QALYs **per three-month cycle**, not cumulative QALYs.
- Shading shows ±1 pooled standard deviation, not a confidence interval. Starting states have equal weight within each group.
- The 21 plotted points mark cycle starts from month 0 to month 60; the last reward cycle ends at month 63.
- `policy_summary.csv` reports cumulative rewards across all eight starting states with equal weights, rather than the figure's group-specific aggregation.

## Variables

| Column | Meaning |
|---|---|
| `patient_id` | Synthetic patient ID |
| `age` | Baseline/start-of-cycle age |
| `visit` | 0 = baseline; 1–40 = quarterly cycle index |
| `frailty` | 0 = fit; 1 = frail |
| `clinical_test` | Binary clinical indicator: 0 = Neg; 1 = Pos; interpretation is discussed in the implementation notes |
| `full_dose` | 0 = reduced dose; 1 = full dose |
| `cum_dose_cat` | Numeric cumulative exposure at cycle start |
| `QOL` | Utility proxy recorded at baseline and every fourth cycle |
| `qaly_quarter` | Simulated cycle QoL × 0.25; zero at baseline |
| `cum_qaly` | Cumulative simulated QALYs |
| `severe_tox` | Nonfatal severe-toxicity event |
| `death_tox` | Toxicity death; final record of a terminated data trajectory |

## Project structure and checks

- `src/`: data generation, MDP, policy simulations and plotting.
- `scripts/`: command-line entry points.
- `tests/`: reproducibility checks against the reference data and results.
- `results/reference/`: stored reference outputs.
- `IMPLEMENTATION_NOTES.md`: technical conventions and modeling limits.
- `CITATION.cff`: citation metadata.

Run the checks with:

```bash
python -m unittest discover -s tests -v
```

Pinned dependencies are provided in `requirements-reproducible.txt`. The checks compare generated data and model outputs against stored references; they assess numerical reproducibility, not clinical validity.

No software license is currently included. Contact the authors about reuse and redistribution permissions.
