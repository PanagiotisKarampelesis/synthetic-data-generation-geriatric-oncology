"""
Standalone command-line script for generating synthetic longitudinal
geriatric-oncology data.

Example
-------
python generate_synthetic_data.py \
    --n_patients 500 \
    --n_visits 40 \
    --seed 42 \
    --output data/Fixed_patient_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Global simulation settings
# ---------------------------------------------------------------------

VISIT_LEN_YEARS = 0.25  # One visit corresponds to one 3-month cycle.
N_VISITS = 40            # Default: 40 quarterly visits = 10 years.

AGE_PENALTY_PER_YEAR = 0.015
NOISE_SIGMA = 0.03

# Dose burden on quality of life.
DOSE_PEN_FULL = (0.040, 0.060)  # (fit, frail)
DOSE_PEN_REDUCED = 0.006

# Efficacy contribution when clinical_test == 1.
EFFICACY_FULL = 0.015
EFFICACY_REDUCED = 0.008

# Toxicity-related quality-of-life effects.
SEVERE_TOX_QOL_HIT_FIT = 0.15
SEVERE_TOX_QOL_HIT_FRAIL = 0.22
CARRYOVER_DAMAGE_FRACTION = 0.80
QOL_DAMAGE_RECOVERY_PER_QUARTER = 0.20

# Cumulative full-dose exposure.
CUM_DOSE_SLOPE = 0.05
MAX_CUM_DOSE_EFFECT = 0.25
CUM_DOSE_RECOVERY_PER_QUARTER = 0.10


def annual_qol_visits(n_visits: int) -> set[int]:
    """Return baseline and annual QoL-assessment visit indices."""
    return {0} | set(range(4, n_visits + 1, 4))


def update_cum_dose_cat(cum_dose_cat: float, dose: int) -> float:
    """Update cumulative full-dose exposure."""
    if dose == 1:
        cum_dose_cat += 1.0
    else:
        cum_dose_cat = max(
            0.0,
            cum_dose_cat - CUM_DOSE_RECOVERY_PER_QUARTER,
        )
    return float(cum_dose_cat)


def cum_dose_effect_on_p_pos(cum_dose_cat: float) -> float:
    """Return the capped cumulative-dose contribution to P(test=1)."""
    extra = CUM_DOSE_SLOPE * cum_dose_cat
    return float(np.clip(extra, 0.0, MAX_CUM_DOSE_EFFECT))


def p_toxicity_g34(age: float, dose: int) -> float:
    """Quarterly probability of grade 3/4 toxicity."""
    base = 0.12 if age < 65 else 0.17
    relative_risk = 1.0 if dose == 1 else 0.55
    return float(np.clip(base * relative_risk, 0.0, 1.0))


def p_toxicity_death(age: float, dose: int) -> float:
    """Quarterly probability of toxicity-related death."""
    base = 0.0015 if age < 65 else 0.0145
    relative_risk = 1.0 if dose == 1 else 0.50
    return float(np.clip(base * relative_risk, 0.0, 1.0))


def base_next_test_positive_prob(
    current_test: int,
    dose: int,
    cum_dose_cat: float = 0.0,
) -> float:
    """Calculate the probability that the next clinical test equals 1."""
    if current_test == 1:
        probability = 0.40
        probability *= 0.92 if dose == 1 else 1.03
    else:
        probability = 0.20
        probability *= 0.95 if dose == 1 else 1.05

    probability += cum_dose_effect_on_p_pos(cum_dose_cat)
    return float(np.clip(probability, 0.05, 0.99))


def next_test_positive_prob(
    current_test: int,
    dose: int,
    cum_dose_cat: float,
) -> float:
    """Return P(next clinical_test = 1)."""
    return base_next_test_positive_prob(
        current_test=current_test,
        dose=dose,
        cum_dose_cat=cum_dose_cat,
    )


def simulate_qol_baseline(
    frailty: int,
    age: float,
    rng: np.random.Generator,
) -> float:
    """Generate baseline quality of life before treatment effects."""
    lower, upper = (0.6, 0.90) if frailty == 0 else (0.4, 0.70)
    base_qol = rng.uniform(lower, upper)
    age_penalty = max(0.0, (age - 75) * AGE_PENALTY_PER_YEAR)
    noise = rng.normal(0.0, NOISE_SIGMA)

    qol = base_qol - age_penalty + noise
    return float(np.clip(qol, 0.0, 1.0))


def simulate_qol(
    frailty: int,
    age: float,
    dose: int,
    clinical_test: int,
    rng: np.random.Generator,
    carryover_damage: float,
) -> float:
    """Generate quarterly quality of life after treatment initiation."""
    lower, upper = (0.6, 0.90) if frailty == 0 else (0.4, 0.70)
    base_qol = rng.uniform(lower, upper)
    age_penalty = max(0.0, (age - 75) * AGE_PENALTY_PER_YEAR)

    if dose == 1 and frailty == 0:
        dose_penalty = DOSE_PEN_FULL[0]
    elif dose == 1 and frailty == 1:
        dose_penalty = DOSE_PEN_FULL[1]
    else:
        dose_penalty = DOSE_PEN_REDUCED

    efficacy = (
        EFFICACY_FULL if dose == 1 else EFFICACY_REDUCED
    ) if clinical_test == 1 else 0.0

    noise = rng.normal(0.0, NOISE_SIGMA)

    qol = (
        base_qol
        - age_penalty
        - dose_penalty
        + efficacy
        + noise
        - carryover_damage
    )
    return float(np.clip(qol, 0.0, 1.0))


def frailty_transition(
    frail: int,
    age: float,
    dose: int,
    clinical_test: int,
    positive_streak: int,
    rng: np.random.Generator,
) -> int:
    """Simulate the next frailty state."""
    if frail == 0:
        probability = (
            0.014 if age <= 75
            else 0.022 if age <= 80
            else 0.036
        )

        if dose == 1:
            probability -= 0.003

        if positive_streak >= 2:
            probability += 0.008 if dose == 0 else 0.003

        return int(
            rng.random() < np.clip(probability, 0.0, 1.0)
        )

    recovery_probability = (
        0.010 if age <= 75
        else 0.008 if age <= 80
        else 0.006
    )

    if dose == 1:
        recovery_probability += 0.004

    if clinical_test == 0:
        recovery_probability += 0.003

    recovered = (
        rng.random()
        < np.clip(recovery_probability, 0.0, 1.0)
    )
    return 0 if recovered else 1


def assign_initial_dose(
    frail: int,
    rng: np.random.Generator,
) -> int:
    """Assign full dose to fit patients and randomize frail patients 1:1."""
    return 1 if frail == 0 else int(rng.random() < 0.5)


def simulate_data(
    n_patients: int = 500,
    n_visits: int = N_VISITS,
    seed: int = 42,
    save_csv: bool = True,
    csv_path: str | Path = "data/Fixed_patient_data.csv",
) -> pd.DataFrame:
    """Generate the synthetic longitudinal dataset."""
    if n_patients <= 0:
        raise ValueError("n_patients must be greater than zero.")

    if n_visits <= 0:
        raise ValueError("n_visits must be greater than zero.")

    rng = np.random.default_rng(seed)
    qol_visits = annual_qol_visits(n_visits)

    patient_ids = np.arange(1, n_patients + 1)
    baseline_ages = rng.integers(71, 96, size=n_patients)
    baseline_frailty = rng.choice(
        [1, 0],
        size=n_patients,
        p=[0.7, 0.3],
    )
    baseline_dose = np.array(
        [
            assign_initial_dose(frailty, rng)
            for frailty in baseline_frailty
        ]
    )
    baseline_test = rng.integers(0, 2, size=n_patients)

    print(
        f"Baseline: "
        f"{np.mean(baseline_frailty == 0) * 100:.1f}% fit / "
        f"{np.mean(baseline_frailty == 1) * 100:.1f}% frail "
        f"(n={n_patients})"
    )

    records: list[dict[str, float | int]] = []

    for (
        patient_id,
        baseline_age,
        frailty_initial,
        dose_initial,
        test_initial,
    ) in zip(
        patient_ids,
        baseline_ages,
        baseline_frailty,
        baseline_dose,
        baseline_test,
    ):
        age = float(baseline_age)
        frail = int(frailty_initial)
        dose = int(dose_initial)
        clinical_test = int(test_initial)
        cumulative_qaly = 0.0
        carryover_damage = 0.0
        cumulative_dose = 0.0
        positive_streak = 1 if clinical_test == 1 else 0

        baseline_qol = simulate_qol_baseline(
            frailty=frail,
            age=age,
            rng=rng,
        )

        records.append(
            {
                "patient_id": int(patient_id),
                "age": round(age, 2),
                "visit": 0,
                "frailty": frail,
                "clinical_test": clinical_test,
                "full_dose": dose,
                "cum_dose_cat": cumulative_dose,
                "QOL": baseline_qol,
                "qaly_quarter": 0.0,
                "cum_qaly": 0.0,
                "severe_tox": 0,
                "death_tox": 0,
            }
        )

        for visit in range(1, n_visits + 1):
            carryover_damage *= (
                1.0 - QOL_DAMAGE_RECOVERY_PER_QUARTER
            )

            qol = simulate_qol(
                frailty=frail,
                age=age,
                dose=dose,
                clinical_test=clinical_test,
                rng=rng,
                carryover_damage=carryover_damage,
            )

            died_from_toxicity = (
                rng.random() < p_toxicity_death(age, dose)
            )
            severe_toxicity = False

            if not died_from_toxicity:
                severe_toxicity = (
                    rng.random() < p_toxicity_g34(age, dose)
                )

                if severe_toxicity:
                    toxicity_hit = (
                        SEVERE_TOX_QOL_HIT_FRAIL
                        if frail == 1
                        else SEVERE_TOX_QOL_HIT_FIT
                    )
                    qol = max(0.0, qol - toxicity_hit)
                    carryover_damage += (
                        toxicity_hit * CARRYOVER_DAMAGE_FRACTION
                    )

            quarterly_qaly = qol * VISIT_LEN_YEARS
            cumulative_qaly += quarterly_qaly

            records.append(
                {
                    "patient_id": int(patient_id),
                    "age": round(age, 2),
                    "visit": visit,
                    "frailty": frail,
                    "clinical_test": clinical_test,
                    "full_dose": dose,
                    "cum_dose_cat": cumulative_dose,
                    "QOL": qol if visit in qol_visits else np.nan,
                    "qaly_quarter": quarterly_qaly,
                    "cum_qaly": cumulative_qaly,
                    "severe_tox": int(severe_toxicity),
                    "death_tox": int(died_from_toxicity),
                }
            )

            if died_from_toxicity:
                break

            cumulative_dose = update_cum_dose_cat(
                cumulative_dose,
                dose,
            )

            probability_next_positive = next_test_positive_prob(
                current_test=clinical_test,
                dose=dose,
                cum_dose_cat=cumulative_dose,
            )

            clinical_test = int(
                rng.random() < probability_next_positive
            )

            if clinical_test == 1:
                positive_streak += 1
            else:
                positive_streak = 0

            frail = frailty_transition(
                frail=frail,
                age=age,
                dose=dose,
                clinical_test=clinical_test,
                positive_streak=positive_streak,
                rng=rng,
            )

            # The dose remains fixed in this natural-history simulation.
            age += VISIT_LEN_YEARS

    dataframe = pd.DataFrame(records)

    if save_csv:
        output_path = Path(csv_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        dataframe.to_csv(
            output_path,
            index=False,
        )
        print(
            f"Saved '{output_path}' "
            f"({len(dataframe)} rows)."
        )

    return dataframe


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic longitudinal geriatric-oncology data."
        )
    )

    parser.add_argument(
        "--n_patients",
        type=int,
        default=500,
        help="Number of synthetic patients to simulate.",
    )
    parser.add_argument(
        "--n_visits",
        type=int,
        default=N_VISITS,
        help="Number of quarterly follow-up visits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/Fixed_patient_data.csv"),
        help="Path of the generated CSV file.",
    )

    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    """Generate the dataset and print a concise summary."""
    args = parse_args()

    dataframe = simulate_data(
        n_patients=args.n_patients,
        n_visits=args.n_visits,
        seed=args.seed,
        save_csv=True,
        csv_path=args.output,
    )

    print("\n--- Synthetic dataset summary ---")
    print(f"Rows: {len(dataframe)}")
    print(
        "Patients: "
        f"{dataframe['patient_id'].nunique()}"
    )
    print(
        "Maximum visit: "
        f"{int(dataframe['visit'].max())}"
    )
    print(
        "Severe toxicity events: "
        f"{int(dataframe['severe_tox'].sum())}"
    )
    print(
        "Toxicity-related deaths: "
        f"{int(dataframe['death_tox'].sum())}"
    )
    print(f"Output file: {args.output.resolve()}")


if __name__ == "__main__":
    main()
