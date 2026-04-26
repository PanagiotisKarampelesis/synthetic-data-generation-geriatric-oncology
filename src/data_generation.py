"""
Synthetic longitudinal data generation for downstream use in
Markov Decision Process (MDP) models in geriatric oncology.

The generated data are synthetic and intended for methodological
demonstration only. They are not real clinical data.
"""

import os
import numpy as np
import pandas as pd


VISIT_LEN_YEARS = 0.25
N_VISITS = 40

QOL_VISITS = {0} | set(range(4, 41, 4))
VISIT_TO_YEARS = {
    0: "baseline",
    **{
        v: f"{v // 4} year" + ("s" if v // 4 > 1 else "")
        for v in sorted(QOL_VISITS)
        if v != 0
    },
}

AGE_PENALTY_PER_YEAR = 0.015
NOISE_SIGMA = 0.03

DOSE_PEN_FULL = (0.040, 0.060)
DOSE_PEN_REDUCED = 0.006

EFFICACY_FULL = 0.015
EFFICACY_REDUCED = 0.008

SEVERE_TOX_QOL_HIT_FIT = 0.15
SEVERE_TOX_QOL_HIT_FRAIL = 0.22
CARRYOVER_DAMAGE_FRACTION = 0.80
QOL_DAMAGE_RECOVERY_PER_QUARTER = 0.20

CUM_DOSE_SLOPE = 0.05
MAX_CUM_DOSE_EFFECT = 0.25
CUM_DOSE_RECOVERY_PER_QUARTER = 0.10


def update_cum_dose_cat(cum_dose_cat: float, dose: int) -> float:
    """Update cumulative full-dose exposure."""
    if dose == 1:
        cum_dose_cat += 1.0
    else:
        cum_dose_cat = max(0.0, cum_dose_cat - CUM_DOSE_RECOVERY_PER_QUARTER)
    return float(cum_dose_cat)


def cum_dose_effect_on_p_unfavorable(cum_dose_cat: float) -> float:
    """Linear cumulative full-dose effect on unfavorable clinical status probability."""
    extra = CUM_DOSE_SLOPE * cum_dose_cat
    return float(np.clip(extra, 0.0, MAX_CUM_DOSE_EFFECT))


def p_toxicity_g34(age: float, dose: int) -> float:
    """Quarterly probability of grade 3/4 toxicity."""
    base = 0.12 if age < 65 else 0.17
    rr = 1.0 if dose == 1 else 0.55
    return float(np.clip(base * rr, 0, 1))


def p_toxicity_death(age: float, dose: int) -> float:
    """Quarterly probability of toxicity-related death."""
    base = 0.0015 if age < 65 else 0.0145
    rr = 1.0 if dose == 1 else 0.50
    return float(np.clip(base * rr, 0, 1))


def base_next_clinical_unfavorable_prob(
    current_test: int,
    dose: int,
    cum_dose_cat: float = 0.0,
) -> float:
    """
    Probability that the next clinical status is unfavorable.

    clinical_test convention:
        0 = favorable / controlled clinical state
        1 = unfavorable / active or poor clinical state
    """
    if current_test == 1:
        p_unfavorable = 0.40
        p_unfavorable *= 0.92 if dose == 1 else 1.03
    else:
        p_unfavorable = 0.20
        p_unfavorable *= 0.95 if dose == 1 else 1.05

    p_unfavorable += cum_dose_effect_on_p_unfavorable(cum_dose_cat)

    return float(np.clip(p_unfavorable, 0.05, 0.99))


def next_clinical_unfavorable_prob(
    current_test: int,
    dose: int,
    cum_dose_cat: float,
) -> float:
    """Wrapper for clinical status transition probability."""
    return base_next_clinical_unfavorable_prob(current_test, dose, cum_dose_cat)


def simulate_qol_baseline(frailty: int, age: float, rng: np.random.Generator) -> float:
    """Simulate baseline quality of life before treatment effects."""
    lo, hi = (0.6, 0.90) if frailty == 0 else (0.4, 0.70)
    base_qol = rng.uniform(lo, hi)
    age_penalty = max(0.0, (age - 75) * AGE_PENALTY_PER_YEAR)
    noise = rng.normal(0, NOISE_SIGMA)
    qol = base_qol - age_penalty + noise
    return float(np.clip(qol, 0, 1))


def simulate_qol(
    frailty: int,
    age: float,
    dose: int,
    clinical_test: int,
    rng: np.random.Generator,
    carryover_damage: float,
) -> float:
    """Simulate quality of life during treatment."""
    lo, hi = (0.6, 0.90) if frailty == 0 else (0.4, 0.70)
    base_qol = rng.uniform(lo, hi)
    age_penalty = max(0.0, (age - 75) * AGE_PENALTY_PER_YEAR)

    if dose == 1 and frailty == 0:
        dose_penalty = DOSE_PEN_FULL[0]
    elif dose == 1 and frailty == 1:
        dose_penalty = DOSE_PEN_FULL[1]
    else:
        dose_penalty = DOSE_PEN_REDUCED

    efficacy = 0.0
    if clinical_test == 1:
        efficacy = EFFICACY_FULL if dose == 1 else EFFICACY_REDUCED

    noise = rng.normal(0, NOISE_SIGMA)
    qol = base_qol - age_penalty - dose_penalty + efficacy + noise - carryover_damage
    return float(np.clip(qol, 0, 1))


def frailty_transition(
    frail: int,
    age: float,
    dose: int,
    clinical_test: int,
    unfavorable_streak: int,
    rng: np.random.Generator,
) -> int:
    """
    Simulate transition between fit and frail states.

    frailty convention:
        0 = fit
        1 = frail

    clinical_test convention:
        0 = favorable / controlled
        1 = unfavorable / active or poor clinical state
    """
    if frail == 0:
        p = 0.014 if age <= 75 else (0.022 if age <= 80 else 0.036)

        if dose == 1:
            p -= 0.003

        if unfavorable_streak >= 2:
            p += 0.008 if dose == 0 else 0.003

        return 1 if rng.random() < np.clip(p, 0, 1) else 0

    recovery_prob = 0.010 if age <= 75 else (0.008 if age <= 80 else 0.006)

    if dose == 1:
        recovery_prob += 0.004

    if clinical_test == 0:
        recovery_prob += 0.003

    return 0 if rng.random() < np.clip(recovery_prob, 0, 1) else 1


def assign_initial_dose(frail: int, rng: np.random.Generator) -> int:
    """
    Assign initial treatment dose.

    full_dose convention:
        0 = reduced or held dose
        1 = full dose
    """
    return 1 if frail == 0 else int(rng.random() < 0.5)


def simulate_data(
    n_patients: int = 500,
    n_visits: int = N_VISITS,
    seed: int = 42,
    save_csv: bool = True,
    csv_path: str = "data/Fixed_patient_data.csv",
) -> pd.DataFrame:
    """
    Generate a synthetic longitudinal patient dataset.

    Returns
    -------
    pandas.DataFrame
        Synthetic patient trajectories.
    """
    rng = np.random.default_rng(seed)

    ids = np.arange(1, n_patients + 1)
    ages = rng.integers(71, 96, size=n_patients)
    frail0 = rng.choice([1, 0], size=n_patients, p=[0.7, 0.3])
    dose0 = np.array([assign_initial_dose(f, rng) for f in frail0])
    test0 = rng.integers(0, 2, size=n_patients)

    print(
        f"Baseline: {np.mean(frail0 == 0) * 100:.1f}% fit / "
        f"{np.mean(frail0 == 1) * 100:.1f}% frail (n={n_patients})"
    )

    records = []

    for pid, age0, frail_init, dose_init, test_init in zip(
        ids, ages, frail0, dose0, test0
    ):
        age = float(age0)
        frail = int(frail_init)
        dose = int(dose_init)
        clinical_test = int(test_init)
        alive = True
        cum_qaly = 0.0

        unfavorable_streak = 1 if clinical_test == 1 else 0
        favorable_streak = 1 if clinical_test == 0 else 0

        carryover_damage = 0.0
        cum_dose_cat = 0.0

        baseline_qol = simulate_qol_baseline(frail, age, rng)

        records.append(
            {
                "patient_id": pid,
                "age": round(age, 2),
                "visit": 0,
                "frailty": frail,
                "clinical_test": clinical_test,
                "full_dose": dose,
                "cum_dose_cat": float(cum_dose_cat),
                "QOL": baseline_qol,
                "qaly_quarter": 0.0,
                "cum_qaly": 0.0,
                "severe_tox": 0,
                "death_tox": 0,
            }
        )

        for visit_idx in range(n_visits):
            visit = visit_idx + 1

            if not alive:
                break

            carryover_damage *= 1.0 - QOL_DAMAGE_RECOVERY_PER_QUARTER

            qol = simulate_qol(
                frailty=frail,
                age=age,
                dose=dose,
                clinical_test=clinical_test,
                rng=rng,
                carryover_damage=carryover_damage,
            )

            died_tox = rng.random() < p_toxicity_death(age, dose)
            severe_tox = False

            if not died_tox:
                severe_tox = rng.random() < p_toxicity_g34(age, dose)

                if severe_tox:
                    hit = (
                        SEVERE_TOX_QOL_HIT_FRAIL
                        if frail == 1
                        else SEVERE_TOX_QOL_HIT_FIT
                    )
                    qol = max(0.0, qol - hit)
                    carryover_damage += hit * CARRYOVER_DAMAGE_FRACTION

            qaly = qol * VISIT_LEN_YEARS
            cum_qaly += qaly

            records.append(
                {
                    "patient_id": pid,
                    "age": round(age, 2),
                    "visit": visit,
                    "frailty": frail,
                    "clinical_test": clinical_test,
                    "full_dose": dose,
                    "cum_dose_cat": float(cum_dose_cat),
                    "QOL": qol if visit in QOL_VISITS else np.nan,
                    "qaly_quarter": qaly,
                    "cum_qaly": cum_qaly,
                    "severe_tox": int(severe_tox),
                    "death_tox": int(died_tox),
                }
            )

            if died_tox:
                alive = False
                break

            cum_dose_cat = update_cum_dose_cat(cum_dose_cat, dose)

            p_next_unfavorable = next_clinical_unfavorable_prob(
                current_test=clinical_test,
                dose=dose,
                cum_dose_cat=cum_dose_cat,
            )

            clinical_test = 1 if rng.random() < p_next_unfavorable else 0

            if clinical_test == 1:
                unfavorable_streak += 1
                favorable_streak = 0
            else:
                favorable_streak += 1
                unfavorable_streak = 0

            frail = frailty_transition(
                frail=frail,
                age=age,
                dose=dose,
                clinical_test=clinical_test,
                unfavorable_streak=unfavorable_streak,
                rng=rng,
            )

            age += VISIT_LEN_YEARS

    df = pd.DataFrame(records)

    if save_csv:
        output_dir = os.path.dirname(csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(csv_path):
            os.remove(csv_path)

        df.to_csv(csv_path, index=False)
        print(f"Saved '{csv_path}' ({len(df)} rows).")

    return df

