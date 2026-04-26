"""
Command-line script for generating synthetic geriatric oncology data.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_generation import N_VISITS, simulate_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic longitudinal patient data for downstream MDP modeling."
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
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/Fixed_patient_data.csv",
        help="Path to save the generated CSV file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = simulate_data(
        n_patients=args.n_patients,
        n_visits=args.n_visits,
        seed=args.seed,
        save_csv=True,
        csv_path=args.output,
    )

    print("\n--- Synthetic dataset summary ---")
    print(f"Rows: {len(df)}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Maximum visit: {int(df['visit'].max())}")
    print(f"Severe toxicity events: {int(df['severe_tox'].sum())}")
    print(f"Toxicity-related deaths: {int(df['death_tox'].sum())}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
