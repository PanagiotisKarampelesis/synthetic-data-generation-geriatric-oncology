"""Run MDP fitting, policy simulation and figure generation."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import run_analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'Fixed_patient_data.csv')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'generated'/'mdp')
    parser.add_argument('--n-simulations',type=int,default=500)
    parser.add_argument('--n-steps',type=int,default=21,
                        help='Number of quarterly reward cycles (default: 21).')
    args=parser.parse_args()
    if args.n_simulations<=0 or args.n_steps<=0:
        parser.error('Counts must be positive.')
    run_analysis(args.input,args.output_dir,args.n_simulations,args.n_steps)
    print(f'Outputs saved to {args.output_dir.resolve()}')


if __name__=='__main__':
    main()
