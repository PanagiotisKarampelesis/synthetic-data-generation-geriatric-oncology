"""Regression against verified original In [43]/In [44] numerical outputs."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mdp import SimpleQOLMDP
from src.trajectories import create_alternative_policies, analyze_trajectories_by_initial_state
from src.plotting import _aggregate_group, get_baseline_groups, visualize_trajectory_by_baseline_groups


class MDPRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=np.load(ROOT/'results/reference/mdp_arrays.npz')
        cls.curves=pd.read_csv(ROOT/'results/reference/figure_curves.csv')
        with contextlib.redirect_stdout(io.StringIO()):
            cls.actual=SimpleQOLMDP(str(ROOT/'Fixed_patient_data.csv'))
            cls.actual.estimate_qol_rewards(lookahead_steps=3,reward_mode='qaly',visit_len_years=.25)
            cls.actual.estimate_transition_probabilities(alpha=.5)
            cls.values,cls.policy=cls.actual.value_iteration()
            cls.policies=create_alternative_policies(cls.actual,cls.policy)
            cls.results=analyze_trajectories_by_initial_state(cls.actual,cls.policy,cls.policies,n_simulations=500,n_steps=21)

    def test_exact_model_arrays(self):
        for name,actual in [('R',self.actual.R),('P',self.actual.P),('values',self.values),('optimal_policy',self.policy)]:
            np.testing.assert_array_equal(actual,self.reference[name])

    def test_only_figure_policies(self):
        self.assertEqual(set(self.policies),{'optimal','always_full','always_reduced'})
        np.testing.assert_array_equal(self.policies['optimal'],self.reference['optimal_policy'])
        np.testing.assert_array_equal(self.policies['always_full'],np.ones(8,dtype=int))
        np.testing.assert_array_equal(self.policies['always_reduced'],np.zeros(8,dtype=int))
        for result in self.results.values():
            self.assertEqual(set(result),set(self.policies))

    def test_figure_curves_and_bands(self):
        for group,names in get_baseline_groups(self.actual,self.results).items():
            for policy in ['optimal','always_full','always_reduced']:
                mean,sd=_aggregate_group(self.results,names,policy)
                ref=self.curves[(self.curves.group==group)&(self.curves.policy==policy)].sort_values('step_index')
                # CSV storage roundoff only; numerical source functions were independently checked.
                np.testing.assert_allclose(mean,ref.mean_qaly_per_cycle,rtol=0,atol=1e-12)
                np.testing.assert_allclose(sd,ref.sd_qaly_per_cycle,rtol=0,atol=1e-12)

    def test_three_policies_in_each_panel(self):
        import matplotlib.pyplot as plt
        fig, _=visualize_trajectory_by_baseline_groups(self.actual,self.policy,self.results,savefig=None)
        for ax in fig.axes:
            self.assertEqual([line.get_label() for line in ax.lines],
                             ['optimal policy','reduced dose','full dose'])
            self.assertEqual(tuple(ax.get_xlim()),(0,60))
        plt.close(fig)

    def test_saved_notebook_summary(self):
        for policy,value in {'optimal':2.127,'always_reduced':1.843,'always_full':1.634}.items():
            actual=np.mean([res[policy]['mean_cumulative_qol'][-1] for res in self.results.values()])
            self.assertEqual(round(float(actual),3),value)

    def test_probabilities(self):
        self.assertTrue(np.isfinite(self.actual.P).all())
        self.assertTrue((self.actual.P>=0).all())
        np.testing.assert_allclose(self.actual.P.sum(axis=2),1,rtol=0,atol=1e-14)


if __name__=='__main__':unittest.main()
