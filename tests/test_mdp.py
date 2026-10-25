"""Numerical equivalence with original notebook functions, not scientific validation."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mdp import SimpleQOLMDP
from src.trajectories import create_alternative_policies, analyze_trajectories_by_initial_state
from src.plotting import _aggregate_group, get_baseline_groups


def original_namespace():
    notebook=json.loads((ROOT/'archive/Untitled10-2.ipynb').read_text())
    cells={c.get('execution_count'):''.join(c.get('source',[])) for c in notebook['cells'] if c.get('execution_count') is not None}
    ns={}
    exec(cells[12],ns)
    # Execute only needed imports and function definitions, avoiding plots and notebook state.
    selected={'simulate_trajectory','simulate_multiple_trajectories','create_alternative_policies','analyze_trajectories_by_initial_state'}
    tree=ast.parse(cells[16])
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in selected:
            exec(ast.get_source_segment(cells[16],node),ns)
    exec(cells[43],ns)
    return ns


class MDPRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=original_namespace()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.actual=SimpleQOLMDP(str(ROOT/'Fixed_patient_data.csv'))
            cls.reference=cls.original['SimpleQOLMDP'](str(ROOT/'Fixed_patient_data.csv'))
            for model in (cls.actual,cls.reference):
                model.estimate_qol_rewards(lookahead_steps=3,reward_mode='qaly',visit_len_years=.25)
                model.estimate_transition_probabilities(alpha=.5)
            cls.values,cls.policy=cls.actual.value_iteration()
            cls.ref_values,cls.ref_policy=cls.reference.value_iteration()
            cls.policies=create_alternative_policies(cls.actual,cls.policy)
            cls.ref_policies=cls.original['create_alternative_policies'](cls.reference,cls.ref_policy)
            cls.results=analyze_trajectories_by_initial_state(cls.actual,cls.policy,cls.policies,n_simulations=500,n_steps=21)
            cls.ref_results=cls.original['analyze_trajectories_by_initial_state'](cls.reference,cls.ref_policy,cls.ref_policies,n_simulations=500,n_steps=21)

    def test_exact_model_arrays(self):
        for a,b in [(self.actual.R,self.reference.R),(self.actual.P,self.reference.P),(self.values,self.ref_values),(self.policy,self.ref_policy)]:
            np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(self.policy,[0,1,0,1,1,0,1,0])

    def test_exact_trajectories_and_policies(self):
        for name in self.policies:
            np.testing.assert_array_equal(self.policies[name],self.ref_policies[name])
        self.assertEqual(self.results,self.ref_results)

    def test_exact_figure_curves_and_bands(self):
        for group,names in get_baseline_groups(self.actual,self.results).items():
            for policy in ['optimal','always_full','always_reduced']:
                if group=='Fit' and policy=='always_reduced':
                    continue
                actual=_aggregate_group(self.results,names,policy)
                reference=self.original['_aggregate_group'](self.ref_results,names,policy)
                for a,b in zip(actual,reference):
                    np.testing.assert_array_equal(a,b)

    def test_saved_notebook_summary(self):
        expected={'optimal':2.127,'always_reduced':1.843,'always_full':1.634,
                  'clinical_test_based':1.831,'frailty_based':1.819,'random':1.488,'inverse_optimal':1.430}
        for policy,value in expected.items():
            with self.subTest(policy=policy):
                actual=np.mean([res[policy]['mean_cumulative_qol'][-1] for res in self.results.values()])
                self.assertEqual(round(float(actual),3),value)

    def test_probabilities(self):
        self.assertTrue(np.isfinite(self.actual.P).all())
        self.assertTrue((self.actual.P>=0).all())
        np.testing.assert_allclose(self.actual.P.sum(axis=2),1,rtol=0,atol=1e-14)


if __name__=='__main__':
    unittest.main()
