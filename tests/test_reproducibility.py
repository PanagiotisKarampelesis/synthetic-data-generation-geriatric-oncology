"""Regression against the supplied reference, not clinical validation."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_generation import simulate_data


def generate(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return simulate_data(save_csv=False, **kwargs)


class ReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = pd.read_csv(ROOT / 'Fixed_patient_data.csv')
        cls.actual = generate()

    def test_reference_file_hashes(self):
        manifest = json.loads((ROOT / 'reference_manifest.json').read_text())
        for relative, expected in manifest['sha256'].items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)

    def test_every_column_matches_reference(self):
        ref, actual = self.reference, self.actual
        self.assertEqual(ref.shape, actual.shape)
        self.assertEqual(list(ref.columns), list(actual.columns))
        exact = {'patient_id', 'visit', 'frailty', 'clinical_test', 'full_dose', 'severe_tox', 'death_tox'}
        for column in ref:
            with self.subTest(column=column):
                np.testing.assert_array_equal(ref[column].isna(), actual[column].isna())
                if column in exact:
                    np.testing.assert_array_equal(ref[column], actual[column])
                else:
                    np.testing.assert_allclose(ref[column], actual[column], rtol=0, atol=1e-12, equal_nan=True)

    def test_seed_repeatability(self):
        pd.testing.assert_frame_equal(self.actual, generate(), check_exact=True)

    def test_reference_summary(self):
        d = self.actual
        self.assertEqual((len(d), d.patient_id.nunique(), len(d)-d.patient_id.nunique()), (15798, 500, 15298))
        self.assertEqual((int(d.severe_tox.sum()), int(d.death_tox.sum())), (2108, 212))
        self.assertEqual(d.groupby('patient_id').full_dose.nunique().max(), 1)

    def test_trajectory_and_reward_integrity(self):
        d = self.actual
        self.assertFalse(d.duplicated(['patient_id', 'visit']).any())
        for _, g in d.groupby('patient_id'):
            np.testing.assert_array_equal(g.visit, np.arange(len(g)))
            self.assertLessEqual(g.death_tox.sum(), 1)
            self.assertEqual(g.death_tox.sum(), g.death_tox.iloc[-1])
            self.assertTrue(g.visit.iloc[-1] == 40 or g.death_tox.iloc[-1] == 1)
        np.testing.assert_allclose(d.cum_qaly, d.groupby('patient_id').qaly_quarter.cumsum(), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(d.QOL.notna(), d.visit.mod(4).eq(0))
        annual = d.visit.gt(0) & d.QOL.notna()
        np.testing.assert_allclose(d.loc[annual, 'qaly_quarter'], d.loc[annual, 'QOL']*.25, rtol=0, atol=1e-12)
        self.assertTrue(d.QOL.dropna().between(0,1).all())
        self.assertTrue(d.qaly_quarter.between(0,.25).all())
        baseline_age = d.groupby('patient_id').age.transform('first')
        np.testing.assert_array_equal(d.age, baseline_age + .25*np.maximum(d.visit-1, 0))

    def test_invalid_counts(self):
        for kwargs in ({'n_patients': 0}, {'n_visits': 0}, {'n_patients': -1}, {'n_visits': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate(**kwargs)

    def test_cli_matches_direct_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.csv'
            result = subprocess.run([sys.executable, str(ROOT/'scripts/generate_synthetic_data.py'), '--n_patients', '3', '--n_visits', '4', '--output', str(output)], cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            pd.testing.assert_frame_equal(pd.read_csv(output), generate(n_patients=3, n_visits=4), check_exact=False, rtol=0, atol=1e-12)

    def test_cli_protects_reference(self):
        path = ROOT / 'Fixed_patient_data.csv'
        before = path.read_bytes()
        result = subprocess.run([sys.executable, str(ROOT/'scripts/generate_synthetic_data.py'), '--output', str(path)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('reference CSV is protected', result.stderr)
        self.assertEqual(before, path.read_bytes())


if __name__ == '__main__':
    unittest.main()
