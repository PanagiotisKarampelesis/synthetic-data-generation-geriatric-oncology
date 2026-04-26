# Synthetic Data Generation for Geriatric Oncology MDP Models

This repository contains code for generating synthetic longitudinal patient data for downstream use in Markov Decision Process (MDP) models for geriatric oncology treatment decision support.

The generated dataset is designed to support methodological experiments in which patient trajectories are used to evaluate adaptive treatment policies. The data are fully synthetic and do not contain real patient-level clinical information.

## Overview

The simulator generates a cohort of older breast cancer patients followed over quarterly visits. Each patient trajectory includes demographic, clinical, treatment, toxicity, quality-of-life, and QALY-related variables.

The generated data can be used as input for downstream MDP models in which treatment decisions are evaluated across patient states.

## Simulation Design

By default, the simulator generates:

- 500 synthetic patients
- 40 quarterly visits, corresponding to 10 years of follow-up
- baseline ages sampled from 71 to 95 years
- baseline frailty assignment with probability 0.7 for frail and 0.3 for fit
- initial treatment dose assignment based on frailty status
- longitudinal clinical status, frailty status, toxicity events, quality of life, and cumulative QALYs

Fit patients start at full dose. Frail patients start either at full dose or at reduced/held dose with equal probability.

The generated trajectories represent synthetic natural-history treatment courses under the initially assigned dose. Adaptive treatment decisions, such as dose escalation, dose reduction, treatment holding, or discontinuation, are intended to be evaluated downstream in an MDP framework.

## Variables

| Variable | Description |
|---|---|
| `patient_id` | Unique synthetic patient identifier |
| `age` | Patient age at each visit |
| `visit` | Visit index; 0 is baseline, 1–40 are quarterly follow-up visits |
| `frailty` | Frailty status; 0 = fit, 1 = frail |
| `clinical_test` | Clinical status indicator; 0 = favorable/controlled, 1 = unfavorable/active or poor clinical state |
| `full_dose` | Treatment dose indicator; 0 = reduced or held dose, 1 = full dose |
| `cum_dose_cat` | Cumulative full-dose exposure |
| `QOL` | Quality-of-life score in [0, 1], recorded at baseline and annual visits |
| `qaly_quarter` | Quality-adjusted life-year contribution during the quarter |
| `cum_qaly` | Cumulative QALYs over follow-up |
| `severe_tox` | Severe grade 3/4 toxicity indicator |
| `death_tox` | Toxicity-related death indicator |

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/YOUR_USERNAME/synthetic-data-generation-geriatric-oncology.git
cd synthetic-data-generation-geriatric-oncology
pip install -r requirements.txt


