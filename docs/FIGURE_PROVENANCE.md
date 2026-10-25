# Selection of the paper-figure workflow

The original notebook uses execution counts, not positional cell numbers.

## Generator selection

Each of the first four generator definitions was run independently with 500 patients, 40 visits, seed 42 and no file writes, then compared with the supplied reference CSV.

| Original execution count | Rows | Columns | Matches reference |
|---|---:|---:|---|
| In [1] | 17,030 | 12 | No |
| In [2] | 16,887 | 11 | No; cumulative-exposure column absent |
| In [3] | 15,266 | 12 | No |
| In [4] | 15,798 | 12 | Yes, all columns within 1e-12 |

The standalone `src/data_generation.py` reproduces In [4] and remains unchanged. Earlier generators are excluded.

## Required dependencies of In [44]

| Original source | Retained purpose |
|---|---|
| In [4] | Final synthetic-data generator |
| In [12] | State definition, reward/transition estimation, value iteration |
| In [15] | Model settings and fitting sequence; extra solver comparison omitted |
| In [16] | Trajectory simulation; only optimal/full/reduced policies retained |
| In [25] | Frail/fit groups and baseline-cohort plotting function |
| In [43] | Mean and pooled-SD aggregation used by the figure |
| In [44] | 500 simulations and 21-step run |

Earlier QoL plots, cumulative plots, diagnostic cells, repeated definitions, policy-iteration comparison, unrelated policy comparisons and later In [57]/In [59]/In [63] plotting experiments are not required by this figure and are excluded from the clean notebook. The original exploratory notebook is no longer shipped in the current branch; it remains in earlier Git history and the supplied original files.

The clean notebook contains four code cells: imports, final data generation, MDP/trajectory run, and image display. Numerical equations and the five displayed curves are preserved. Plot layout and unit labels are clarified as documented in the README.
