# Complete modeling package

Prepared from the uploaded generator, CSV and original `Untitled10-2.ipynb`.

- Added MDP class, trajectory simulation, baseline-cohort plotting and an explicit CLI pipeline.
- Preserved generator bytes, original reference CSV, original notebook code and execution counts, clinical coding, model parameters, random seeds, smoothing/fallback rules, state order and numerical equations.
- Added CSV/NPZ exports, run metadata, an ordered notebook, numerical regression tests and thesis starting points.
- Figure presentation changes only: removed empty subplot row; restored visible numeric time ticks; replaced ambiguous QoL/QALY label with QALY per 3-month cycle. Displayed 0–4-year window and numerical curves are preserved.
- Added Matplotlib to dependencies. The extracted workflow does not need Seaborn or Jupyter; the archived original notebook imports Seaborn and needs a separate notebook environment if executed.
- Documented the 21-step graph run rather than repeating the erroneous 40-step comment.

No clinical-code inversion, dose-switch rule, frailty sign correction, terminal-state correction or reward-estimator change is applied. The existing generator, initializer and CSV were verified against GitHub main commit `602616628cd5beaa9f935e71b53633bb97ecbee2` and are unchanged. Existing ignore rules are retained.

- For GitHub, removed bulky saved outputs from the archived notebook; preserved its code, cell order and execution counts. The complete original with saved outputs remains in the supplied ZIP.

## Focused figure workflow

- Independently compared all four notebook generators: only the final In [4] matches the reference CSV in every column.
- Removed the exploratory notebook from the current branch; previous Git history retains it.
- The single clean notebook now contains four code cells, including final data generation and only the In [43]/In [44] figure workflow.
- Removed unused diagnostic/policy-iteration methods and policies not displayed in that figure. Preserved value iteration, three relevant policies, model arrays, random seeds and plotted curves.
- Regression tests use the previously verified reference arrays and curves, so no historical notebook is required.

## Show all three policies for Fit

Removed the display/export filter hiding reduced dose from Fit. Both panels now display all three policies over 0–60 months, with the layout and axis labels of the supplied reference image. Existing five curves are numerically unchanged; exported curve data now include the sixth series.

- The final display was located in original In [57], executed by In [59]. The current plotting function is extracted directly from that cell, retaining its return `(fig, axes)`; the pipeline is adapted accordingly.
