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
