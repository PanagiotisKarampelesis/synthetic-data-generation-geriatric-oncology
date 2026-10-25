# Understanding the baseline-group figure

The analysis compares three policies: the optimized state-dependent policy, always reduced dose, and always full dose.

## Groups and aggregation

The Frail group includes frail starting states across both clinical profiles and dose levels. The Fit group includes fit/full-dose starting states across both clinical profiles, consistent with baseline dose assignment in the data generator.

Each starting state has equal weight within its group. At each step, the mean is the average of the state-specific means. Pooled variance is `mean(within_state_variance + state_mean**2) - group_mean**2`. Shading shows the resulting mean ±1 standard deviation, not a confidence interval or an empirical cohort-frequency weighting.

## Axes and outputs

The vertical axis is QALYs per three-month cycle, not cumulative QALYs. The default horizontal axis labels 21 cycle starts from 0 to 60 months. There are 21 reward cycles, totaling 5.25 years.

Both panels display all three policies. `figure_curves.csv` contains six series and 126 rows at the default settings; its time column is expressed in years. `policy_summary.csv` separately reports final cumulative rewards with equal weighting across all eight starting states.

Run `python scripts/run_mdp.py` from the project root. The figure and underlying data are saved in `generated/mdp/`. See `IMPLEMENTATION_NOTES.md` for modeling assumptions and interpretation limits.
