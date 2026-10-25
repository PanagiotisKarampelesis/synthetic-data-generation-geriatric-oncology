"""Final baseline-cohort plot from In [57], executed by In [59]."""

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

def _aggregate_group(trajectory_results, state_names, policy_name):
    """
    Patient-level aggregation:
    - mean(t): mean των mean_qol_over_time(t) across states
    - std(t): std πάνω σε ΟΛΕΣ τις προσομοιώσεις των states (law of total variance)
    """
    mus = []
    sigs = []
    T = None

    for st in state_names:
        if st not in trajectory_results:
            continue
        if policy_name not in trajectory_results[st]:
            continue

        res = trajectory_results[st][policy_name]
        mu  = np.array(res['mean_qol_over_time'], dtype=float)
        sig = np.array(res['std_qol_over_time'],  dtype=float)

        if T is None:
            T = len(mu)
        else:
            T = min(T, len(mu))

        mus.append(mu[:T])
        sigs.append(sig[:T])

    if not mus:
        return None, None

    mus  = np.vstack(mus)   # shape: (S, T)
    sigs = np.vstack(sigs)  # shape: (S, T)

    # mean QALYs per cycle στο cohort
    mean = mus.mean(axis=0)   # E_s[μ_s(t)]

    # Var_total(t) = E_s[σ_s^2 + μ_s^2] - (E_s[μ_s])^2
    mean_mu2_plus_sig2 = (sigs**2 + mus**2).mean(axis=0)
    var  = mean_mu2_plus_sig2 - mean**2
    var  = np.clip(var, 0, None)
    std  = np.sqrt(var)

    return mean, std

def get_baseline_groups(mdp, trajectory_results):
    """
    Returns dict: {group_name: [state_names]} for the baseline cohorts:
      - Frail  (all frail, any dose)
      - Fit    (fit + full dose at baseline)
    """
    # map index->name that exists in trajectory_results
    idx_to_name = {
        i: mdp.states[i]['name']
        for i in range(mdp.n_states)
        if mdp.states[i]['name'] in trajectory_results
    }

    # all frail (dose 0 or 1)
    frail_states = [
        idx_to_name[i] for i in idx_to_name
        if mdp.states[i]['frailty'] == 1
    ]

    # fit + full dose at baseline
    fit_full_states = [
        idx_to_name[i] for i in idx_to_name
        if mdp.states[i]['frailty'] == 0 and mdp.states[i]['dose'] == 1
    ]

    groups = {
        "Frail": frail_states,
        "Fit": fit_full_states,
    }
    return groups

def visualize_trajectory_by_baseline_groups(
    mdp,
    optimal_policy,
    trajectory_results,
    savefig="Figure_5_corrected.png",
    policy_colors=None,
    max_months=60
):
    """
    Visualize QALY trajectories for frail and fit baseline groups.

    Parameters
    ----------
    max_months : int or None
        Maximum displayed follow-up time in months.
        Use 60 to reproduce the five-year figure.
        Use None to display the full simulated trajectory.
    """

    if policy_colors is None:
        policy_colors = {
            "optimal": "red",
            "always_reduced": "gold",
            "always_full": "blue",
        }

    policy_labels = {
        "optimal": "optimal policy",
        "always_reduced": "reduced dose",
        "always_full": "full dose",
    }

    groups = get_baseline_groups(mdp, trajectory_results)

    # Κρατάμε μόνο τα δύο panels που χρειαζόμαστε
    titles = [title for title in ("Frail", "Fit") if title in groups]

    if not titles:
        raise ValueError(
            "No 'Frail' or 'Fit' baseline groups were found."
        )

    # Και οι τρεις πολιτικές εμφανίζονται και στις δύο ομάδες
    allowed_policies_by_group = {
        "Frail": {"optimal", "always_full", "always_reduced"},
        "Fit": {"optimal", "always_full", "always_reduced"},
    }

    fig, axes = plt.subplots(
        1,
        len(titles),
        figsize=(16, 6),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    axes = axes.ravel()

    y_min, y_max = 0.005, 0.17

    for ax, title in zip(axes, titles):
        states_in_group = groups[title]
        allowed_policies = allowed_policies_by_group[title]

        group_max_months = 0

        for policy_name, color in policy_colors.items():
            if policy_name not in allowed_policies:
                continue

            mean_values, spread_values = _aggregate_group(
                trajectory_results,
                states_in_group,
                policy_name
            )

            if mean_values is None:
                continue

            mean_values = np.asarray(mean_values)
            spread_values = np.asarray(spread_values)

            # Κάθε visit αντιστοιχεί σε τρίμηνο
            visit_idx = np.arange(len(mean_values))
            x_months = visit_idx * 3

            if len(x_months) > 0:
                group_max_months = max(
                    group_max_months,
                    float(x_months[-1])
                )

            label = policy_labels.get(
                policy_name,
                policy_name.replace("_", " ")
            )

            ax.plot(
                x_months,
                mean_values,
                label=label,
                linewidth=3.0,
                color=color,
                alpha=0.9
            )

            ax.fill_between(
                x_months,
                mean_values - spread_values,
                mean_values + spread_values,
                color=color,
                alpha=0.18
            )

        ax.set_title(title, fontsize=20)

        # Axis labels και στα δύο panels
        ax.set_xlabel("Time (months)", fontsize=17)
        ax.set_ylabel("QALYs per 3-month cycle", fontsize=17)

        # Εμφάνιση y-axis tick labels και στο δεξί panel
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=13,
            labelleft=True
        )

        if max_months is not None:
            ax.set_xlim(0, max_months)

            # Για 0–60 μήνες: ticks ανά 3 μήνες, όπως στο αρχικό figure
            if max_months <= 60:
                ax.xaxis.set_major_locator(
                    mticker.MultipleLocator(3)
                )
            else:
                ax.xaxis.set_major_locator(
                    mticker.MultipleLocator(12)
                )
        else:
            displayed_max = group_max_months or 60
            ax.set_xlim(0, displayed_max)
            ax.xaxis.set_major_locator(
                mticker.MultipleLocator(12)
            )

        ax.xaxis.set_major_formatter(
            mticker.FormatStrFormatter("%d")
        )

        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13, loc="best")

    fig.suptitle(
        "QALY trajectories for frail and fit patients under different policies",
        fontsize=22,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if savefig:
        plt.savefig(
            savefig,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    return fig, axes
