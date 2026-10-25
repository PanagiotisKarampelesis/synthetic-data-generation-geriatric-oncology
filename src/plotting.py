"""Baseline-cohort plot used by In [44], with unambiguous labels and visible ticks."""

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
    mdp, optimal_policy, trajectory_results,
    savefig='qol_trajectory_baseline_groups.png',
    policy_colors=None
):
    # Χρωματισμοί για τις πολιτικές
    if policy_colors is None:
        policy_colors = {
            'optimal': 'red',
            'always_reduced': 'gold',
            'always_full': 'blue',
        }

    # Labels για να φαίνονται «ανθρώπινα» στο legend
    policy_labels = {
        'optimal': 'optimal policy',
        'always_full': 'full dose',
        'always_reduced': 'reduced dose',
    }

    groups = get_baseline_groups(mdp, trajectory_results)

    desired_titles = ["Frail", "Fit"]
    titles = [t for t in desired_titles if t in groups] + \
             [t for t in groups.keys() if t not in desired_titles]

    # Επιτρεπόμενες πολιτικές ανά group
    allowed_policies_by_group = {
        "Frail": {"optimal", "always_full", "always_reduced"},
        "Fit": {"optimal", "always_full"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True, sharey=True)
    axes = axes.ravel()
    fig.suptitle(
        'QALY per cycle for frail and fit patients under different policies',
        fontsize=24, fontweight='bold'
    )

    y_min, y_max = 0.005, 0.15

    for i, (ax, title) in enumerate(zip(axes, titles)):
        states_in_group = groups[title]
        ax.set_title(f"{title}", fontsize=18)

        allowed = allowed_policies_by_group.get(title, set(policy_colors.keys()))

        max_years = 0.0

        for pname, color in policy_colors.items():
            if pname not in allowed:
                continue

            m, s = _aggregate_group(trajectory_results, states_in_group, pname)
            if m is None:
                continue

            visit_idx = np.arange(len(m))   # quarters
            x_years = visit_idx * 0.25      # convert to years
            if len(x_years) > 0:
                max_years = max(max_years, x_years[-1])

            label = policy_labels.get(pname, pname.replace('_', ' '))

            ax.plot(
                x_years, m,
                label=label,
                linewidth=3.0,
                color=color,
                alpha=0.9
            )
            ax.fill_between(x_years, m - s, m + s, alpha=0.18, color=color)

        # -------- ΕΔΩ κλειδώνουμε τον άξονα σε 0–10 και βάζουμε 0,1,2,... ----------
        if max_years == 0:
            max_years = 5   # fallback αν για κάποιο λόγο δεν μπήκαν δεδομένα

        # αν θες ακριβώς 10 χρόνια:
        ax.set_xlim(0, 4)

        ax.xaxis.set_major_locator(mticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

        # -------------------------------------------------------------------- #

        ax.set_xlabel('Time (years)', fontsize=16)
        if i % 2 == 0:
            ax.set_ylabel('QALY per 3-month cycle', fontsize=16)
        else:
            ax.set_ylabel('')

        ax.tick_params(axis='both', which='major', labelsize=14, labelbottom=True)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13, ncol=2)

    # Σβήσε τα άδεια panels
    for j in range(len(titles), len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')
    return fig
