"""Fit the MDP, simulate dose policies and export analysis results."""
from pathlib import Path
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mdp import SimpleQOLMDP
from .trajectories import create_alternative_policies, analyze_trajectories_by_initial_state
from .plotting import _aggregate_group, get_baseline_groups, visualize_trajectory_by_baseline_groups


def run_analysis(csv_path, output_dir, n_simulations=500, n_steps=21):
    if n_simulations <= 0 or n_steps <= 0:
        raise ValueError('Simulation and step counts must be positive.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mdp = SimpleQOLMDP(str(csv_path), annual_discount=0.97)
    mdp.estimate_qol_rewards(lookahead_steps=3, reward_mode='qaly', visit_len_years=0.25)
    mdp.estimate_transition_probabilities(alpha=0.5)
    values, policy = mdp.value_iteration()
    alternatives = create_alternative_policies(mdp, optimal_policy=policy)
    results = analyze_trajectories_by_initial_state(
        mdp, policy, alternatives, n_simulations=n_simulations, n_steps=n_steps)
    fig, _ = visualize_trajectory_by_baseline_groups(mdp, policy, results, savefig=output/'figure_baseline_groups.png')
    plt.close(fig)

    np.savez_compressed(output/'mdp_arrays.npz', R=mdp.R, P=mdp.P, values=values, optimal_policy=policy)
    pd.DataFrame([
        {'state_id':i, **state, 'value':float(values[i]), 'optimal_action':int(policy[i]),
         'optimal_action_name':mdp.actions[int(policy[i])]}
        for i,state in mdp.states.items()
    ]).to_csv(output/'optimal_policy.csv', index=False)
    pd.DataFrame([
        {'state_id':s, 'state':mdp.states[s]['name'], 'action':a,
         'action_name':mdp.actions[a], 'reward_qaly_per_cycle':mdp.R[s,a]}
        for s in range(mdp.n_states) for a in range(mdp.n_actions)
    ]).to_csv(output/'rewards.csv',index=False)
    pd.DataFrame([
        {'state_id':s, 'action':a, 'next_state_id':s2, 'probability':mdp.P[s,a,s2]}
        for s in range(mdp.n_states) for a in range(mdp.n_actions) for s2 in range(mdp.n_states)
    ]).to_csv(output/'transitions.csv',index=False)

    rows=[]
    for group,names in get_baseline_groups(mdp,results).items():
        for name in ['optimal','always_reduced','always_full']:
            mean,sd=_aggregate_group(results,names,name)
            for t,(mu,sigma) in enumerate(zip(mean,sd)):
                rows.append({'group':group,'policy':name,'step_index':t,'time_years':t*.25,
                             'mean_qaly_per_cycle':mu,'sd_qaly_per_cycle':sigma})
    pd.DataFrame(rows).to_csv(output/'figure_curves.csv',index=False)
    summary=[]
    for policy_name in alternatives:
        finals=np.array([res[policy_name]['mean_cumulative_qol'][-1] for res in results.values()])
        summary.append({'policy':policy_name,'mean_final_cumulative_qaly':float(finals.mean()),
                        'sd_across_start_state_means':float(finals.std()),
                        'min_start_state_mean':float(finals.min()),'max_start_state_mean':float(finals.max())})
    pd.DataFrame(summary).to_csv(output/'policy_summary.csv',index=False)
    metadata={
        'scope':'Eight-state MDP, three-policy simulation and baseline-group QALY figure',
        'clinical_test':{'0':'favorable (legacy Neg)','1':'unfavorable (legacy Pos)'},
        'n_simulations_per_start_state_policy':n_simulations,'n_steps':n_steps,
        'trajectory_seed':123,'random_policy_seed':42,'annual_discount':.97,
        'quarterly_discount':mdp.gamma,'lookahead_steps':3,'smoothing_alpha':.5,
        'cycle_years':.25,'displayed_x_limits_months':[0,(n_steps-1)*3],
        'number_of_reward_cycles':n_steps,'reward_horizon_years':n_steps*.25,
        'last_plotted_point_time_years':(n_steps-1)*.25,
        'band':'plus/minus one pooled standard deviation; not a confidence interval',
        'weighting':'equal weight per selected starting state, not empirical cohort frequencies',
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'matplotlib':matplotlib.__version__,
        'limitations':['Missing switching actions use assumed transitions and state-based reward fallback.',
                       'No absorbing death state in this MDP.',
                       'See IMPLEMENTATION_NOTES.md before scientific interpretation.']}
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return mdp, policy, results
