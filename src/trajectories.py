"""
QOL Trajectory Analysis Extension for MDP (quarterly-consistent)
- Steps = visits (default 40 = 10 years)
- Rewards assumed QALY/visit αν το R έχει οριστεί έτσι
"""

import numpy as np
from collections import defaultdict
import pandas as pd

def simulate_trajectory(mdp, initial_state, policy, n_steps=40, gamma=None, rng=None):
    """
    Simulate a single patient trajectory following a given policy.
    Each step = one visit (quarter).
    Rewards are taken from mdp.R[s,a] (QOL or QALY/visit per your setup).
    """
    if gamma is None:
        gamma = mdp.gamma
    if rng is None:
        rng = np.random.default_rng()

    traj = {
        'states': [initial_state],
        'actions': [],
        'rewards': [],
        'qol_values': [],
        'cumulative_qol': [],
        'discounted_qol': []
    }

    s = initial_state
    cum = 0.0
    disc = 0.0

    for t in range(n_steps):
        a = int(policy[s])
        r = float(mdp.R[s, a])

        traj['actions'].append(a)
        traj['rewards'].append(r)
        traj['qol_values'].append(r)

        cum += r
        disc += r * (gamma ** t)
        traj['cumulative_qol'].append(cum)
        traj['discounted_qol'].append(disc)

        p_next = mdp.P[s, a, :]
        s = int(rng.choice(mdp.n_states, p=p_next))
        traj['states'].append(s)

    return traj

def simulate_multiple_trajectories(mdp, initial_state, policy, n_simulations=500, n_steps=40, gamma=None, seed=123):
    if gamma is None:
        gamma = mdp.gamma
    rng = np.random.default_rng(seed)

    all_traj = []
    for _ in range(n_simulations):
        # διαφορετικό rng state ανά simulation για ανεξαρτησία
        tr = simulate_trajectory(mdp, initial_state, policy, n_steps=n_steps, gamma=gamma, rng=rng)
        all_traj.append(tr)

    agg = {
        'mean_qol_over_time': [],
        'std_qol_over_time': [],
        'mean_cumulative_qol': [],
        'std_cumulative_qol': [],
        'mean_discounted_qol': [],
        'std_discounted_qol': [],
        'state_frequencies': defaultdict(list),
        'action_frequencies': defaultdict(list)
    }

    for t in range(n_steps):
        q_t = [tr['qol_values'][t] for tr in all_traj]
        agg['mean_qol_over_time'].append(float(np.mean(q_t)))
        agg['std_qol_over_time'].append(float(np.std(q_t)))

        c_t = [tr['cumulative_qol'][t] for tr in all_traj]
        agg['mean_cumulative_qol'].append(float(np.mean(c_t)))
        agg['std_cumulative_qol'].append(float(np.std(c_t)))

        d_t = [tr['discounted_qol'][t] for tr in all_traj]
        agg['mean_discounted_qol'].append(float(np.mean(d_t)))
        agg['std_discounted_qol'].append(float(np.std(d_t)))

        # states at step t
        states_t = [tr['states'][t] for tr in all_traj]
        for s in range(mdp.n_states):
            agg['state_frequencies'][s].append(float(np.mean(np.array(states_t)==s)))

        # actions at step t
        actions_t = [tr['actions'][t] for tr in all_traj]
        for a in range(mdp.n_actions):
            agg['action_frequencies'][a].append(float(np.mean(np.array(actions_t)==a)))

    return agg

def create_alternative_policies(mdp, optimal_policy=None):
    """
    Returns a dict of policies for comparison.
    If optimal_policy is None, compute via value iteration.
    """
    policies = {}

    # 1) Always reduced
    policies['always_reduced'] = np.zeros(mdp.n_states, dtype=int)

    # 2) Always full
    policies['always_full'] = np.ones(mdp.n_states, dtype=int)

    # 3) Frailty-based (reduced if frail, full if fit)
    policies['frailty_based'] = np.array([
        0 if mdp.states[s]['frailty'] == 1 else 1
        for s in range(mdp.n_states)
    ], dtype=int)

    # 4) Test-based (full if positive, reduced if negative)
    policies['clinical_test_based'] = np.array([
        1 if mdp.states[s]['clinical_test'] == 1 else 0
        for s in range(mdp.n_states)
    ], dtype=int)

    # 5) Random (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    policies['random'] = rng.integers(0, 2, size=mdp.n_states, dtype=int)

    # 6) Inverse-optimal (choose the worse action wrt Q)
    if optimal_policy is None:
        V_opt, optimal_policy = mdp.value_iteration()  # uses mdp.gamma
    else:
        V_opt, _ = mdp.value_iteration()  # ενημερωμένα V για Q-margins

    gamma = mdp.gamma
    inv = np.zeros(mdp.n_states, dtype=int)
    for s in range(mdp.n_states):
        Q = np.array([mdp.R[s, a] + gamma * np.dot(mdp.P[s, a, :], V_opt) for a in range(mdp.n_actions)])
        inv[s] = int(np.argmin(Q))
    policies['inverse_optimal'] = inv

    # Include optimal too for convenience
    policies['optimal'] = np.array(optimal_policy, dtype=int)

    return policies

def analyze_trajectories_by_initial_state(mdp, optimal_policy, alternative_policies, n_simulations=500, n_steps=40):
    """
    Simulate for all initial states and all policies.
    Returns nested dict: results[state_name][policy_name] -> aggregated stats.
    """
    results = {}
    for s in range(mdp.n_states):
        state_name = mdp.states[s]['name']
        results[state_name] = {}

        # Optimal
        results[state_name]['optimal'] = simulate_multiple_trajectories(
            mdp, s, optimal_policy, n_simulations=n_simulations, n_steps=n_steps, gamma=mdp.gamma
        )
        # Alternatives
        for pname, pol in alternative_policies.items():
            results[state_name][pname] = simulate_multiple_trajectories(
                mdp, s, pol, n_simulations=n_simulations, n_steps=n_steps, gamma=mdp.gamma
            )
    return results

