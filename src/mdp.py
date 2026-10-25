"""Original eight-state MDP from notebook In [12]; mathematics unchanged."""
import numpy as np
import pandas as pd
from collections import defaultdict

class SimpleQOLMDP:
    """
    QOL-based MDP με 8 states: (frailty ∈{0,1}, clinical_test ∈{0,1}, dose ∈{0,1})
    Actions: 0=reduced_dose, 1=full_dose
    """

    # ---------- CHUNK 1: init / setup ----------
    def __init__(self, csv_file='Fixed_patient_data.csv', annual_discount=0.97):
        print("="*80); print("SIMPLE QOL-BASED MDP — unified class"); print("="*80)
        self.df = pd.read_csv(csv_file)
        print(f"Loaded: {len(self.df)} rows, {self.df['patient_id'].nunique()} patients")

        must = ['patient_id','visit','frailty','clinical_test','full_dose']
        missing = [c for c in must if c not in self.df.columns]
        assert not missing, f"Missing columns: {missing}"
        for c in must: self.df[c] = self.df[c].astype(int)

        self.gamma = float(annual_discount**0.25)   # quarterly discount

        print(f"Quarterly gamma = {self.gamma:.5f}")



        self.setup_states_and_actions()

    def setup_states_and_actions(self):
        self.states = {}
        idx = 0
        for frailty in [0,1]:
            for test in [0,1]:
                for dose in [0,1]:
                    self.states[idx] = {
                        'frailty': frailty,
                        'clinical_test': test,
                        'dose': dose,
                        'name': f"{'Fit' if frailty==0 else 'Frail'}_"
                                f"{'Neg' if test==0 else 'Pos'}_"
                                f"{'Red' if dose==0 else 'Full'}"
                    }
                    idx += 1
        self.n_states = len(self.states)
        self.actions = {0:'reduced_dose', 1:'full_dose'}
        self.n_actions = len(self.actions)
        self.state_mapping = {(s['frailty'], s['clinical_test'], s['dose']): i
                              for i,s in self.states.items()}
        print("States:")
        for i,s in self.states.items():
            print(f"  {i}: {s['name']} (frailty={s['frailty']}, test={s['clinical_test']}, dose={s['dose']})")

    def get_state_index(self, frailty, clinical_test, dose):
        return self.state_mapping.get((int(frailty), int(clinical_test), int(dose)), None)

    def analyze_data_coverage(self):
        print("\n" + "="*60); print("DATA COVERAGE"); print("="*60)
        has_qol = 'QOL' in self.df.columns
        for i, st in self.states.items():
            m = (
                (self.df['frailty']==st['frailty']) &
                (self.df['clinical_test']==st['clinical_test']) &
                (self.df['full_dose']==st['dose'])
            )
            n = int(m.sum())
            n_qol = int(self.df.loc[m,'QOL'].notna().sum()) if has_qol else 0
            print(f"{st['name']:<15}: {n:4d} rows, {n_qol:4d} with QOL")

    # ---------- CHUNK 2: rewards ----------
    def estimate_qol_rewards(self, lookahead_steps=3, reward_mode="qaly", visit_len_years=0.25):
        """
        Φτιάχνει R[s,a] από δεδομένα.
        - transition-based με lookahead έως `lookahead_steps` visits
        - direct state QOL ως fallback
        - defaults αν λείπουν όλα
        reward_mode: 'qol' ή 'qaly' (QOL * visit_len_years)
        """
        print("\n" + "="*60); print("QOL-BASED REWARD ESTIMATION"); print("="*60)

        def _next_measured_qol(traj_df, start_idx, max_lookahead):
            n = len(traj_df)
            for k in range(1, max_lookahead+1):
                j = start_idx + k
                if j < n:
                    q = traj_df.iloc[j]['QOL']
                    if pd.notna(q):
                        return float(q), k
            return None, None

        # Method 1: direct QOL per state
        print("\nDirect QOL by state:")
        qol_by_state = {}
        for s in range(self.n_states):
            st = self.states[s]
            mask = (
                (self.df['frailty']==st['frailty']) &
                (self.df['clinical_test']==st['clinical_test']) &
                (self.df['full_dose']==st['dose']) &
                (self.df['QOL'].notna())
            )
            vals = self.df.loc[mask, 'QOL'].astype(float).values
            if vals.size:
                qol_by_state[s] = {
                    'mean': float(vals.mean()),
                    'std':  float(vals.std(ddof=0)),
                    'count': int(vals.size),
                }
                print(f"  {st['name']:<15}: {qol_by_state[s]['mean']:.3f} ± {qol_by_state[s]['std']:.3f} (n={qol_by_state[s]['count']})")
            else:
                print(f"  {st['name']:<15}: No QOL data")

        # Method 2: transition-based with look-ahead
        print("\nQOL by (state, action) with look-ahead:")
        qol_by_sa = defaultdict(list)
        for pid in self.df['patient_id'].unique():
            traj = (self.df[self.df['patient_id']==pid]
                    .sort_values('visit').reset_index(drop=True))
            for i in range(len(traj)-1):
                cur, nxt = traj.iloc[i], traj.iloc[i+1]
                s = self.get_state_index(cur['frailty'], cur['clinical_test'], cur['full_dose'])
                if s is None: 
                    continue
                a = int(nxt['full_dose'])
                q = nxt['QOL'] if pd.notna(nxt['QOL']) else None
                if q is None:
                    q, _ = _next_measured_qol(traj, i, lookahead_steps)
                if q is not None:
                    reward = float(q) if reward_mode=="qol" else float(q)*float(visit_len_years)
                    qol_by_sa[(s,a)].append(reward)

        for (s,a), L in qol_by_sa.items():
            if L:
                m, sd, n = float(np.mean(L)), float(np.std(L, ddof=0)), len(L)
                unit = "QALY/visit" if reward_mode=="qaly" else "QOL"
                print(f"  {self.states[s]['name']:<15} → {self.actions[a]:<12}: {m:.3f} ± {sd:.3f} ({unit}, n={n})")

        # Compose R
        self.R = np.zeros((self.n_states, self.n_actions), dtype=float)
        print("\nFinal Reward Matrix ({})".format("QALY/visit" if reward_mode=="qaly" else "QOL"))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if (s,a) in qol_by_sa and len(qol_by_sa[(s,a)])>0:
                    avg, src = float(np.mean(qol_by_sa[(s,a)])), "transition"
                elif s in qol_by_state:
                    base = qol_by_state[s]['mean']
                    avg, src = (base if reward_mode=="qol" else base*visit_len_years), "direct"
                else:
                    st = self.states[s]
                    if st['frailty']==0: base = 0.75 if a==1 else 0.65
                    else:                base = 0.45 if a==0 else 0.35
                    avg, src = (base if reward_mode=="qol" else base*visit_len_years), "default"
                self.R[s,a] = avg
                print(f"  {self.states[s]['name']:<15} + {self.actions[a]:<12}: {avg:.3f} ({src})")

        print("\nReward range:", f"{self.R.min():.3f} .. {self.R.max():.3f}")
        return None, None  # κρατάμε απλό API

    # ---------- CHUNK 3: transitions ----------
    def estimate_transition_probabilities(self, alpha=0.5):
        """
        P[s,a,s'] από διαδοχικές επισκέψεις (state_t, action_{t+1}, state_{t+1}),
        με Dirichlet smoothing alpha.
        """
        print("\n" + "="*60); print("TRANSITION PROBABILITY ESTIMATION"); print("="*60)
        has_death = 'death_tox' in self.df.columns
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        total = 0

        for pid in self.df['patient_id'].unique():
            traj = (self.df[self.df['patient_id']==pid]
                    .sort_values('visit').reset_index(drop=True))
            n = len(traj)
            for i in range(n-1):
                cur, nxt = traj.iloc[i], traj.iloc[i+1]
                if has_death and int(cur.get('death_tox',0))==1:
                    break
                s  = self.get_state_index(cur['frailty'], cur['clinical_test'], cur['full_dose'])
                a  = int(nxt['full_dose'])
                s2 = self.get_state_index(nxt['frailty'], nxt['clinical_test'], nxt['full_dose'])
                if s is None or s2 is None: 
                    continue
                counts[s][a][s2] += 1
                total += 1

        print(f"Total transitions observed: {total}")
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=float)

        for s in range(self.n_states):
            for a in range(self.n_actions):
                vec = np.array([counts[s][a][s2] for s2 in range(self.n_states)], dtype=float)
                if vec.sum()>0:
                    probs = (vec + alpha) / (vec.sum() + alpha*self.n_states)
                else:
                    probs = np.full(self.n_states, 0.3/(self.n_states-1)); probs[s]=0.7
                self.P[s,a,:] = probs

        print("\nTransition Summary (top 3 when n>5):")
        for s in range(self.n_states):
            for a in range(self.n_actions):
                n_sa = int(sum(counts[s][a].values()))
                if n_sa > 5:
                    top = sorted([(s2, self.P[s,a,s2]) for s2 in range(self.n_states)],
                                 key=lambda x:x[1], reverse=True)[:3]
                    print(f"  {self.states[s]['name']:<15} + {self.actions[a]:<12} (n={n_sa}):")
                    for s2,p in top:
                        if p>0.10:
                            print(f"    → {self.states[s2]['name']:<15}: {p:.3f}")

    # ---------- CHUNK 4: solvers ----------
    def value_iteration(self, gamma=None, epsilon=1e-3, max_iterations=5000):
        gamma = self.gamma if gamma is None else float(gamma)
        print("\n" + "="*60); print(f"VALUE ITERATION (γ={gamma:.5f})"); print("="*60)
        V = np.zeros(self.n_states, dtype=float)
        policy = np.zeros(self.n_states, dtype=int)
        for it in range(max_iterations):
            V_old = V.copy()
            for s in range(self.n_states):
                Qa = np.array([self.R[s,a] + gamma*np.dot(self.P[s,a,:], V_old)
                               for a in range(self.n_actions)], dtype=float)
                V[s] = Qa.max()
                policy[s] = int(Qa.argmax())
            delta = float(np.max(np.abs(V - V_old)))
            if it % 100 == 0: print(f"  Iter {it:4d}: max Δ = {delta:.6f}")
            if delta < epsilon:
                print(f"Converged at iter {it+1} (max Δ = {delta:.6f})"); break
        else:
            print(f"Reached max iterations ({max_iterations}).")
        return V, policy

    def policy_evaluation(self, policy, gamma=None, epsilon=1e-3, max_iterations=5000):
        gamma = self.gamma if gamma is None else float(gamma)
        V = np.zeros(self.n_states, dtype=float)
        for it in range(max_iterations):
            delta = 0.0
            for s in range(self.n_states):   # Gauss–Seidel-like
                a = int(policy[s])
                Vs_old = V[s]
                V[s] = self.R[s,a] + gamma * np.dot(self.P[s,a,:], V)
                delta = max(delta, abs(V[s]-Vs_old))
            if it % 200 == 0: print(f"  PolicyEval {it:4d}: max Δ = {delta:.6f}")
            if delta < epsilon:
                print(f"  PolicyEval converged at {it+1} (max Δ = {delta:.6f})"); break
        else:
            print(f"  PolicyEval reached max iterations ({max_iterations}).")
        return V

    def policy_iteration(self, gamma=None, max_iterations=100, eval_epsilon=1e-3, eval_max_iter=5000):
        gamma = self.gamma if gamma is None else float(gamma)
        print("\n" + "="*60); print(f"POLICY ITERATION (γ={gamma:.5f})"); print("="*60)
        policy = np.argmax(self.R, axis=1).astype(int)  # sensible start
        for it in range(max_iterations):
            V = self.policy_evaluation(policy, gamma=gamma, epsilon=eval_epsilon, max_iterations=eval_max_iter)
            stable = True
            for s in range(self.n_states):
                Qa = np.array([self.R[s,a] + gamma*np.dot(self.P[s,a,:], V) for a in range(self.n_actions)])
                best_a = int(Qa.argmax())
                if best_a != policy[s]:
                    policy[s] = best_a
                    stable = False
            if stable:
                print(f"Policy converged after {it+1} improvement iterations."); break
        else:
            print(f"Reached maximum improvement iterations ({max_iterations}).")
        return V, policy