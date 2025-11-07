import pandas as pd
import numpy as np
from collections import defaultdict

# Fixed total state counts per case
STATE_COUNTS = {
    "small": 100,
    "medium": 50000,
    "large": 302020,
}

# ----------------------------------------------------------
# 1. Parse dataset
# ----------------------------------------------------------
def parseData(case, root="data"):
    """
    Read a 's,a,r,sp' CSV and return (df, states, actions).
    """
    df = pd.read_csv(f"{root}/{case}.csv")
    assert set(df.columns) >= {"s", "a", "r", "sp"}, "CSV must have columns: s,a,r,sp"
    states  = sorted(set(df["s"]).union(df["sp"]))
    actions = sorted(df["a"].unique())  # discrete, 1-indexed
    return df, states, actions

# ----------------------------------------------------------
# 2. Offline Q-Learning
# ----------------------------------------------------------
def offline_q_learning(df, actions, gamma=0.99, alpha=0.1, epochs=50, init_q=0.0, seed=0):
    """
    Tabular, batch Q-learning over a fixed dataset.
    Repeats TD(0) updates across the dataset for 'epochs' passes.
    """
    rng = np.random.default_rng(seed)
    Q = defaultdict(lambda: defaultdict(lambda: init_q))

    # Materialize transitions for speed
    transitions = list(zip(df["s"].tolist(), df["a"].tolist(), df["r"].tolist(), df["sp"].tolist()))

    for _ in range(epochs):
        rng.shuffle(transitions)
        for s, a, r, sp in transitions:
            # Bootstrap target: r + gamma * max_a' Q(sp, a')
            max_next = max((Q[sp][ap] for ap in actions), default=0.0)
            target = r + gamma * max_next
            Q[s][a] += alpha * (target - Q[s][a])
    return Q

# ----------------------------------------------------------
# 3. Greedy policy extraction
# ----------------------------------------------------------
def greedy_policy(Q, actions):
    """
    π(s) = argmax_a Q(s,a) for states present in Q.
    """
    pi = {}
    for s, Qa in Q.items():
        pi[s] = max(actions, key=lambda a: Qa[a])
    return pi

# ----------------------------------------------------------
# 4. Simple preview helper (optional)
# ----------------------------------------------------------
def preview_policy(Q, actions, limit=20):
    rows = []
    for i, s in enumerate(sorted(Q.keys())):
        if i >= limit:
            break
        best_a = max(actions, key=lambda a: Q[s][a])
        row = {"state": s, "pi(s)": best_a}
        for a in actions:
            row[f"Q(a={a})"] = round(Q[s][a], 3)
        rows.append(row)
    return pd.DataFrame(rows)

# ----------------------------------------------------------
# 5. Write the policy vector file (one line per state ID)
# ----------------------------------------------------------
def write_policy_vector(pi, num_states, filename, actions, seed=0):
    """
    Write a policy file with |states| lines.
    Line i (1-indexed) contains the action number to take at state i.
    For states not present in 'pi', write a random valid action (1-indexed).
    """
    rng = np.random.default_rng(seed)
    with open(filename, "w") as f:
        for s in range(1, num_states + 1):
            action = pi.get(s, None)
            if action is None:
                action = int(rng.choice(actions))  # random valid action, never 0
            f.write(f"{action}\n")

# ----------------------------------------------------------
# 6. Combined runner
# ----------------------------------------------------------
def QLearning(case, root="data", gamma=0.99, alpha=0.1, epochs=500):
    df, states, actions = parseData(case, root=root)
    Q = offline_q_learning(df, actions, gamma=gamma, alpha=alpha, epochs=epochs)
    pi = greedy_policy(Q, actions)

    # Optional: preview learned states only (does not affect output)
    print(f"\n=== {case.upper()} ===")
    prev = preview_policy(Q, actions, limit=15)
    print(prev.to_string(index=False))

    # Use known total number of states for this case
    num_states = STATE_COUNTS[case]

    # Save compact policy vector with strict line = state mapping
    out_path = f"output/{case}.policy"
    # Seed can be fixed per case for reproducibility if desired:
    seed = {"small": 1, "medium": 2, "large": 3}[case]
    write_policy_vector(pi, num_states, out_path, actions, seed=seed)
    print(f"Saved linewise policy to {out_path}")

# ----------------------------------------------------------
# 7. Main
# ----------------------------------------------------------
if __name__ == "__main__":
    cases = ["small", "medium", "large"]
    for case in cases:
        QLearning(case, root="data")
