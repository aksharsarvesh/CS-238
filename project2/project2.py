import pandas as pd
import numpy as np
from collections import defaultdict

# Hardcoding the state counts because not every state seen in all data
STATE_COUNTS = {
    "small": 100,
    "medium": 50000,
    "large": 302020,
}

def parseData(case, root="data"):
    df = pd.read_csv(f"{root}/{case}.csv")
    states  = sorted(set(df["s"]).union(df["sp"]))
    actions = sorted(df["a"].unique())
    return df, states, actions


def offline_q_learning(df, actions, gamma, alpha=0.1, epochs=500, init_q=0.0, seed=0):
    rng = np.random.default_rng(seed)
    Q = defaultdict(lambda: defaultdict(lambda: init_q))
    transitions = list(zip(df["s"].tolist(), df["a"].tolist(), df["r"].tolist(), df["sp"].tolist()))
    for _ in range(epochs):
        rng.shuffle(transitions)
        for s, a, r, sp in transitions:
            max_next = max((Q[sp][ap] for ap in actions), default=0.0)
            target = r + gamma * max_next
            Q[s][a] += alpha * (target - Q[s][a])
    return Q

def greedy_policy(Q, actions):
    pi = {}
    for s, Qa in Q.items():
        pi[s] = max(actions, key=lambda a: Qa[a])
    return pi

def write_policy_file(pi, num_states, filename, actions, seed=0):
    rng = np.random.default_rng(seed)
    with open(filename, "w") as f:
        for s in range(1, num_states + 1):
            action = pi.get(s)
            if action is None:
                action = int(rng.choice(actions))
            f.write(f"{action}\n")

def QLearning(case, root="data", alpha=0.1, epochs=500):
    gamma_map = {"small": 0.95, "medium": 1.0, "large": 0.95}
    gamma = gamma_map[case]

    df, states, actions = parseData(case, root=root)
    Q = offline_q_learning(df, actions, gamma=gamma, alpha=alpha, epochs=epochs)
    pi = greedy_policy(Q, actions)

    num_states = STATE_COUNTS[case]
    out_path = f"output/{case}.policy"
    seed = {"small": 1, "medium": 2, "large": 3}[case]
    write_policy_file(pi, num_states, out_path, actions, seed=seed)
    print(f"Saved policy to {out_path}")

if __name__ == "__main__":
    cases = ["small", "medium", "large"]
    for case in cases:
        QLearning(case, root="data")
