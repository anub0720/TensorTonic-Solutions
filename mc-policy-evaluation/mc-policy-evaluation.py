import numpy as np

def mc_policy_evaluation(episodes: list, gamma: float, n_states: int) -> np.ndarray:
    returns = [[] for _ in range(n_states)]

    for episode in episodes:
        G = 0
        episode_returns = [0.0] * len(episode)

        for t in range(len(episode) - 1, -1, -1):
            _, reward = episode[t]
            G = reward + gamma * G
            episode_returns[t] = G

        visited = set()

        for t, (state, _) in enumerate(episode):
            if state not in visited:
                returns[state].append(episode_returns[t])
                visited.add(state)

    values = np.zeros(n_states)

    for state in range(n_states):
        if returns[state]:
            values[state] = np.mean(returns[state])

    return np.round(values, 4)