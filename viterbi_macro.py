import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def viterbi_score(returns, macro_df, n_states=3):
    """
    Viterbi algorithm with macro‑driven transition probabilities.
    Returns the log‑likelihood of the most probable path.
    """
    if len(returns) < 20 or macro_df is None or len(macro_df) < 20:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 20:
        return 0.0
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    # Compute composite macro factor (first principal component)
    pca = PCA(n_components=1)
    macro_factor = pca.fit_transform(macro_scaled).flatten()
    macro_factor = (macro_factor - macro_factor.min()) / (macro_factor.max() - macro_factor.min() + 1e-8)
    # Discretise macro factor into n_states (for transition probabilities)
    macro_disc = np.digitize(macro_factor, np.linspace(0, 1, n_states+1)[1:-1])
    # Estimate state means and variances from returns
    # Use k-means on returns to initialise state parameters
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    states = kmeans.fit_predict(returns.reshape(-1, 1))
    # Estimate emission parameters (mean, std per state)
    means = np.zeros(n_states)
    stds = np.zeros(n_states)
    for s in range(n_states):
        idx = (states == s)
        if np.sum(idx) > 0:
            means[s] = np.mean(returns[idx])
            stds[s] = np.std(returns[idx]) + 1e-6
        else:
            means[s] = np.mean(returns)
            stds[s] = np.std(returns) + 1e-6
    # Estimate transition probabilities from macro state
    trans = np.zeros((n_states, n_states))
    for i in range(n_states):
        # Count transitions from macro state i to all states
        # Use logistic regression to predict next state from current macro
        # Simplified: use empirical transition counts from the data
        # We'll compute the frequency of transitions for each current macro state
        for t in range(len(macro_disc)-1):
            curr_macro = macro_disc[t]
            next_macro = macro_disc[t+1]
            if curr_macro < n_states and next_macro < n_states:
                trans[curr_macro, next_macro] += 1
    # Normalise rows
    for i in range(n_states):
        if trans[i].sum() > 0:
            trans[i] = trans[i] / trans[i].sum()
        else:
            trans[i] = np.ones(n_states) / n_states
    # Viterbi algorithm
    T = len(returns)
    # Initial probability: uniform
    init = np.ones(n_states) / n_states
    # Log probabilities
    log_init = np.log(init + 1e-12)
    log_trans = np.log(trans + 1e-12)
    # Emission log-likelihoods: log N(returns[t] | mean_s, std_s)
    log_emit = np.zeros((T, n_states))
    for t in range(T):
        for s in range(n_states):
            log_emit[t, s] = -0.5 * ((returns[t] - means[s]) / stds[s])**2 - 0.5 * np.log(2 * np.pi * stds[s]**2)
    # Viterbi forward pass
    delta = np.zeros((T, n_states))
    psi = np.zeros((T, n_states), dtype=int)
    delta[0] = log_init + log_emit[0]
    for t in range(1, T):
        for s in range(n_states):
            delta[t, s] = np.max(delta[t-1] + log_trans[:, s]) + log_emit[t, s]
            psi[t, s] = np.argmax(delta[t-1] + log_trans[:, s])
    # Backtrack to find most likely path
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    # Score = log-likelihood of the most probable path
    log_likelihood = np.max(delta[-1])
    return float(log_likelihood)
