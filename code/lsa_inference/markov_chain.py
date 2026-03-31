"""Markov chain generation and simulation."""

import numpy as np
from numpy.linalg import eig


def generate_transition_matrix(n_states, rng):
    """Generate a random irreducible aperiodic transition matrix.

    Returns:
        P: (n_states, n_states) row-stochastic transition matrix.
        pi: (n_states,) stationary distribution.
    """
    while True:
        M = rng.uniform(0, 1, (n_states, n_states))
        P = M / M.sum(axis=1, keepdims=True)
        vals, vecs = eig(P.T)
        idx = np.argmin(np.abs(vals - 1.0))
        pi = np.real(vecs[:, idx])
        pi /= pi.sum()
        if np.all(pi > 0):
            return P, pi


def simulate_chains_batch(P, pi, T, n_traj, rng):
    """Simulate n_traj independent Markov chain trajectories simultaneously.

    Returns:
        trajs: (n_traj, T) int array of state indices.
    """
    n_states = len(pi)
    cum_P = np.cumsum(P, axis=1)
    cum_pi = np.cumsum(pi)

    trajs = np.empty((n_traj, T), dtype=np.int32)

    u = rng.uniform(size=n_traj)
    trajs[:, 0] = np.searchsorted(cum_pi, u)

    for t in range(1, T):
        u = rng.uniform(size=n_traj)
        prev = trajs[:, t - 1]
        trajs[:, t] = (u[:, None] < cum_P[prev]).argmax(axis=1)

    return trajs
