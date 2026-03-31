"""Vectorized LSA engine: constant step, diminishing step, RR extrapolation.

Processes all n_traj trajectories simultaneously via numpy broadcasting.

Shapes:
    thetas:  (n_traj, d)
    A_arr:   (n_states, d, d)
    b_arr:   (n_states, d)
    trajs:   (n_traj, T)
"""

import numpy as np

# Threshold for detecting divergence.  Values above this are replaced with NaN.
_DIVERG_THRESH = 1e6


def prepare_arrays(A_list, b_list):
    """Stack A_list and b_list into contiguous numpy arrays."""
    return np.array(A_list), np.array(b_list)


def _clamp_diverged(thetas):
    """Replace diverged trajectories (inf/nan/large) with NaN in-place."""
    bad = ~np.isfinite(thetas) | (np.abs(thetas) > _DIVERG_THRESH)
    if np.any(bad):
        # Mark entire trajectory (all coords) as NaN if any coord diverged
        bad_rows = np.any(bad, axis=1)
        thetas[bad_rows] = np.nan


def _clamp_batch_means(batch_means):
    """Replace non-finite or large batch means with NaN."""
    batch_means = np.where(
        np.isfinite(batch_means) & (np.abs(batch_means) < _DIVERG_THRESH),
        batch_means, np.nan
    )
    return batch_means


# ---------------------------------------------------------------------------
# Constant-stepsize LSA  (Huo et al. 2023)
# ---------------------------------------------------------------------------

def run_lsa_const(A_arr, b_arr, trajs, alpha, K, burn_in=100, n0=0):
    """Run constant-stepsize LSA for all trajectories simultaneously.

    Returns:
        batch_means: (n_traj, K, d) array of batch means.
        n: Batch size.
    """
    n_traj, T = trajs.shape
    d = b_arr.shape[1]
    usable_T = T - burn_in
    n = usable_T // K

    thetas = np.zeros((n_traj, d))
    batch_sums = np.zeros((n_traj, K, d))

    current_batch = 0
    batch_count = 0

    for t in range(T):
        x_t = trajs[:, t]
        A_t = A_arr[x_t]
        b_t = b_arr[x_t]
        thetas += alpha * (np.einsum('nij,nj->ni', A_t, thetas) + b_t)

        # Early divergence detection every 100 steps
        if t % 100 == 99:
            _clamp_diverged(thetas)

        if t < burn_in:
            continue
        if current_batch >= K:
            break

        if batch_count >= n0:
            batch_sums[:, current_batch, :] += thetas

        batch_count += 1
        if batch_count == n:
            current_batch += 1
            batch_count = 0

    effective = n - n0
    batch_means = batch_sums / effective if effective > 0 else batch_sums
    return _clamp_batch_means(batch_means), n


# ---------------------------------------------------------------------------
# Diminishing-stepsize LSA with CLTZ20 batching  (Huo et al. 2023 baseline)
# ---------------------------------------------------------------------------

def run_lsa_diminishing(A_arr, b_arr, trajs, alpha0, alpha_exp=0.5, K=50):
    """Run diminishing-stepsize LSA with non-uniform CLTZ20 batching.

    Returns:
        batch_means: (n_traj, K, d) array.
        n_eff: Effective average batch size.
    """
    n_traj, T = trajs.shape
    d = b_arr.shape[1]

    r = T ** (1 - alpha_exp) / (K + 1)
    endpoints = [0]
    for k in range(1, K + 1):
        e_k = int(((k + 1) * r) ** (1 / (1 - alpha_exp)))
        e_k = min(e_k, T)
        endpoints.append(e_k)

    thetas = np.zeros((n_traj, d))
    batch_sums = np.zeros((n_traj, K, d))
    batch_counts = np.zeros(K, dtype=np.int64)

    batch_for_t = np.full(T, -1, dtype=np.int32)
    for k in range(K):
        batch_for_t[endpoints[k]:endpoints[k + 1]] = k
        batch_counts[k] = endpoints[k + 1] - endpoints[k]

    for t in range(T):
        x_t = trajs[:, t]
        A_t = A_arr[x_t]
        b_t = b_arr[x_t]
        step = alpha0 / (t + 1) ** alpha_exp
        thetas += step * (np.einsum('nij,nj->ni', A_t, thetas) + b_t)

        if t % 100 == 99:
            _clamp_diverged(thetas)

        k = batch_for_t[t]
        if k >= 0:
            batch_sums[:, k, :] += thetas

    batch_means = np.zeros((n_traj, K, d))
    total_used = 0
    for k in range(K):
        if batch_counts[k] > 0:
            batch_means[:, k, :] = batch_sums[:, k, :] / batch_counts[k]
            total_used += batch_counts[k]

    n_eff = total_used // K if K > 0 else T
    return _clamp_batch_means(batch_means), n_eff


# ---------------------------------------------------------------------------
# Diminishing-stepsize LSA with Polyak-Ruppert averaging  (Samsonov et al.)
# Returns all iterates for OBM/bootstrap use.
# ---------------------------------------------------------------------------

def run_lsa_polyak_ruppert(A_arr, b_arr, trajs, c0, k0, gamma=0.75):
    """Run diminishing-stepsize LSA, return all iterates + PR average.

    Step sizes: alpha_k = c0 / (k + k0)^gamma

    Returns:
        all_thetas: (n_traj, T, d) all iterates.
        theta_bar: (n_traj, d) Polyak-Ruppert average.
    """
    n_traj, T = trajs.shape
    d = b_arr.shape[1]

    all_thetas = np.empty((n_traj, T, d))
    thetas = np.zeros((n_traj, d))

    for t in range(T):
        x_t = trajs[:, t]
        A_t = A_arr[x_t]
        b_t = b_arr[x_t]
        alpha_t = c0 / (t + k0) ** gamma
        thetas += alpha_t * (np.einsum('nij,nj->ni', A_t, thetas) + b_t)

        if t % 100 == 99:
            _clamp_diverged(thetas)

        all_thetas[:, t, :] = thetas

    theta_bar = np.nanmean(all_thetas, axis=1)
    return all_thetas, theta_bar


# ---------------------------------------------------------------------------
# Richardson-Romberg coefficients & extrapolation  (Huo et al. 2023)
# ---------------------------------------------------------------------------

def rr_coefficients(alphas):
    """Compute RR extrapolation weights via Lagrange interpolation."""
    alphas = np.asarray(alphas, dtype=float)
    M = len(alphas)
    h = np.ones(M)
    for m in range(M):
        for l in range(M):
            if l != m:
                h[m] *= alphas[l] / (alphas[l] - alphas[m])
    return h


def run_rr(A_arr, b_arr, trajs, alphas, K, burn_in=100, n0=0):
    """Run RR extrapolation: same trajectories, multiple stepsizes.

    Returns:
        rr_batch_means: (n_traj, K, d) extrapolated batch means.
        n: Batch size.
    """
    h = rr_coefficients(alphas)
    all_bm = []
    n = None
    for m, alpha in enumerate(alphas):
        bm, n_m = run_lsa_const(A_arr, b_arr, trajs, alpha, K, burn_in, n0)
        all_bm.append(bm)
        if n is None:
            n = n_m

    rr_batch_means = sum(h[m] * all_bm[m] for m in range(len(alphas)))
    return rr_batch_means, n
