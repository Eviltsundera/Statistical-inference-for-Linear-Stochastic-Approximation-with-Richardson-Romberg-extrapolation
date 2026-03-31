"""Inference procedures for LSA iterates.

Two families:
  1. Batch-mean covariance (Huo et al. 2023) — works with non-overlapping batch means.
  2. Overlapping Batch Mean / Multiplier Subsample Bootstrap (Samsonov et al. 2025)
     — works with the full iterate sequence and overlapping windows.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Batch-mean inference  (Huo et al. 2023)
# ---------------------------------------------------------------------------

def batch_mean_ci(batch_means, n, theta_star, n0=0, q=0.05, coord=0):
    """Compute L2 error, CI width, and coverage from non-overlapping batch means.

    Args:
        batch_means: (n_traj, K, d) array of batch means.
        n: Batch size (iterates per batch).
        theta_star: (d,) true solution.
        n0: Intra-batch discard.
        q: CI error level (0.05 for 95% CI).
        coord: Coordinate for CI/coverage.

    Returns:
        l2_errors: (n_traj,)
        ci_widths: (n_traj,)
        coverages: (n_traj,) binary (0/1).
    """
    z = stats.norm.ppf(1 - q / 2)
    n_traj, K, d = batch_means.shape

    theta_bars = np.nanmean(batch_means, axis=1)  # (n_traj, d)
    l2_errors = np.linalg.norm(theta_bars - theta_star, axis=1)

    diffs = batch_means[:, :, coord] - theta_bars[:, coord:coord + 1]
    var_coord = (n - n0) / K * np.nansum(diffs ** 2, axis=1)

    se = np.sqrt(var_coord / (K * (n - n0)))
    ci_widths = 2 * z * se

    lo = theta_bars[:, coord] - z * se
    hi = theta_bars[:, coord] + z * se
    coverages = ((lo <= theta_star[coord]) & (theta_star[coord] <= hi)).astype(float)

    has_nan = np.any(np.isnan(theta_bars), axis=1)
    l2_errors[has_nan] = np.nan
    ci_widths[has_nan] = np.nan
    coverages[has_nan] = 0.0

    return l2_errors, ci_widths, coverages


# ---------------------------------------------------------------------------
# 2. Overlapping Batch Mean (OBM) inference  (Samsonov et al. 2025)
# ---------------------------------------------------------------------------

def obm_variance(all_thetas, theta_bar, b_n, u):
    """Compute OBM variance estimator for projection direction u.

    sigma_hat^2(u) = b_n / (n - b_n + 1) * sum_t ((theta_{b_n,t} - theta_bar)^T u)^2

    where theta_{b_n,t} = (1/b_n) sum_{l=t}^{t+b_n-1} theta_l

    Args:
        all_thetas: (n_traj, T, d) all iterates.
        theta_bar: (n_traj, d) Polyak-Ruppert average.
        b_n: Block size (lag parameter).
        u: (d,) unit projection vector.

    Returns:
        sigma_hat_sq: (n_traj,) OBM variance estimates.
    """
    n_traj, T, d = all_thetas.shape
    n_blocks = T - b_n + 1

    # Compute cumulative sums for efficient block averages
    # cumsum shape: (n_traj, T+1, d)
    proj = np.einsum('ntd,d->nt', all_thetas, u)  # (n_traj, T)
    cumsum_proj = np.concatenate([np.zeros((n_traj, 1)), np.cumsum(proj, axis=1)], axis=1)

    # Block averages: theta_{b_n,t}^T u = (cumsum[t+b_n] - cumsum[t]) / b_n
    block_avgs = (cumsum_proj[:, b_n:] - cumsum_proj[:, :n_blocks]) / b_n  # (n_traj, n_blocks)

    # theta_bar^T u
    bar_proj = np.einsum('nd,d->n', theta_bar, u)  # (n_traj,)

    diffs = block_avgs - bar_proj[:, None]  # (n_traj, n_blocks)
    sigma_hat_sq = (b_n / n_blocks) * np.nansum(diffs ** 2, axis=1)

    return sigma_hat_sq


def obm_ci(all_thetas, theta_bar, b_n, theta_star, q=0.05, coord=0):
    """Construct CI using OBM variance estimator (Samsonov et al. 2025).

    Uses the standard normal quantile (plug-in approach).

    Args:
        all_thetas: (n_traj, T, d) all iterates.
        theta_bar: (n_traj, d) Polyak-Ruppert average.
        b_n: Block size.
        theta_star: (d,) true solution.
        q: CI error level.
        coord: Coordinate for CI/coverage.

    Returns:
        l2_errors: (n_traj,)
        ci_widths: (n_traj,)
        coverages: (n_traj,) binary.
    """
    n_traj, T, d = all_thetas.shape
    z = stats.norm.ppf(1 - q / 2)

    l2_errors = np.linalg.norm(theta_bar - theta_star, axis=1)

    u = np.zeros(d)
    u[coord] = 1.0

    sigma_hat_sq = obm_variance(all_thetas, theta_bar, b_n, u)
    se = np.sqrt(sigma_hat_sq / T)
    ci_widths = 2 * z * se

    lo = theta_bar[:, coord] - z * se
    hi = theta_bar[:, coord] + z * se
    coverages = ((lo <= theta_star[coord]) & (theta_star[coord] <= hi)).astype(float)

    has_nan = np.any(np.isnan(theta_bar), axis=1)
    l2_errors[has_nan] = np.nan
    ci_widths[has_nan] = np.nan
    coverages[has_nan] = 0.0

    return l2_errors, ci_widths, coverages


def msb_ci(all_thetas, theta_bar, b_n, theta_star, n_bootstrap=500,
           q=0.05, coord=0, rng=None):
    """Construct CI using Multiplier Subsample Bootstrap (Samsonov et al. 2025).

    For each trajectory, draw n_bootstrap sets of multiplier weights
    w_t ~ N(0,1), compute bootstrap statistic, and use bootstrap quantiles.

    Args:
        all_thetas: (n_traj, T, d) all iterates.
        theta_bar: (n_traj, d) Polyak-Ruppert average.
        b_n: Block size.
        theta_star: (d,) true solution.
        n_bootstrap: Number of bootstrap replications.
        q: CI error level.
        coord: Coordinate for CI/coverage.
        rng: numpy random Generator.

    Returns:
        l2_errors: (n_traj,)
        ci_widths: (n_traj,)
        coverages: (n_traj,) binary.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_traj, T, d = all_thetas.shape
    n_blocks = T - b_n + 1

    l2_errors = np.linalg.norm(theta_bar - theta_star, axis=1)

    # Precompute overlapping block averages for the coordinate of interest
    proj = all_thetas[:, :, coord]  # (n_traj, T)
    cumsum_proj = np.concatenate([np.zeros((n_traj, 1)), np.cumsum(proj, axis=1)], axis=1)
    block_avgs = (cumsum_proj[:, b_n:] - cumsum_proj[:, :n_blocks]) / b_n  # (n_traj, n_blocks)

    bar_coord = theta_bar[:, coord]  # (n_traj,)
    centered = block_avgs - bar_coord[:, None]  # (n_traj, n_blocks)

    z_lo = q / 2
    z_hi = 1 - q / 2

    ci_widths = np.full(n_traj, np.nan)
    coverages = np.zeros(n_traj)

    # Bootstrap: for each trajectory, generate multiplier weights and compute quantiles
    for i in range(n_traj):
        if np.any(np.isnan(theta_bar[i])):
            continue

        c_i = centered[i]  # (n_blocks,)
        # Draw multiplier weights: (n_bootstrap, n_blocks)
        weights = rng.standard_normal((n_bootstrap, n_blocks))
        # Bootstrap statistic (eq.12 of Samsonov 2025):
        #   theta_{n,b_n}(u) = sqrt(b_n) / sqrt(n - b_n + 1) * sum_t w_t * centered_t
        boot_stats = np.sqrt(b_n) / np.sqrt(n_blocks) * (weights * c_i[None, :]).sum(axis=1)

        # Quantiles of the bootstrap distribution
        q_lo = np.percentile(boot_stats, z_lo * 100)
        q_hi = np.percentile(boot_stats, z_hi * 100)

        # CI: theta_bar +/- bootstrap quantiles / sqrt(T)
        ci_lo = bar_coord[i] - q_hi / np.sqrt(T)
        ci_hi = bar_coord[i] - q_lo / np.sqrt(T)

        ci_widths[i] = ci_hi - ci_lo
        coverages[i] = float(ci_lo <= theta_star[coord] <= ci_hi)

    has_nan = np.any(np.isnan(theta_bar), axis=1)
    l2_errors[has_nan] = np.nan
    coverages[has_nan] = 0.0

    return l2_errors, ci_widths, coverages
