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
#
# These functions accept a 1-D projection `proj` of shape (n_traj, T)
# instead of the full (n_traj, T, d) iterate array.  This reduces memory
# by a factor of d.
# ---------------------------------------------------------------------------

def _block_avgs_from_proj(proj, b_n):
    """Compute overlapping block averages from scalar projection.

    Args:
        proj: (n_traj, T) scalar projection theta_t[coord].
        b_n: Block size.

    Returns:
        block_avgs: (n_traj, T - b_n + 1) overlapping block averages.
    """
    n_traj, T = proj.shape
    n_blocks = T - b_n + 1
    cumsum = np.concatenate(
        [np.zeros((n_traj, 1)), np.nancumsum(proj, axis=1)], axis=1
    )
    return (cumsum[:, b_n:] - cumsum[:, :n_blocks]) / b_n


def obm_ci(proj, theta_bar, b_n, theta_star, q=0.05, coord=0):
    """Construct CI using OBM variance estimator (Samsonov et al. 2025).

    Args:
        proj: (n_traj, T) projection of iterates onto coordinate `coord`.
        theta_bar: (n_traj, d) average (all coordinates, for L2 error).
        b_n: Block size.
        theta_star: (d,) true solution.
        q: CI error level.
        coord: Coordinate index (must match the coord used for proj).

    Returns:
        l2_errors, ci_widths, coverages: each (n_traj,).
    """
    n_traj, T = proj.shape
    z = stats.norm.ppf(1 - q / 2)

    l2_errors = np.linalg.norm(theta_bar - theta_star, axis=1)

    block_avgs = _block_avgs_from_proj(proj, b_n)
    n_blocks = block_avgs.shape[1]

    bar_coord = theta_bar[:, coord]
    diffs = block_avgs - bar_coord[:, None]
    sigma_hat_sq = (b_n / n_blocks) * np.nansum(diffs ** 2, axis=1)

    se = np.sqrt(sigma_hat_sq / T)
    ci_widths = 2 * z * se

    lo = bar_coord - z * se
    hi = bar_coord + z * se
    coverages = ((lo <= theta_star[coord]) & (theta_star[coord] <= hi)).astype(float)

    has_nan = np.any(np.isnan(theta_bar), axis=1)
    l2_errors[has_nan] = np.nan
    ci_widths[has_nan] = np.nan
    coverages[has_nan] = 0.0

    return l2_errors, ci_widths, coverages


def msb_ci(proj, theta_bar, b_n, theta_star, n_bootstrap=500,
           q=0.05, coord=0, rng=None):
    """Construct CI using Multiplier Subsample Bootstrap (Samsonov et al. 2025).

    Args:
        proj: (n_traj, T) projection of iterates onto coordinate `coord`.
        theta_bar: (n_traj, d) average (all coordinates, for L2 error).
        b_n: Block size.
        theta_star: (d,) true solution.
        n_bootstrap: Number of bootstrap replications.
        q: CI error level.
        coord: Coordinate index (must match the coord used for proj).
        rng: numpy random Generator.

    Returns:
        l2_errors, ci_widths, coverages: each (n_traj,).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_traj, T = proj.shape

    l2_errors = np.linalg.norm(theta_bar - theta_star, axis=1)

    block_avgs = _block_avgs_from_proj(proj, b_n)
    n_blocks = block_avgs.shape[1]

    bar_coord = theta_bar[:, coord]
    centered = block_avgs - bar_coord[:, None]

    z_lo = q / 2
    z_hi = 1 - q / 2

    ci_widths = np.full(n_traj, np.nan)
    coverages = np.zeros(n_traj)

    for i in range(n_traj):
        if np.isnan(bar_coord[i]):
            continue

        c_i = centered[i]
        weights = rng.standard_normal((n_bootstrap, n_blocks))
        boot_stats = np.sqrt(b_n) / np.sqrt(n_blocks) * (weights * c_i[None, :]).sum(axis=1)

        q_lo = np.percentile(boot_stats, z_lo * 100)
        q_hi = np.percentile(boot_stats, z_hi * 100)

        ci_lo = bar_coord[i] - q_hi / np.sqrt(T)
        ci_hi = bar_coord[i] - q_lo / np.sqrt(T)

        ci_widths[i] = ci_hi - ci_lo
        coverages[i] = float(ci_lo <= theta_star[coord] <= ci_hi)

    has_nan = np.any(np.isnan(theta_bar), axis=1)
    l2_errors[has_nan] = np.nan
    coverages[has_nan] = 0.0

    return l2_errors, ci_widths, coverages
