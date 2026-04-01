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


def _obm_variance_from_proj(proj, bar_coord, b_n):
    """Compute OBM variance without materializing full block_avgs array.

    Uses cumsum + online sum-of-squares to avoid (n_traj, n_blocks) allocation
    when n_blocks is large (T ~ 10^6).
    """
    n_traj, T = proj.shape
    n_blocks = T - b_n + 1
    cumsum = np.concatenate(
        [np.zeros((n_traj, 1)), np.nancumsum(proj, axis=1)], axis=1
    )

    # Process in chunks to avoid allocating (n_traj, n_blocks) all at once
    _CHUNK = max(1, min(n_blocks, 50_000_000 // max(n_traj, 1)))
    sum_sq = np.zeros(n_traj)

    for start in range(0, n_blocks, _CHUNK):
        end = min(start + _CHUNK, n_blocks)
        ba = (cumsum[:, start + b_n:end + b_n] - cumsum[:, start:end]) / b_n
        diffs = ba - bar_coord[:, None]
        sum_sq += np.nansum(diffs ** 2, axis=1)
        del ba, diffs

    return (b_n / n_blocks) * sum_sq


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
    bar_coord = theta_bar[:, coord]

    sigma_hat_sq = _obm_variance_from_proj(proj, bar_coord, b_n)

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

    Vectorized: processes all trajectories at once.  The bootstrap statistic
    S_b = sqrt(b_n)/sqrt(n_blocks) * sum_t w_t * c_t  is a weighted sum of
    centered block averages.  Since w_t ~ N(0,1), S_b|data ~ N(0, sigma_b^2)
    where sigma_b^2 = (b_n/n_blocks) * sum_t c_t^2 = OBM variance.  However,
    finite-bootstrap quantiles differ from normal quantiles due to sampling
    noise, which is the point of MSB.  We use a shared weight matrix across
    trajectories for efficiency.

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

    n_blocks = T - b_n + 1
    bar_coord = theta_bar[:, coord]

    cumsum = np.concatenate(
        [np.zeros((n_traj, 1)), np.nancumsum(proj, axis=1)], axis=1
    )

    scale = np.sqrt(b_n) / np.sqrt(n_blocks)
    z_lo = q / 2
    z_hi = 1 - q / 2

    # Vectorized bootstrap via chunked block_avgs to control memory.
    # boot_stats[b, i] = scale * sum_t w_{b,t} * (block_avg[i,t] - bar[i])
    # We compute (weights @ centered^T) in block-chunks to avoid
    # materializing the full (n_traj, n_blocks) centered array.
    _BLK_CHUNK = max(1, min(n_blocks, 50_000_000 // max(n_traj, 1)))

    # Generate all weights upfront — (n_bootstrap, n_blocks) in float32
    # to halve memory.  For quantile estimation float32 is sufficient.
    weights = rng.standard_normal(
        (n_bootstrap, n_blocks)).astype(np.float32)

    boot_stats = np.zeros((n_bootstrap, n_traj), dtype=np.float64)
    for start in range(0, n_blocks, _BLK_CHUNK):
        end = min(start + _BLK_CHUNK, n_blocks)
        ba = (cumsum[:, start + b_n:end + b_n] - cumsum[:, start:end]) / b_n
        centered_chunk = ba - bar_coord[:, None]  # (n_traj, chunk)
        # (n_bootstrap, chunk) @ (chunk, n_traj) -> (n_bootstrap, n_traj)
        boot_stats += weights[:, start:end] @ centered_chunk.T
        del ba, centered_chunk

    boot_stats *= scale
    del weights

    # Quantiles per trajectory: (n_traj,)
    q_lo_vals = np.percentile(boot_stats, z_lo * 100, axis=0)
    q_hi_vals = np.percentile(boot_stats, z_hi * 100, axis=0)

    ci_lo = bar_coord - q_hi_vals / np.sqrt(T)
    ci_hi = bar_coord - q_lo_vals / np.sqrt(T)

    ci_widths = ci_hi - ci_lo
    coverages = ((ci_lo <= theta_star[coord]) & (theta_star[coord] <= ci_hi)).astype(float)

    has_nan = np.any(np.isnan(theta_bar), axis=1)
    l2_errors[has_nan] = np.nan
    ci_widths[has_nan] = np.nan
    coverages[has_nan] = 0.0

    return l2_errors, ci_widths, coverages
