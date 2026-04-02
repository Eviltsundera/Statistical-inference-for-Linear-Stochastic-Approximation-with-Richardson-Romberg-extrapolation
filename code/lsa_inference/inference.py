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

def batch_mean_ci(batch_means, n, theta_star, n0=0, q=0.05, direction=None):
    """Compute L2 error, CI width, and coverage from non-overlapping batch means.

    Args:
        batch_means: (n_traj, K, d) array of batch means.
        n: Batch size (iterates per batch).
        theta_star: (d,) true solution.
        n0: Intra-batch discard.
        q: CI error level (0.05 for 95% CI).
        direction: (d,) unit vector for projection. If None, defaults to e_0.

    Returns:
        l2_errors: (n_traj,)
        ci_widths: (n_traj,)
        coverages: (n_traj,) binary (0/1).
    """
    z = stats.norm.ppf(1 - q / 2)
    n_traj, K, d = batch_means.shape

    if direction is None:
        direction = np.eye(d)[0]

    theta_bars = np.nanmean(batch_means, axis=1)  # (n_traj, d)
    l2_errors = np.linalg.norm(theta_bars - theta_star, axis=1)

    # Project batch means and overall mean onto direction
    bm_proj = np.einsum('nkd,d->nk', batch_means, direction)  # (n_traj, K)
    bar_proj = theta_bars @ direction  # (n_traj,)
    diffs = bm_proj - bar_proj[:, None]
    var_dir = (n - n0) / K * np.nansum(diffs ** 2, axis=1)

    se = np.sqrt(var_dir / (K * (n - n0)))
    ci_widths = 2 * z * se

    star_proj = theta_star @ direction
    lo = bar_proj - z * se
    hi = bar_proj + z * se
    coverages = ((lo <= star_proj) & (star_proj <= hi)).astype(float)

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


def obm_ci(proj, theta_bar, b_n, theta_star, q=0.05, direction=None):
    """Construct CI using OBM variance estimator (Samsonov et al. 2025).

    Args:
        proj: (n_traj, T) projection of iterates onto `direction`.
        theta_bar: (n_traj, d) average (all coordinates, for L2 error).
        b_n: Block size.
        theta_star: (d,) true solution.
        q: CI error level.
        direction: (d,) unit vector (must match the direction used for proj).
            If None, defaults to e_0.

    Returns:
        l2_errors, ci_widths, coverages: each (n_traj,).
    """
    n_traj, T = proj.shape
    d = theta_bar.shape[1]
    z = stats.norm.ppf(1 - q / 2)

    if direction is None:
        direction = np.eye(d)[0]

    l2_errors = np.linalg.norm(theta_bar - theta_star, axis=1)
    bar_proj = theta_bar @ direction
    star_proj = theta_star @ direction

    sigma_hat_sq = _obm_variance_from_proj(proj, bar_proj, b_n)

    se = np.sqrt(sigma_hat_sq / T)
    ci_widths = 2 * z * se

    lo = bar_proj - z * se
    hi = bar_proj + z * se
    coverages = ((lo <= star_proj) & (star_proj <= hi)).astype(float)

    has_nan = np.any(np.isnan(theta_bar), axis=1)
    l2_errors[has_nan] = np.nan
    ci_widths[has_nan] = np.nan
    coverages[has_nan] = 0.0

    return l2_errors, ci_widths, coverages


def obm_rr_ci(proj, theta_bar, b_n, theta_star, lam=2, q=0.05, direction=None):
    """Construct CI using RR-corrected OBM variance estimator.

    Uses two block sizes b_n and lam*b_n, extrapolates to cancel the
    leading O(1/b_n) truncation bias of the OBM variance estimate.

    sigma^2_RR = lam/(lam-1) * sigma^2(lam*b) - 1/(lam-1) * sigma^2(b)

    Args:
        proj: (n_traj, T) projection of iterates onto `direction`.
        theta_bar: (n_traj, d) average (all coordinates, for L2 error).
        b_n: Base block size.
        theta_star: (d,) true solution.
        lam: Block size multiplier (default 2).
        q: CI error level.
        direction: (d,) unit vector. If None, defaults to e_0.

    Returns:
        l2_errors, ci_widths, coverages: each (n_traj,).
    """
    n_traj, T = proj.shape
    d = theta_bar.shape[1]
    z = stats.norm.ppf(1 - q / 2)

    if direction is None:
        direction = np.eye(d)[0]

    l2_errors = np.linalg.norm(theta_bar - theta_star, axis=1)
    bar_proj = theta_bar @ direction
    star_proj = theta_star @ direction

    b_large = int(lam * b_n)
    b_large = min(b_large, T - 1)

    sigma_sq_small = _obm_variance_from_proj(proj, bar_proj, b_n)
    sigma_sq_large = _obm_variance_from_proj(proj, bar_proj, b_large)

    # RR combination: cancel O(1/b) bias
    w_large = lam / (lam - 1)
    w_small = 1 / (lam - 1)
    sigma_hat_sq = w_large * sigma_sq_large - w_small * sigma_sq_small
    # Clamp to non-negative (RR can produce negative estimates)
    sigma_hat_sq = np.maximum(sigma_hat_sq, 0.0)

    se = np.sqrt(sigma_hat_sq / T)
    ci_widths = 2 * z * se

    lo = bar_proj - z * se
    hi = bar_proj + z * se
    coverages = ((lo <= star_proj) & (star_proj <= hi)).astype(float)

    has_nan = np.any(np.isnan(theta_bar), axis=1)
    l2_errors[has_nan] = np.nan
    ci_widths[has_nan] = np.nan
    coverages[has_nan] = 0.0

    return l2_errors, ci_widths, coverages


def msb_ci(proj, theta_bar, b_n, theta_star, n_bootstrap=500,
           q=0.05, direction=None, rng=None):
    """Construct CI using Multiplier Subsample Bootstrap (Samsonov et al. 2025).

    Args:
        proj: (n_traj, T) projection of iterates onto `direction`.
        theta_bar: (n_traj, d) average (all coordinates, for L2 error).
        b_n: Block size.
        theta_star: (d,) true solution.
        n_bootstrap: Number of bootstrap replications.
        q: CI error level.
        direction: (d,) unit vector (must match the direction used for proj).
            If None, defaults to e_0.
        rng: numpy random Generator.

    Returns:
        l2_errors, ci_widths, coverages: each (n_traj,).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_traj, T = proj.shape
    d = theta_bar.shape[1]

    if direction is None:
        direction = np.eye(d)[0]

    l2_errors = np.linalg.norm(theta_bar - theta_star, axis=1)

    n_blocks = T - b_n + 1
    bar_proj = theta_bar @ direction
    star_proj = theta_star @ direction

    cumsum = np.concatenate(
        [np.zeros((n_traj, 1)), np.nancumsum(proj, axis=1)], axis=1
    )

    scale = np.sqrt(b_n) / np.sqrt(n_blocks)
    z_lo = q / 2
    z_hi = 1 - q / 2

    _BLK_CHUNK = max(1, min(n_blocks, 50_000_000 // max(n_traj, 1)))

    weights = rng.standard_normal(
        (n_bootstrap, n_blocks)).astype(np.float32)

    boot_stats = np.zeros((n_bootstrap, n_traj), dtype=np.float64)
    for start in range(0, n_blocks, _BLK_CHUNK):
        end = min(start + _BLK_CHUNK, n_blocks)
        ba = (cumsum[:, start + b_n:end + b_n] - cumsum[:, start:end]) / b_n
        centered_chunk = ba - bar_proj[:, None]  # (n_traj, chunk)
        boot_stats += weights[:, start:end] @ centered_chunk.T
        del ba, centered_chunk

    boot_stats *= scale
    del weights

    q_lo_vals = np.percentile(boot_stats, z_lo * 100, axis=0)
    q_hi_vals = np.percentile(boot_stats, z_hi * 100, axis=0)

    ci_lo = bar_proj - q_hi_vals / np.sqrt(T)
    ci_hi = bar_proj - q_lo_vals / np.sqrt(T)

    ci_widths = ci_hi - ci_lo
    coverages = ((ci_lo <= star_proj) & (star_proj <= ci_hi)).astype(float)

    has_nan = np.any(np.isnan(theta_bar), axis=1)
    l2_errors[has_nan] = np.nan
    ci_widths[has_nan] = np.nan
    coverages[has_nan] = 0.0

    return l2_errors, ci_widths, coverages
