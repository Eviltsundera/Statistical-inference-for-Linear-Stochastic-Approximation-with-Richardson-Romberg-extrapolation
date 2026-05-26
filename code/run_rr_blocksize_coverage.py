"""RR confidence-interval coverage sweep over OBM block sizes.

For each problem and horizon, this runner computes the RR averaged
constant-stepsize path once, then evaluates OBM and lugsail/OBM-RR intervals
over a grid of block sizes

    b = floor(T^eta).

It also records raw variance-estimator diagnostics against the analytic
finite-state long-run variance sigma^2_inf(u), including the negative rate of
the signed OBM-RR estimator before clamping.
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lsa_inference.markov_chain import (
    generate_transition_matrix,
    simulate_chains_batch,
)
from lsa_inference.lsa_problem import (
    compute_asymptotic_variance,
    compute_theta_star,
    generate_A,
    generate_b,
    problem_diagnostics,
)
from lsa_inference.lsa_engine import prepare_arrays, run_rr_full


def _make_b_grid(T, exponents, lam):
    """Return unique block sizes floor(T^eta), capped so lam*b < T."""
    cap = max(int((T - 1) // lam), 5)
    values = []
    for eta in exponents:
        b = int(T ** eta)
        b = max(5, min(b, cap))
        values.append((float(eta), int(b)))

    # Preserve first exponent producing a block size if duplicates appear.
    seen = set()
    out = []
    for eta, b in values:
        if b not in seen:
            seen.add(b)
            out.append((eta, b))
    return out


def _obm_variance_many(proj, bar_proj, b_values):
    """Compute OBM variances for several block sizes with one cumsum."""
    n_traj, T_eff = proj.shape
    cumsum = np.concatenate(
        [np.zeros((n_traj, 1)), np.nancumsum(proj, axis=1)], axis=1
    )

    out = {}
    for b in sorted(set(int(b) for b in b_values)):
        if b <= 0 or b >= T_eff:
            continue
        n_blocks = T_eff - b + 1
        chunk = max(1, min(n_blocks, 50_000_000 // max(n_traj, 1)))
        sum_sq = np.zeros(n_traj)

        for start in range(0, n_blocks, chunk):
            end = min(start + chunk, n_blocks)
            block_avg = (
                cumsum[:, start + b:end + b] - cumsum[:, start:end]
            ) / b
            diffs = block_avg - bar_proj[:, None]
            sum_sq += np.nansum(diffs ** 2, axis=1)
            del block_avg, diffs

        out[b] = (b / n_blocks) * sum_sq

    return out


def _ci_from_sigma(theta_bar, theta_star, direction, sigma_sq, n_eff,
                   q=0.05):
    """Return L2, width, and coverage arrays using per-trajectory sigma_sq."""
    z = stats.norm.ppf(1 - q / 2)
    sigma_sq = np.asarray(sigma_sq, dtype=float)
    se = np.sqrt(np.maximum(sigma_sq, 0.0) / n_eff)

    l2 = np.linalg.norm(theta_bar - theta_star, axis=1)
    bar_proj = theta_bar @ direction
    star_proj = theta_star @ direction
    width = 2 * z * se
    coverage = (
        (bar_proj - z * se <= star_proj) & (star_proj <= bar_proj + z * se)
    ).astype(float)

    has_nan = np.any(np.isnan(theta_bar), axis=1) | ~np.isfinite(sigma_sq)
    l2[has_nan] = np.nan
    width[has_nan] = np.nan
    coverage[has_nan] = 0.0
    return l2, width, coverage


def _summarize_ci(l2, width, coverage, theta_bar):
    diverged = int(np.sum(np.any(np.isnan(theta_bar), axis=1)))
    return {
        'l2_mean_x1e3': float(np.nanmean(l2)) * 1e3,
        'width_mean_x1e3': float(np.nanmean(width)) * 1e3,
        'coverage_pct': float(np.nanmean(coverage)) * 100,
        'diverged': diverged,
    }


def _variance_diagnostics(raw_sigma, used_sigma, sigma_true):
    raw_sigma = np.asarray(raw_sigma, dtype=float)
    used_sigma = np.asarray(used_sigma, dtype=float)
    finite_raw = raw_sigma[np.isfinite(raw_sigma)]
    finite_used = used_sigma[np.isfinite(used_sigma)]

    if finite_raw.size == 0:
        return {
            'sigma_mean_raw': np.nan,
            'rel_bias_raw': np.nan,
            'mse_raw': np.nan,
            'sigma_mean_used': np.nan,
            'rel_bias_used': np.nan,
            'mse_used': np.nan,
            'negative_rate': np.nan,
            'clamped_rate': np.nan,
        }

    return {
        'sigma_mean_raw': float(np.mean(finite_raw)),
        'rel_bias_raw': float((np.mean(finite_raw) - sigma_true) / sigma_true),
        'mse_raw': float(np.mean((finite_raw - sigma_true) ** 2)),
        'sigma_mean_used': float(np.mean(finite_used)),
        'rel_bias_used': float(
            (np.mean(finite_used) - sigma_true) / sigma_true
        ),
        'mse_used': float(np.mean((finite_used - sigma_true) ** 2)),
        'negative_rate': float(np.mean(finite_raw < 0.0)),
        'clamped_rate': float(np.mean((finite_raw < 0.0) & (finite_used == 0.0))),
    }


def _worker(args):
    (prob_seed, n_traj, T, n_states, d, K, burn_in, b_grid, lam,
     direction_coord, eig_min, eig_max, noise_target, rr_alphas) = args

    rng = np.random.default_rng(prob_seed)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, _ = generate_A(
        n_states, d, pi, rng,
        eig_min=eig_min, eig_max=eig_max, noise_target=noise_target,
    )
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)
    A_arr, b_arr = prepare_arrays(A_list, b_list)
    diagnostics = problem_diagnostics(A_list, alpha_warn=max(rr_alphas))

    if direction_coord is None:
        u = rng.standard_normal(d)
        direction = u / np.linalg.norm(u)
    else:
        direction = np.eye(d)[direction_coord]

    sigma_true = compute_asymptotic_variance(
        A_list, b_list, P, pi, theta_star, direction,
    )

    traj_rng = np.random.default_rng(rng.integers(0, 2**31))
    trajs = simulate_chains_batch(P, pi, T, n_traj, traj_rng)
    rr_proj, rr_theta_bar, _, _, _, _, _ = run_rr_full(
        A_arr, b_arr, trajs, list(rr_alphas), K, burn_in,
        direction=direction,
    )

    n_eff = rr_proj.shape[1]
    bar_proj = rr_theta_bar @ direction
    diverged = np.any(np.isnan(rr_theta_bar), axis=1)

    base_b = [b for _, b in b_grid]
    all_b = set(base_b)
    all_b.update(int(round(lam * b)) for b in base_b)
    sigma_obm = _obm_variance_many(rr_proj, bar_proj, all_b)
    for values in sigma_obm.values():
        values[diverged] = np.nan

    rows = []

    l2, width, coverage = _ci_from_sigma(
        rr_theta_bar, theta_star, direction,
        np.full(n_traj, sigma_true), n_eff,
    )
    row = {
        'problem_seed': prob_seed,
        'T': int(T),
        'T_eff': int(n_eff),
        'eta': np.nan,
        'b_n': 0,
        'lam': float(lam),
        'estimator': 'ORACLE',
        'sigma_true': float(sigma_true),
        **_summarize_ci(l2, width, coverage, rr_theta_bar),
        'sigma_mean_raw': float(sigma_true),
        'rel_bias_raw': 0.0,
        'mse_raw': 0.0,
        'sigma_mean_used': float(sigma_true),
        'rel_bias_used': 0.0,
        'mse_used': 0.0,
        'negative_rate': 0.0,
        'clamped_rate': 0.0,
    }
    rows.append(row)

    for eta, b in b_grid:
        if b not in sigma_obm:
            continue

        raw = sigma_obm[b]
        used = np.maximum(raw, 0.0)
        l2, width, coverage = _ci_from_sigma(
            rr_theta_bar, theta_star, direction, used, n_eff,
        )
        row = {
            'problem_seed': prob_seed,
            'T': int(T),
            'T_eff': int(n_eff),
            'eta': float(eta),
            'b_n': int(b),
            'lam': 0.0,
            'estimator': 'OBM',
            'sigma_true': float(sigma_true),
            **_summarize_ci(l2, width, coverage, rr_theta_bar),
            **_variance_diagnostics(raw, used, sigma_true),
        }
        row.update({
            'rr_alpha_1': float(rr_alphas[0]),
            'rr_alpha_2': float(rr_alphas[1]),
            'max_a_norm': diagnostics['max_a_norm'],
            'max_rho': diagnostics['max_rho'],
            'warn_unstable': diagnostics['warn_unstable'],
            'warn_assumption': diagnostics['warn_assumption'],
        })
        rows.append(row)

        b_large = int(round(lam * b))
        if b_large not in sigma_obm:
            continue

        small = sigma_obm[b]
        large = sigma_obm[b_large]
        raw_rr = (lam / (lam - 1)) * large - (1 / (lam - 1)) * small
        used_rr = np.maximum(raw_rr, 0.0)
        l2, width, coverage = _ci_from_sigma(
            rr_theta_bar, theta_star, direction, used_rr, n_eff,
        )
        row = {
            'problem_seed': prob_seed,
            'T': int(T),
            'T_eff': int(n_eff),
            'eta': float(eta),
            'b_n': int(b),
            'lam': float(lam),
            'estimator': 'OBM_RR',
            'sigma_true': float(sigma_true),
            **_summarize_ci(l2, width, coverage, rr_theta_bar),
            **_variance_diagnostics(raw_rr, used_rr, sigma_true),
            'rr_alpha_1': float(rr_alphas[0]),
            'rr_alpha_2': float(rr_alphas[1]),
            'max_a_norm': diagnostics['max_a_norm'],
            'max_rho': diagnostics['max_rho'],
            'warn_unstable': diagnostics['warn_unstable'],
            'warn_assumption': diagnostics['warn_assumption'],
        }
        rows.append(row)

    return rows


def _aggregate(rows):
    df = pd.DataFrame(rows)
    group_cols = ['T', 'eta', 'b_n', 'estimator']
    agg = df.groupby(group_cols, dropna=False, sort=True).agg(
        T_eff=('T_eff', 'first'),
        n_problems=('problem_seed', 'nunique'),
        sigma_true_median=('sigma_true', 'median'),
        l2_median_x1e3=('l2_mean_x1e3', 'median'),
        width_median_x1e3=('width_mean_x1e3', 'median'),
        coverage_median_pct=('coverage_pct', 'median'),
        coverage_mean_pct=('coverage_pct', 'mean'),
        rel_bias_raw_median=('rel_bias_raw', 'median'),
        rel_bias_raw_mean=('rel_bias_raw', 'mean'),
        rel_bias_used_median=('rel_bias_used', 'median'),
        mse_raw_mean=('mse_raw', 'mean'),
        mse_used_mean=('mse_used', 'mean'),
        negative_rate_mean=('negative_rate', 'mean'),
        clamped_rate_mean=('clamped_rate', 'mean'),
        diverged_total=('diverged', 'sum'),
    ).reset_index()

    oracle_width_by_T = (
        agg.loc[agg['estimator'] == 'ORACLE']
        .set_index('T')['width_median_x1e3']
    )
    agg['width_ratio_to_oracle'] = (
        agg['width_median_x1e3'] / agg['T'].map(oracle_width_by_T)
    )
    return df, agg


def run_experiment(n_problems, n_traj, T_values, n_states, d, seed=42,
                   n_workers=None, direction_coord=None,
                   eig_min=0.25, eig_max=0.60, noise_target=0.35,
                   rr_alphas=(0.2, 0.1), bn_exps=None, lam=2.0,
                   out='results/rr_blocksize_coverage.csv'):
    if bn_exps is None:
        bn_exps = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    if isinstance(T_values, (int, np.integer)):
        T_values = [int(T_values)]
    else:
        T_values = [int(T) for T in T_values]

    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_problems)

    dir_desc = f"e_{direction_coord}" if direction_coord is not None else "random"
    print(f"RR block-size coverage sweep: {n_problems} problems x "
          f"{n_traj} traj, T={T_values}, workers={n_workers}")
    print(f"  RR alphas: {rr_alphas}")
    print(f"  bn_exps: {bn_exps}, lam={lam}")
    print(f"  Problem gen: eig=[{eig_min},{eig_max}], noise={noise_target}")
    print(f"  Direction: {dir_desc}")
    print(flush=True)

    rng_master = np.random.default_rng(seed)
    seeds = [int(rng_master.integers(0, 2**31)) for _ in range(n_problems)]

    all_rows = []
    t_all = time.time()
    for T in T_values:
        K = max(int(T ** 0.3), 5)
        burn_in = min(1000, T // 10)
        b_grid = _make_b_grid(T, bn_exps, lam)
        print(f"\nT={T}: K={K}, burn_in={burn_in}, b_grid={b_grid}",
              flush=True)

        task_args = [
            (s, n_traj, T, n_states, d, K, burn_in, b_grid, lam,
             direction_coord, eig_min, eig_max, noise_target, tuple(rr_alphas))
            for s in seeds
        ]

        t_start = time.time()
        completed = 0
        with mp.Pool(n_workers) as pool:
            for rows in pool.imap_unordered(_worker, task_args):
                completed += 1
                all_rows.extend(rows)
                if completed % max(1, n_problems // 20) == 0 or completed == 1:
                    elapsed = time.time() - t_start
                    eta_sec = elapsed / completed * (n_problems - completed)
                    oracle = next(r for r in rows if r['estimator'] == 'ORACLE')
                    print(f"  T={T} [{completed}/{n_problems}] "
                          f"last oracle cov={oracle['coverage_pct']:.0f}% | "
                          f"{elapsed:.0f}s elapsed, ~{eta_sec:.0f}s left",
                          flush=True)

        print(f"Finished T={T} in {time.time() - t_start:.0f}s", flush=True)

    df, agg = _aggregate(all_rows)
    total = time.time() - t_all

    print(f"\n{'=' * 110}")
    print(f"RESULTS ({n_problems} problems, T={T_values}, {n_traj} traj) "
          f"in {total:.0f}s ({total / 60:.1f}min)")
    print("Median over problems. Width and L2 are x 1e-3.")
    print("=" * 110)
    header = (f"{'T':>9} {'eta':>5} {'b':>7} {'est':>8} {'L2':>8} "
              f"{'Width':>8} {'Cov med':>8} {'Cov mean':>9} "
              f"{'W/orcl':>8} {'bias':>9} {'neg%':>7}")
    print(header)
    print("-" * len(header))
    for _, row in agg.iterrows():
        eta_str = '-' if np.isnan(row['eta']) else f"{row['eta']:.1f}"
        print(f"{int(row['T']):>9} {eta_str:>5} {int(row['b_n']):>7} "
              f"{row['estimator']:>8} {row['l2_median_x1e3']:>8.2f} "
              f"{row['width_median_x1e3']:>8.2f} "
              f"{row['coverage_median_pct']:>8.1f} "
              f"{row['coverage_mean_pct']:>9.1f} "
              f"{row['width_ratio_to_oracle']:>8.3f} "
              f"{row['rel_bias_raw_median']:>9.3f} "
              f"{100 * row['negative_rate_mean']:>7.2f}")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    summary_path = out_path.with_name(out_path.stem + '_summary.csv')
    agg.to_csv(summary_path, index=False)
    print(f"\nSaved per-problem rows to {out_path}")
    print(f"Saved summary to          {summary_path}")

    return df, agg


def main():
    parser = argparse.ArgumentParser(
        description="RR coverage sweep over OBM/OBM-RR block sizes",
    )
    parser.add_argument('--n-problems', type=int, default=10)
    parser.add_argument('--n-traj', type=int, default=50)
    parser.add_argument('--T', type=int, nargs='+', default=[10000])
    parser.add_argument('--n-states', type=int, default=10)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-workers', type=int, default=None)
    parser.add_argument('--direction-coord', type=int, default=None)
    parser.add_argument('--eig-min', type=float, default=0.25)
    parser.add_argument('--eig-max', type=float, default=0.60)
    parser.add_argument('--noise-target', type=float, default=0.35)
    parser.add_argument('--rr-alphas', type=float, nargs=2,
                        default=[0.2, 0.1])
    parser.add_argument('--bn-exps', type=float, nargs='+',
                        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument('--lam', type=float, default=2.0)
    parser.add_argument('--out', type=str,
                        default='results/rr_blocksize_coverage.csv')
    args = parser.parse_args()

    run_experiment(
        args.n_problems, args.n_traj, args.T,
        args.n_states, args.d, args.seed,
        args.n_workers, args.direction_coord,
        eig_min=args.eig_min, eig_max=args.eig_max,
        noise_target=args.noise_target,
        rr_alphas=tuple(args.rr_alphas),
        bn_exps=args.bn_exps,
        lam=args.lam,
        out=args.out,
    )


if __name__ == '__main__':
    main()

