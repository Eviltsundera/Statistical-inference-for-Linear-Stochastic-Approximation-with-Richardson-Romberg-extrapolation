"""RR coverage stress sweep over Markov-chain mixing rates.

The transition matrix is generated as a lazy mixture

    P_rho = rho I + (1 - rho) P_0,

where ``P_0`` is the usual dense random transition matrix.  This preserves the
stationary distribution while shrinking the spectral gap, so it changes the
Markov dependence without changing the finite-state LSA problem generator.
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
from run_rr_blocksize_coverage import (
    _ci_from_sigma,
    _make_b_grid,
    _obm_variance_many,
    _summarize_ci,
    _variance_diagnostics,
)


DEFAULT_SCENARIOS = [
    ('baseline', 0.0),
    ('lazy_0p50', 0.50),
    ('lazy_0p80', 0.80),
    ('lazy_0p90', 0.90),
]


def _parse_scenario(text):
    """Parse label:lazy_probability."""
    parts = text.split(':')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            'scenario must have form label:lazy_probability'
        )
    label = parts[0].strip()
    if not label:
        raise argparse.ArgumentTypeError('scenario label must be nonempty')
    try:
        lazy_prob = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'lazy_probability must be numeric'
        ) from exc
    if lazy_prob < 0.0 or lazy_prob >= 1.0:
        raise argparse.ArgumentTypeError('lazy_probability must be in [0, 1)')
    return label, lazy_prob


def _lazy_transition(P_base, lazy_prob):
    n_states = P_base.shape[0]
    return lazy_prob * np.eye(n_states) + (1.0 - lazy_prob) * P_base


def _mixing_diagnostics(P):
    eigvals = np.linalg.eigvals(P)
    abs_vals = np.sort(np.abs(eigvals))
    slem = float(abs_vals[-2]) if len(abs_vals) >= 2 else 0.0
    gap = float(max(1.0 - slem, 0.0))
    relaxation = float(1.0 / gap) if gap > 0 else np.inf
    return {
        'slem': slem,
        'spectral_gap': gap,
        'relaxation_time': relaxation,
    }


def _worker(args):
    (scenario_label, lazy_prob, prob_seed, n_traj, T, n_states, d, K,
     burn_in, b_grid, lam, direction_coord, eig_min, eig_max, noise_target,
     rr_alphas) = args

    rng = np.random.default_rng(prob_seed)
    P_base, pi = generate_transition_matrix(n_states, rng)
    P = _lazy_transition(P_base, lazy_prob)
    mix_diag = _mixing_diagnostics(P)

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

    common = {
        'scenario': scenario_label,
        'lazy_prob': float(lazy_prob),
        'problem_seed': prob_seed,
        'T': int(T),
        'T_eff': int(n_eff),
        'sigma_true': float(sigma_true),
        'rr_alpha_1': float(rr_alphas[0]),
        'rr_alpha_2': float(rr_alphas[1]),
        'eig_min': float(eig_min),
        'eig_max': float(eig_max),
        'noise_target': float(noise_target),
        'max_a_norm': diagnostics['max_a_norm'],
        'max_rho': diagnostics['max_rho'],
        'warn_unstable': diagnostics['warn_unstable'],
        'warn_assumption': diagnostics['warn_assumption'],
        **mix_diag,
    }

    rows = []
    l2, width, coverage = _ci_from_sigma(
        rr_theta_bar, theta_star, direction,
        np.full(n_traj, sigma_true), n_eff,
    )
    rows.append({
        **common,
        'eta': np.nan,
        'b_n': 0,
        'lam': float(lam),
        'estimator': 'ORACLE',
        **_summarize_ci(l2, width, coverage, rr_theta_bar),
        'sigma_mean_raw': float(sigma_true),
        'rel_bias_raw': 0.0,
        'mse_raw': 0.0,
        'sigma_mean_used': float(sigma_true),
        'rel_bias_used': 0.0,
        'mse_used': 0.0,
        'negative_rate': 0.0,
        'clamped_rate': 0.0,
    })

    for eta, b in b_grid:
        if b not in sigma_obm:
            continue

        raw = sigma_obm[b]
        used = np.maximum(raw, 0.0)
        l2, width, coverage = _ci_from_sigma(
            rr_theta_bar, theta_star, direction, used, n_eff,
        )
        rows.append({
            **common,
            'eta': float(eta),
            'b_n': int(b),
            'lam': 0.0,
            'estimator': 'OBM',
            **_summarize_ci(l2, width, coverage, rr_theta_bar),
            **_variance_diagnostics(raw, used, sigma_true),
        })

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
        rows.append({
            **common,
            'eta': float(eta),
            'b_n': int(b),
            'lam': float(lam),
            'estimator': 'OBM_RR',
            **_summarize_ci(l2, width, coverage, rr_theta_bar),
            **_variance_diagnostics(raw_rr, used_rr, sigma_true),
        })

    return rows


def _aggregate(rows):
    df = pd.DataFrame(rows)
    group_cols = ['scenario', 'lazy_prob', 'T', 'eta', 'b_n', 'estimator']
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
        slem_median=('slem', 'median'),
        spectral_gap_median=('spectral_gap', 'median'),
        relaxation_time_median=('relaxation_time', 'median'),
        max_rho_median=('max_rho', 'median'),
        warn_unstable_rate=('warn_unstable', 'mean'),
    ).reset_index()

    oracle_width = (
        agg.loc[agg['estimator'] == 'ORACLE']
        .set_index(['scenario', 'T'])['width_median_x1e3']
    )
    agg['width_ratio_to_oracle'] = [
        row.width_median_x1e3 / oracle_width.loc[(row.scenario, row.T)]
        for row in agg.itertuples()
    ]
    return df, agg


def run_experiment(n_problems, n_traj, T_values, n_states, d, seed=42,
                   n_workers=None, direction_coord=None,
                   eig_min=0.25, eig_max=0.60, noise_target=0.35,
                   rr_alphas=(0.2, 0.1), bn_exps=None, lam=2.0,
                   scenarios=None,
                   out='results/stress/rr_mixing_stress.csv'):
    if bn_exps is None:
        bn_exps = [0.4, 0.5, 0.6]
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if isinstance(T_values, (int, np.integer)):
        T_values = [int(T_values)]
    else:
        T_values = [int(T) for T in T_values]
    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_problems)

    dir_desc = f"e_{direction_coord}" if direction_coord is not None else "random"
    print(f"RR mixing stress sweep: {n_problems} problems x {n_traj} traj, "
          f"T={T_values}, workers={n_workers}")
    print(f"  RR alphas: {rr_alphas}")
    print(f"  bn_exps: {bn_exps}, lam={lam}")
    print(f"  Problem gen: eig=[{eig_min},{eig_max}], noise={noise_target}")
    print(f"  Direction: {dir_desc}")
    print("  Scenarios:")
    for label, lazy_prob in scenarios:
        print(f"    {label}: lazy_prob={lazy_prob}")
    print(flush=True)

    rng_master = np.random.default_rng(seed)
    seeds = [int(rng_master.integers(0, 2**31)) for _ in range(n_problems)]

    all_rows = []
    t_all = time.time()
    for scenario_label, lazy_prob in scenarios:
        print(f"\nScenario {scenario_label}: lazy_prob={lazy_prob}",
              flush=True)

        for T in T_values:
            K = max(int(T ** 0.3), 5)
            burn_in = min(1000, T // 10)
            b_grid = _make_b_grid(T, bn_exps, lam)
            print(f"  T={T}: K={K}, burn_in={burn_in}, b_grid={b_grid}",
                  flush=True)

            task_args = [
                (
                    scenario_label, lazy_prob, s, n_traj, T, n_states, d,
                    K, burn_in, b_grid, lam, direction_coord, eig_min,
                    eig_max, noise_target, tuple(rr_alphas),
                )
                for s in seeds
            ]

            t_start = time.time()
            completed = 0
            with mp.Pool(n_workers) as pool:
                for rows in pool.imap_unordered(_worker, task_args):
                    completed += 1
                    all_rows.extend(rows)
                    if (
                        completed % max(1, n_problems // 20) == 0
                        or completed == 1
                    ):
                        elapsed = time.time() - t_start
                        eta_sec = elapsed / completed * (
                            n_problems - completed
                        )
                        oracle = next(
                            r for r in rows if r['estimator'] == 'ORACLE'
                        )
                        print(
                            f"    {scenario_label} T={T} "
                            f"[{completed}/{n_problems}] "
                            f"last oracle cov={oracle['coverage_pct']:.0f}% | "
                            f"{elapsed:.0f}s elapsed, ~{eta_sec:.0f}s left",
                            flush=True,
                        )
            print(f"  Finished {scenario_label} T={T} in "
                  f"{time.time() - t_start:.0f}s", flush=True)

    df, agg = _aggregate(all_rows)
    total = time.time() - t_all

    print(f"\n{'=' * 128}")
    print(f"RESULTS ({n_problems} problems, T={T_values}, {n_traj} traj) "
          f"in {total:.0f}s ({total / 60:.1f}min)")
    print("Median over problems. Width and L2 are x 1e-3.")
    print("=" * 128)
    header = (
        f"{'scenario':<12} {'lazy':>5} {'T':>9} {'eta':>5} {'est':>8} "
        f"{'L2':>8} {'Width':>8} {'Cov med':>8} {'W/orcl':>8} "
        f"{'bias':>9} {'neg%':>7} {'gap':>8}"
    )
    print(header)
    print("-" * len(header))
    for _, row in agg.iterrows():
        eta_str = '-' if np.isnan(row['eta']) else f"{row['eta']:.1f}"
        print(
            f"{row['scenario']:<12} {row['lazy_prob']:>5.2f} "
            f"{int(row['T']):>9} {eta_str:>5} "
            f"{row['estimator']:>8} {row['l2_median_x1e3']:>8.2f} "
            f"{row['width_median_x1e3']:>8.2f} "
            f"{row['coverage_median_pct']:>8.1f} "
            f"{row['width_ratio_to_oracle']:>8.3f} "
            f"{row['rel_bias_raw_median']:>9.3f} "
            f"{100 * row['negative_rate_mean']:>7.2f} "
            f"{row['spectral_gap_median']:>8.3f}"
        )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    summary_path = out_path.with_name(out_path.stem + '_summary.csv')
    agg.to_csv(summary_path, index=False)
    print(f"\nSaved per-problem rows to {out_path}")
    print(f"Saved summary to          {summary_path}")

    return df, agg


def main():
    parser = argparse.ArgumentParser(description="RR mixing stress sweep")
    parser.add_argument('--n-problems', type=int, default=10)
    parser.add_argument('--n-traj', type=int, default=50)
    parser.add_argument('--T', type=int, nargs='+', default=[100000])
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
                        default=[0.4, 0.5, 0.6])
    parser.add_argument('--lam', type=float, default=2.0)
    parser.add_argument(
        '--scenario', action='append', type=_parse_scenario,
        help=('Scenario label:lazy_probability. Can be repeated. Defaults to '
              'baseline, lazy_0p50, lazy_0p80, lazy_0p90.'),
    )
    parser.add_argument('--out', type=str,
                        default='results/stress/rr_mixing_stress.csv')
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
        scenarios=args.scenario,
        out=args.out,
    )


if __name__ == '__main__':
    main()
