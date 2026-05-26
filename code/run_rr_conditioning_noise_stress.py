"""RR coverage stress sweep over conditioning and matrix-noise levels.

This runner reuses the block-size coverage worker and varies the finite-state
LSA problem generator.  The goal is to check whether the OBM/lugsail coverage
picture survives when the mean matrix contracts more slowly or the
state-dependent matrix perturbation is larger.
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_rr_blocksize_coverage import _make_b_grid, _worker as _block_worker


DEFAULT_SCENARIOS = [
    ('baseline', 0.25, 0.60, 0.35),
    ('weak_mean', 0.12, 0.30, 0.18),
    ('high_noise', 0.25, 0.60, 0.45),
    ('weak_high_noise', 0.15, 0.35, 0.25),
]


def _parse_scenario(text):
    """Parse label:eig_min:eig_max:noise_target."""
    parts = text.split(':')
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            'scenario must have form label:eig_min:eig_max:noise_target'
        )
    label = parts[0].strip()
    if not label:
        raise argparse.ArgumentTypeError('scenario label must be nonempty')
    try:
        eig_min = float(parts[1])
        eig_max = float(parts[2])
        noise_target = float(parts[3])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'eig_min, eig_max, and noise_target must be numeric'
        ) from exc
    if eig_min <= 0 or eig_max <= 0 or eig_min > eig_max:
        raise argparse.ArgumentTypeError(
            'scenario requires 0 < eig_min <= eig_max'
        )
    if noise_target < 0:
        raise argparse.ArgumentTypeError('noise_target must be nonnegative')
    return label, eig_min, eig_max, noise_target


def _worker(args):
    (scenario_label, eig_min, eig_max, noise_target, block_args) = args
    rows = _block_worker(block_args)

    diagnostic = next((r for r in rows if 'max_rho' in r), {})
    for row in rows:
        row.update({
            'scenario': scenario_label,
            'eig_min': float(eig_min),
            'eig_max': float(eig_max),
            'noise_target': float(noise_target),
            'max_a_norm': diagnostic.get('max_a_norm', np.nan),
            'max_rho': diagnostic.get('max_rho', np.nan),
            'warn_unstable': diagnostic.get('warn_unstable', False),
            'warn_assumption': diagnostic.get('warn_assumption', False),
        })
    return rows


def _aggregate(rows):
    df = pd.DataFrame(rows)
    group_cols = [
        'scenario', 'eig_min', 'eig_max', 'noise_target',
        'T', 'eta', 'b_n', 'estimator',
    ]
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
        max_a_norm_median=('max_a_norm', 'median'),
        max_rho_median=('max_rho', 'median'),
        warn_unstable_rate=('warn_unstable', 'mean'),
        warn_assumption_rate=('warn_assumption', 'mean'),
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
                   rr_alphas=(0.2, 0.1), bn_exps=None, lam=2.0,
                   scenarios=None,
                   out='results/stress/rr_conditioning_noise_stress.csv'):
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
    print(f"RR conditioning/noise stress sweep: {n_problems} problems x "
          f"{n_traj} traj, T={T_values}, workers={n_workers}")
    print(f"  RR alphas: {rr_alphas}")
    print(f"  bn_exps: {bn_exps}, lam={lam}")
    print(f"  Direction: {dir_desc}")
    print("  Scenarios:")
    for label, eig_min, eig_max, noise_target in scenarios:
        print(f"    {label}: eig=[{eig_min},{eig_max}], "
              f"noise={noise_target}")
    print(flush=True)

    rng_master = np.random.default_rng(seed)
    seeds = [int(rng_master.integers(0, 2**31)) for _ in range(n_problems)]

    all_rows = []
    t_all = time.time()
    for scenario_label, eig_min, eig_max, noise_target in scenarios:
        print(f"\nScenario {scenario_label}: eig=[{eig_min},{eig_max}], "
              f"noise={noise_target}", flush=True)

        for T in T_values:
            K = max(int(T ** 0.3), 5)
            burn_in = min(1000, T // 10)
            b_grid = _make_b_grid(T, bn_exps, lam)
            print(f"  T={T}: K={K}, burn_in={burn_in}, b_grid={b_grid}",
                  flush=True)

            task_args = []
            for s in seeds:
                block_args = (
                    s, n_traj, T, n_states, d, K, burn_in, b_grid, lam,
                    direction_coord, eig_min, eig_max, noise_target,
                    tuple(rr_alphas),
                )
                task_args.append(
                    (scenario_label, eig_min, eig_max, noise_target,
                     block_args)
                )

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

    print(f"\n{'=' * 122}")
    print(f"RESULTS ({n_problems} problems, T={T_values}, {n_traj} traj) "
          f"in {total:.0f}s ({total / 60:.1f}min)")
    print("Median over problems. Width and L2 are x 1e-3.")
    print("=" * 122)
    header = (
        f"{'scenario':<16} {'T':>9} {'eta':>5} {'est':>8} {'L2':>8} "
        f"{'Width':>8} {'Cov med':>8} {'W/orcl':>8} {'bias':>9} "
        f"{'neg%':>7} {'rho':>7} {'warn%':>7}"
    )
    print(header)
    print("-" * len(header))
    for _, row in agg.iterrows():
        eta_str = '-' if np.isnan(row['eta']) else f"{row['eta']:.1f}"
        print(
            f"{row['scenario']:<16} {int(row['T']):>9} {eta_str:>5} "
            f"{row['estimator']:>8} {row['l2_median_x1e3']:>8.2f} "
            f"{row['width_median_x1e3']:>8.2f} "
            f"{row['coverage_median_pct']:>8.1f} "
            f"{row['width_ratio_to_oracle']:>8.3f} "
            f"{row['rel_bias_raw_median']:>9.3f} "
            f"{100 * row['negative_rate_mean']:>7.2f} "
            f"{row['max_rho_median']:>7.3f} "
            f"{100 * row['warn_unstable_rate']:>7.1f}"
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
    parser = argparse.ArgumentParser(
        description="RR conditioning/noise stress sweep",
    )
    parser.add_argument('--n-problems', type=int, default=10)
    parser.add_argument('--n-traj', type=int, default=50)
    parser.add_argument('--T', type=int, nargs='+', default=[100000])
    parser.add_argument('--n-states', type=int, default=10)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-workers', type=int, default=None)
    parser.add_argument('--direction-coord', type=int, default=None)
    parser.add_argument('--rr-alphas', type=float, nargs=2,
                        default=[0.2, 0.1])
    parser.add_argument('--bn-exps', type=float, nargs='+',
                        default=[0.4, 0.5, 0.6])
    parser.add_argument('--lam', type=float, default=2.0)
    parser.add_argument(
        '--scenario', action='append', type=_parse_scenario,
        help=('Scenario label:eig_min:eig_max:noise_target. Can be repeated. '
              'Defaults to baseline, weak_mean, high_noise, weak_high_noise.'),
    )
    parser.add_argument('--out', type=str,
                        default='results/stress/'
                                'rr_conditioning_noise_stress.csv')
    args = parser.parse_args()

    run_experiment(
        args.n_problems, args.n_traj, args.T,
        args.n_states, args.d, args.seed,
        args.n_workers, args.direction_coord,
        rr_alphas=tuple(args.rr_alphas),
        bn_exps=args.bn_exps,
        lam=args.lam,
        scenarios=args.scenario,
        out=args.out,
    )


if __name__ == '__main__':
    main()
