"""Sweep NOBM vs OBM confidence intervals over stepsize pairs and block sizes.

Hypothesis check: the near-equivalence of NOBM and OBM observed at the
production settings (pair (0.20, 0.10), b_n = T^0.6) may break down at other
stepsizes (slower mixing of the iterate process for small alpha) or other
block sizes (few blocks -> noisy NOBM variance estimate).

For each problem the Markov trajectories are simulated once and shared by
all RR stepsize pairs; for each pair the RR projection is computed once and
then evaluated with the analytic oracle variance and with NOBM/OBM at every
block-size exponent gamma (b_n = floor(T^gamma)).  Thus NOBM and OBM are
compared pairwise on identical data.
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
from lsa_inference.inference import nobm_ci, obm_ci, oracle_ci


METHOD_LABELS = {
    'RR_ORACLE': 'RR + oracle variance',
    'RR_NOBM': 'RR + NOBM',
    'RR_OBM': 'RR + OBM',
}


def _summarize(method, l2, width, cov, theta_bar):
    diverged = int(np.sum(np.any(np.isnan(theta_bar), axis=1)))
    return {
        'method': method,
        'label': METHOD_LABELS[method],
        'l2_mean_x1e3': float(np.nanmean(l2)) * 1e3,
        'width_mean_x1e3': float(np.nanmean(width)) * 1e3,
        'coverage_pct': float(np.nanmean(cov)) * 100,
        'diverged': diverged,
    }


def _worker(args):
    (prob_seed, n_traj, T, n_states, d, K, burn_in, gammas,
     direction_coord, eig_min, eig_max, noise_target, pairs) = args

    # Same RNG consumption order as run_oracle_variance_comparison.py, so the
    # problems, directions, and trajectories match the earlier runs.
    rng = np.random.default_rng(prob_seed)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, _ = generate_A(
        n_states, d, pi, rng,
        eig_min=eig_min, eig_max=eig_max, noise_target=noise_target,
    )
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)
    A_arr, b_arr = prepare_arrays(A_list, b_list)

    if direction_coord is None:
        u = rng.standard_normal(d)
        direction = u / np.linalg.norm(u)
    else:
        direction = np.eye(d)[direction_coord]

    sigma_true = compute_asymptotic_variance(
        A_list, b_list, P, pi, theta_star, direction,
    )

    traj_rng = np.random.default_rng(rng.integers(0, 2**31))
    rng.integers(0, 2**31)  # skip the boot_rng draw of the oracle runner
    trajs = simulate_chains_batch(P, pi, T, n_traj, traj_rng)

    rows = []
    for pair in pairs:
        diagnostics = problem_diagnostics(A_list, alpha_warn=max(pair))
        rr_proj, rr_theta_bar, _, _, _, _, _ = run_rr_full(
            A_arr, b_arr, trajs, list(pair), K, burn_in,
            direction=direction,
        )
        n_eff = rr_proj.shape[1]

        base = {
            'problem_seed': prob_seed,
            'sigma_true': float(sigma_true),
            'rr_alpha_1': float(pair[0]),
            'rr_alpha_2': float(pair[1]),
            'T': int(T),
            'T_eff': int(n_eff),
            'n_traj': int(n_traj),
            'max_rho': diagnostics['max_rho'],
            'warn_unstable': diagnostics['warn_unstable'],
        }

        l2, width, cov = oracle_ci(
            rr_theta_bar, theta_star, sigma_true, n_eff, direction=direction,
        )
        row = _summarize('RR_ORACLE', l2, width, cov, rr_theta_bar)
        row.update(base)
        row.update({'gamma': np.nan, 'b_n': 0, 'n_blocks_nobm': 0})
        rows.append(row)

        for gamma in gammas:
            b_n = max(int(T ** gamma), 10)
            b_n = min(b_n, n_eff // 4)
            for method, fn in (('RR_NOBM', nobm_ci), ('RR_OBM', obm_ci)):
                l2, width, cov = fn(
                    rr_proj, rr_theta_bar, b_n, theta_star,
                    direction=direction,
                )
                row = _summarize(method, l2, width, cov, rr_theta_bar)
                row.update(base)
                row.update({
                    'gamma': float(gamma),
                    'b_n': int(b_n),
                    'n_blocks_nobm': int(n_eff // b_n),
                })
                rows.append(row)

        del rr_proj, rr_theta_bar

    return rows


def _aggregate(rows):
    df = pd.DataFrame(rows)
    agg = df.groupby(
        ['T', 'rr_alpha_1', 'rr_alpha_2', 'gamma', 'method', 'label'],
        sort=False, dropna=False,
    ).agg(
        T_eff=('T_eff', 'first'),
        b_n=('b_n', 'first'),
        n_blocks_nobm=('n_blocks_nobm', 'first'),
        n_problems=('problem_seed', 'nunique'),
        l2_median_x1e3=('l2_mean_x1e3', 'median'),
        width_median_x1e3=('width_mean_x1e3', 'median'),
        coverage_median_pct=('coverage_pct', 'median'),
        coverage_mean_pct=('coverage_pct', 'mean'),
        diverged_total=('diverged', 'sum'),
    ).reset_index()

    oracle = (
        agg.loc[agg['method'] == 'RR_ORACLE']
        .set_index(['T', 'rr_alpha_1'])['width_median_x1e3']
    )
    key = list(zip(agg['T'], agg['rr_alpha_1']))
    agg['width_ratio_to_oracle'] = (
        agg['width_median_x1e3'] / oracle.loc[key].to_numpy()
    )
    return df, agg


def run_experiment(n_problems, n_traj, T_values, n_states, d, seed=42,
                   n_workers=None, direction_coord=None,
                   eig_min=0.25, eig_max=0.60, noise_target=0.35,
                   pairs=((0.02, 0.01), (0.04, 0.02),
                          (0.10, 0.05), (0.20, 0.10)),
                   gammas=(0.4, 0.5, 0.6, 0.7, 0.8),
                   out='results/nobm_obm_sweep/nobm_obm_sweep.csv'):
    if isinstance(T_values, (int, np.integer)):
        T_values = [int(T_values)]
    else:
        T_values = [int(T) for T in T_values]

    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_problems)

    print(f"NOBM vs OBM sweep: {n_problems} problems x {n_traj} traj, "
          f"T={T_values}, d={d}, |X|={n_states}, workers={n_workers}")
    print(f"  RR pairs: {list(pairs)}")
    print(f"  Block exponents gamma: {list(gammas)}")
    print(f"  Problem gen: eig=[{eig_min},{eig_max}], noise={noise_target}")
    print(flush=True)

    rng_master = np.random.default_rng(seed)
    seeds = [int(rng_master.integers(0, 2**31)) for _ in range(n_problems)]
    all_rows = []
    t_all_start = time.time()

    for T in T_values:
        K = max(int(T ** 0.3), 5)
        burn_in = min(1000, T // 10)
        task_args = [
            (s, n_traj, T, n_states, d, K, burn_in, tuple(gammas),
             direction_coord, eig_min, eig_max, noise_target,
             tuple(tuple(p) for p in pairs))
            for s in seeds
        ]

        print(f"\nT={T}: burn_in={burn_in}", flush=True)
        t_start = time.time()
        completed = 0
        with mp.Pool(n_workers) as pool:
            for rows in pool.imap_unordered(_worker, task_args):
                completed += 1
                all_rows.extend(rows)
                if completed % max(1, n_problems // 10) == 0 or completed == 1:
                    elapsed = time.time() - t_start
                    eta = elapsed / completed * (n_problems - completed)
                    print(f"  T={T} [{completed}/{n_problems}] "
                          f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                          flush=True)

        print(f"Finished T={T} in {time.time() - t_start:.0f}s", flush=True)

    t_total = time.time() - t_all_start
    df, agg = _aggregate(all_rows)

    print(f"\n{'=' * 100}")
    print(f"RESULTS ({n_problems} problems, T={T_values}, {n_traj} traj) "
          f"in {t_total:.0f}s ({t_total / 60:.1f}min)")
    print("Median over problems (coverage in %, width x 1e-3)")
    print("=" * 100)
    for (T, a1), grp in agg.groupby(['T', 'rr_alpha_1'], sort=True):
        a2 = grp['rr_alpha_2'].iloc[0]
        orc = grp.loc[grp['method'] == 'RR_ORACLE'].iloc[0]
        print(f"\nT={T}, pair=({a1}, {a2}): oracle width="
              f"{orc['width_median_x1e3']:.2f}, cov med "
              f"{orc['coverage_median_pct']:.1f}%, cov mean "
              f"{orc['coverage_mean_pct']:.1f}%")
        header = (f"  {'gamma':>5} {'b_n':>7} {'K_nobm':>7} "
                  f"{'NOBM cov':>9} {'OBM cov':>8} "
                  f"{'NOBM mean':>10} {'OBM mean':>9} "
                  f"{'NOBM w/o':>9} {'OBM w/o':>8}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        sub = grp.loc[grp['method'] != 'RR_ORACLE']
        for gamma, gg in sub.groupby('gamma', sort=True):
            nb = gg.loc[gg['method'] == 'RR_NOBM'].iloc[0]
            ob = gg.loc[gg['method'] == 'RR_OBM'].iloc[0]
            print(f"  {gamma:>5.1f} {int(nb['b_n']):>7} "
                  f"{int(nb['n_blocks_nobm']):>7} "
                  f"{nb['coverage_median_pct']:>9.1f} "
                  f"{ob['coverage_median_pct']:>8.1f} "
                  f"{nb['coverage_mean_pct']:>10.2f} "
                  f"{ob['coverage_mean_pct']:>9.2f} "
                  f"{nb['width_ratio_to_oracle']:>9.3f} "
                  f"{ob['width_ratio_to_oracle']:>8.3f}")

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
        description="Sweep NOBM vs OBM over stepsize pairs and block sizes",
    )
    parser.add_argument('--n-problems', type=int, default=10)
    parser.add_argument('--n-traj', type=int, default=50)
    parser.add_argument('--T', type=int, nargs='+', default=[20000])
    parser.add_argument('--n-states', type=int, default=10)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-workers', type=int, default=None)
    parser.add_argument('--direction-coord', type=int, default=None)
    parser.add_argument('--eig-min', type=float, default=0.25)
    parser.add_argument('--eig-max', type=float, default=0.60)
    parser.add_argument('--noise-target', type=float, default=0.35)
    parser.add_argument('--pairs', type=float, nargs='+',
                        default=[0.02, 0.01, 0.04, 0.02,
                                 0.10, 0.05, 0.20, 0.10],
                        help="Flat list of (2*alpha, alpha) pairs.")
    parser.add_argument('--gammas', type=float, nargs='+',
                        default=[0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument('--out', type=str,
                        default='results/nobm_obm_sweep/nobm_obm_sweep.csv')
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        parser.error("--pairs needs an even number of values")
    pairs = [(args.pairs[i], args.pairs[i + 1])
             for i in range(0, len(args.pairs), 2)]

    run_experiment(
        args.n_problems, args.n_traj, args.T,
        args.n_states, args.d, args.seed,
        args.n_workers, args.direction_coord,
        eig_min=args.eig_min, eig_max=args.eig_max,
        noise_target=args.noise_target,
        pairs=pairs, gammas=tuple(args.gammas),
        out=args.out,
    )


if __name__ == '__main__':
    main()
