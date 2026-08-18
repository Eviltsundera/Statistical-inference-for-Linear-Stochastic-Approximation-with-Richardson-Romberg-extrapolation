"""Sweep the Huo et al. (2023) batch-mean estimator over K and intra-batch burn-in.

The Huo estimator differs from the rest of the pipeline in its *center*:
after the global burn-in the trajectory is split into K equal batches, the
first n0 iterates of every batch are discarded, theta is estimated by the
batch mean, and the point estimate is the average of the K batch means.
The variance is estimated from the same windows:

    sigma^2 = ((n - n0) / K) * sum_k (bm_k - center)^2
    CI      = center +/- z * sqrt(sigma^2 / (K * (n - n0)))

(see `code/docs/huo2023_experiment_spec.md`; `batch_mean_ci` implements the
same formulas on engine-side batch means with n0 = 0).

For each problem the Markov trajectories are simulated once (same RNG
consumption order as run_oracle_variance_comparison.py, so problems,
directions, and trajectories match the earlier runs); for each RR stepsize
pair the RR projection is computed once and the Huo estimator is evaluated
post-hoc at every (K, n0/n) configuration.  Reference rows: the analytic
oracle CI and the production OBM CI (b_n = T^0.6), both centered at the
plain post-burn-in average.  Center quality is compared by the absolute
error along the CI direction (the full d-dim L2 is not recoverable from
the projection).
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
from lsa_inference.inference import obm_ci, oracle_ci


METHOD_LABELS = {
    'RR_ORACLE': 'RR + oracle variance',
    'RR_OBM': 'RR + OBM (b=T^0.6)',
    'RR_HUO': 'RR + Huo batch means',
}


def _huo_ci_from_proj(proj, K, n0, star_proj, q=0.05):
    """Huo batch-mean CI from the post-burn-in projection.

    Returns (err, width, cov, n) where err is the absolute error of the
    mean-of-batch-means center along the projection direction and n is the
    batch size.  The tail shorter than a batch is dropped; the first n0
    points of every batch are discarded.
    """
    z = stats.norm.ppf(1 - q / 2)
    n_traj, T_eff = proj.shape
    n = T_eff // K

    bm = np.empty((n_traj, K))
    for k in range(K):
        bm[:, k] = proj[:, k * n + n0:(k + 1) * n].mean(axis=1)

    center = bm.mean(axis=1)
    diffs = bm - center[:, None]
    # se = sqrt( ((n-n0)/K) * sum diffs^2 / (K*(n-n0)) ) = sqrt(sum diffs^2)/K
    se = np.sqrt(np.sum(diffs ** 2, axis=1)) / K

    err = np.abs(center - star_proj)
    width = 2 * z * se
    lo = center - z * se
    hi = center + z * se
    cov = ((lo <= star_proj) & (star_proj <= hi)).astype(float)
    cov[np.isnan(center)] = 0.0
    return err, width, cov, n


def _summarize(method, err, width, cov, n_diverged):
    return {
        'method': method,
        'label': METHOD_LABELS[method],
        'err_dir_mean_x1e3': float(np.nanmean(err)) * 1e3,
        'width_mean_x1e3': float(np.nanmean(width)) * 1e3,
        'coverage_pct': float(np.nanmean(cov)) * 100,
        'diverged': int(n_diverged),
    }


def _worker(args):
    (prob_seed, n_traj, T, n_states, d, K_engine, burn_in, K_values,
     n0_fracs, direction_coord, eig_min, eig_max, noise_target, pairs) = args

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

    star_proj = float(theta_star @ direction)
    rows = []
    for pair in pairs:
        diagnostics = problem_diagnostics(A_list, alpha_warn=max(pair))
        rr_proj, rr_theta_bar, _, _, _, _, _ = run_rr_full(
            A_arr, b_arr, trajs, list(pair), K_engine, burn_in,
            direction=direction,
        )
        n_eff = rr_proj.shape[1]
        bar_proj = rr_theta_bar @ direction
        tail_err = np.abs(bar_proj - star_proj)
        n_diverged = int(np.sum(np.isnan(bar_proj)))

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

        _, width, cov = oracle_ci(
            rr_theta_bar, theta_star, sigma_true, n_eff, direction=direction,
        )
        row = _summarize('RR_ORACLE', tail_err, width, cov, n_diverged)
        row.update(base)
        row.update({'K': 0, 'n_batch': 0, 'n0': 0, 'n0_frac': np.nan})
        rows.append(row)

        b_n = min(max(int(T ** 0.6), 10), n_eff // 4)
        _, width, cov = obm_ci(
            rr_proj, rr_theta_bar, b_n, theta_star, direction=direction,
        )
        row = _summarize('RR_OBM', tail_err, width, cov, n_diverged)
        row.update(base)
        row.update({'K': 0, 'n_batch': int(b_n), 'n0': 0, 'n0_frac': np.nan})
        rows.append(row)

        for K in K_values:
            n = n_eff // K
            for frac in n0_fracs:
                n0 = int(round(frac * n))
                if n - n0 < 2:
                    continue
                err, width, cov, _ = _huo_ci_from_proj(
                    rr_proj, K, n0, star_proj,
                )
                row = _summarize('RR_HUO', err, width, cov, n_diverged)
                row.update(base)
                row.update({
                    'K': int(K),
                    'n_batch': int(n),
                    'n0': int(n0),
                    'n0_frac': float(frac),
                })
                rows.append(row)

        del rr_proj, rr_theta_bar

    return rows


def _aggregate(rows):
    df = pd.DataFrame(rows)
    agg = df.groupby(
        ['T', 'rr_alpha_1', 'rr_alpha_2', 'method', 'label', 'K', 'n0_frac'],
        sort=False, dropna=False,
    ).agg(
        T_eff=('T_eff', 'first'),
        n_batch=('n_batch', 'first'),
        n0=('n0', 'first'),
        n_problems=('problem_seed', 'nunique'),
        err_dir_median_x1e3=('err_dir_mean_x1e3', 'median'),
        width_median_x1e3=('width_mean_x1e3', 'median'),
        coverage_median_pct=('coverage_pct', 'median'),
        coverage_mean_pct=('coverage_pct', 'mean'),
        diverged_total=('diverged', 'sum'),
    ).reset_index()

    oracle = (
        agg.loc[agg['method'] == 'RR_ORACLE']
        .set_index(['T', 'rr_alpha_1'])
    )
    key = list(zip(agg['T'], agg['rr_alpha_1']))
    agg['width_ratio_to_oracle'] = (
        agg['width_median_x1e3']
        / oracle['width_median_x1e3'].loc[key].to_numpy()
    )
    agg['err_ratio_to_tail'] = (
        agg['err_dir_median_x1e3']
        / oracle['err_dir_median_x1e3'].loc[key].to_numpy()
    )
    return df, agg


def run_experiment(n_problems, n_traj, T_values, n_states, d, seed=42,
                   n_workers=None, direction_coord=None,
                   eig_min=0.25, eig_max=0.60, noise_target=0.35,
                   pairs=((0.02, 0.01), (0.04, 0.02),
                          (0.10, 0.05), (0.20, 0.10)),
                   K_extra=(50, 100),
                   n0_fracs=(0.0, 0.1, 0.25, 0.5),
                   out='results/huo_bm_sweep/huo_bm_sweep.csv'):
    if isinstance(T_values, (int, np.integer)):
        T_values = [int(T_values)]
    else:
        T_values = [int(T) for T in T_values]

    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_problems)

    print(f"Huo batch-mean sweep: {n_problems} problems x {n_traj} traj, "
          f"T={T_values}, d={d}, |X|={n_states}, workers={n_workers}")
    print(f"  RR pairs: {list(pairs)}")
    print(f"  K grid: floor(T^0.3) + {list(K_extra)}, "
          f"n0/n grid: {list(n0_fracs)}")
    print(f"  Problem gen: eig=[{eig_min},{eig_max}], noise={noise_target}")
    print(flush=True)

    rng_master = np.random.default_rng(seed)
    seeds = [int(rng_master.integers(0, 2**31)) for _ in range(n_problems)]
    all_rows = []
    t_all_start = time.time()

    for T in T_values:
        K_engine = max(int(T ** 0.3), 5)
        K_values = tuple(sorted({K_engine, *K_extra}))
        burn_in = min(1000, T // 10)
        task_args = [
            (s, n_traj, T, n_states, d, K_engine, burn_in, K_values,
             tuple(n0_fracs), direction_coord, eig_min, eig_max,
             noise_target, tuple(tuple(p) for p in pairs))
            for s in seeds
        ]

        print(f"\nT={T}: burn_in={burn_in}, K grid={list(K_values)}",
              flush=True)
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
    print("Median over problems (coverage in %, width and err x 1e-3)")
    print("=" * 100)
    for (T, a1), grp in agg.groupby(['T', 'rr_alpha_1'], sort=True):
        a2 = grp['rr_alpha_2'].iloc[0]
        orc = grp.loc[grp['method'] == 'RR_ORACLE'].iloc[0]
        obm = grp.loc[grp['method'] == 'RR_OBM'].iloc[0]
        print(f"\nT={T}, pair=({a1}, {a2}): tail err="
              f"{orc['err_dir_median_x1e3']:.2f}, oracle width="
              f"{orc['width_median_x1e3']:.2f} cov "
              f"{orc['coverage_mean_pct']:.1f}%, OBM(b=T^0.6) width="
              f"{obm['width_median_x1e3']:.2f} cov "
              f"{obm['coverage_mean_pct']:.1f}%")
        header = (f"  {'K':>4} {'n':>7} {'n0':>6} "
                  f"{'err':>7} {'err/tail':>8} "
                  f"{'width':>7} {'w/orc':>6} "
                  f"{'cov med':>8} {'cov mean':>9}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        sub = grp.loc[grp['method'] == 'RR_HUO']
        for _, r in sub.sort_values(['K', 'n0_frac']).iterrows():
            print(f"  {int(r['K']):>4} {int(r['n_batch']):>7} "
                  f"{int(r['n0']):>6} "
                  f"{r['err_dir_median_x1e3']:>7.2f} "
                  f"{r['err_ratio_to_tail']:>8.3f} "
                  f"{r['width_median_x1e3']:>7.2f} "
                  f"{r['width_ratio_to_oracle']:>6.3f} "
                  f"{r['coverage_median_pct']:>8.1f} "
                  f"{r['coverage_mean_pct']:>9.2f}")

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
        description="Sweep the Huo batch-mean estimator over K and "
                    "intra-batch burn-in n0",
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
    parser.add_argument('--K-extra', type=int, nargs='+', default=[50, 100],
                        help="Batch counts tried besides floor(T^0.3).")
    parser.add_argument('--n0-fracs', type=float, nargs='+',
                        default=[0.0, 0.1, 0.25, 0.5],
                        help="Intra-batch discard as a fraction of n.")
    parser.add_argument('--out', type=str,
                        default='results/huo_bm_sweep/huo_bm_sweep.csv')
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
        pairs=pairs, K_extra=tuple(args.K_extra),
        n0_fracs=tuple(args.n0_fracs),
        out=args.out,
    )


if __name__ == '__main__':
    main()
