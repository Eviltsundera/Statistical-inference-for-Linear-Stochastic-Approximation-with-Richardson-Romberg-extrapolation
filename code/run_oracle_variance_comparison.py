"""Compare RR confidence intervals with analytic and estimated variances.

The oracle interval uses the finite-state analytic long-run variance

    sigma^2_inf(u) = u^T A_bar^{-1} Gamma_eps A_bar^{-T} u,

computed by ``compute_asymptotic_variance``.  The interval center is the same
RR-averaged constant-step estimator used by the OBM, OBM-RR, and MSB rows.
Thus differences in coverage come from the variance estimator and the normal
approximation, not from a different point estimator.
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
from lsa_inference.inference import (
    batch_mean_ci,
    obm_ci,
    obm_rr_ci,
    oracle_ci,
    msb_ci,
)


METHOD_LABELS = {
    'RR_BATCH': 'RR + batch means',
    'RR_ORACLE': 'RR + oracle variance',
    'RR_OBM': 'RR + OBM',
    'RR_OBM_RR': 'RR + OBM-RR',
    'RR_MSB': 'RR + MSB',
}

METHODS_ORDER = list(METHOD_LABELS.keys())


def _summarize_method(method, label, l2, width, cov, theta_bar):
    """Return per-problem summary metrics for one CI method."""
    diverged = int(np.sum(np.any(np.isnan(theta_bar), axis=1)))
    return {
        'method': method,
        'label': label,
        'l2_mean_x1e3': float(np.nanmean(l2)) * 1e3,
        'width_mean_x1e3': float(np.nanmean(width)) * 1e3,
        'coverage_pct': float(np.nanmean(cov)) * 100,
        'diverged': diverged,
    }


def _worker(args):
    (prob_seed, n_traj, T, n_states, d, K, burn_in, b_n,
     n_bootstrap, direction_coord, eig_min, eig_max, noise_target,
     rr_alphas) = args

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
    boot_rng = np.random.default_rng(rng.integers(0, 2**31))
    trajs = simulate_chains_batch(P, pi, T, n_traj, traj_rng)

    rr_proj, rr_theta_bar, rr_bm, _, _, _, n_batch = run_rr_full(
        A_arr, b_arr, trajs, list(rr_alphas), K, burn_in,
        direction=direction,
    )
    n_eff = rr_proj.shape[1]

    method_results = {}
    method_results['RR_BATCH'] = batch_mean_ci(
        rr_bm, n_batch, theta_star, direction=direction,
    )
    method_results['RR_ORACLE'] = oracle_ci(
        rr_theta_bar, theta_star, sigma_true, n_eff, direction=direction,
    )
    method_results['RR_OBM'] = obm_ci(
        rr_proj, rr_theta_bar, b_n, theta_star, direction=direction,
    )
    method_results['RR_OBM_RR'] = obm_rr_ci(
        rr_proj, rr_theta_bar, b_n, theta_star, direction=direction,
    )
    method_results['RR_MSB'] = msb_ci(
        rr_proj, rr_theta_bar, b_n, theta_star, n_bootstrap=n_bootstrap,
        direction=direction, rng=boot_rng,
    )

    rows = []
    for method in METHODS_ORDER:
        l2, width, cov = method_results[method]
        row = _summarize_method(
            method, METHOD_LABELS[method], l2, width, cov, rr_theta_bar,
        )
        row.update({
            'problem_seed': prob_seed,
            'sigma_true': float(sigma_true),
            'rr_alpha_1': float(rr_alphas[0]),
            'rr_alpha_2': float(rr_alphas[1]),
            'T': int(T),
            'T_eff': int(n_eff),
            'n_traj': int(n_traj),
            'b_n': int(b_n),
            'n_bootstrap': int(n_bootstrap),
            'max_a_norm': diagnostics['max_a_norm'],
            'max_rho': diagnostics['max_rho'],
            'warn_unstable': diagnostics['warn_unstable'],
            'warn_assumption': diagnostics['warn_assumption'],
        })
        rows.append(row)

    return rows


def _aggregate(rows):
    df = pd.DataFrame(rows)
    agg = df.groupby(['method', 'label'], sort=False).agg(
        rr_alpha_1=('rr_alpha_1', 'first'),
        rr_alpha_2=('rr_alpha_2', 'first'),
        T=('T', 'first'),
        T_eff=('T_eff', 'first'),
        n_problems=('problem_seed', 'nunique'),
        n_traj=('n_traj', 'first'),
        b_n=('b_n', 'first'),
        n_bootstrap=('n_bootstrap', 'first'),
        sigma_true_median=('sigma_true', 'median'),
        l2_median_x1e3=('l2_mean_x1e3', 'median'),
        width_median_x1e3=('width_mean_x1e3', 'median'),
        coverage_median_pct=('coverage_pct', 'median'),
        coverage_mean_pct=('coverage_pct', 'mean'),
        diverged_total=('diverged', 'sum'),
    ).reset_index()

    oracle_width = float(
        agg.loc[agg['method'] == 'RR_ORACLE', 'width_median_x1e3'].iloc[0]
    )
    agg['width_ratio_to_oracle'] = agg['width_median_x1e3'] / oracle_width
    return df, agg


def run_experiment(n_problems, n_traj, T, n_states, d, seed=42,
                   n_workers=None, n_bootstrap=500, direction_coord=None,
                   eig_min=0.25, eig_max=0.60, noise_target=0.35,
                   rr_alphas=(0.2, 0.1), out='results/oracle_variance.csv'):
    K = max(int(T ** 0.3), 5)
    burn_in = min(1000, T // 10)
    b_n = max(int(T ** 0.6), 10)
    b_n = min(b_n, T // 4)

    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_problems)

    dir_desc = f"e_{direction_coord}" if direction_coord is not None else "random"
    print(f"Oracle variance comparison: {n_problems} problems x {n_traj} traj, "
          f"T={T}, d={d}, |X|={n_states}, workers={n_workers}")
    print(f"  RR: K={K}, burn_in={burn_in}, rr_alphas={rr_alphas}")
    print(f"  Variance estimators: oracle, OBM, OBM-RR, MSB; "
          f"b_n={b_n}, n_bootstrap={n_bootstrap}")
    print(f"  Problem gen: eig=[{eig_min},{eig_max}], noise={noise_target}")
    print(f"  Direction: {dir_desc}")
    print(flush=True)

    rng_master = np.random.default_rng(seed)
    seeds = [int(rng_master.integers(0, 2**31)) for _ in range(n_problems)]
    task_args = [
        (s, n_traj, T, n_states, d, K, burn_in, b_n,
         n_bootstrap, direction_coord, eig_min, eig_max, noise_target,
         tuple(rr_alphas))
        for s in seeds
    ]

    all_rows = []
    t_start = time.time()
    completed = 0
    with mp.Pool(n_workers) as pool:
        for rows in pool.imap_unordered(_worker, task_args):
            completed += 1
            all_rows.extend(rows)
            if completed % max(1, n_problems // 20) == 0 or completed == 1:
                elapsed = time.time() - t_start
                eta = elapsed / completed * (n_problems - completed)
                rr_oracle = next(r for r in rows if r['method'] == 'RR_ORACLE')
                print(f"  [{completed}/{n_problems}] "
                      f"last oracle cov={rr_oracle['coverage_pct']:.0f}% | "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                      flush=True)

    t_total = time.time() - t_start
    df, agg = _aggregate(all_rows)

    print(f"\n{'=' * 88}")
    print(f"RESULTS ({n_problems} problems, T={T}, {n_traj} traj) "
          f"in {t_total:.0f}s ({t_total / 60:.1f}min)")
    print("Median over problems (coverage in %, L2 and CI width x 1e-3)")
    print("=" * 88)
    header = (f"{'Method':<24} {'L2':>9} {'Width':>9} {'Cov med':>9} "
              f"{'Cov mean':>10} {'W/oracle':>10} {'Div':>7}")
    print(header)
    print("-" * len(header))
    for _, row in agg.iterrows():
        print(f"{row['label']:<24} {row['l2_median_x1e3']:>9.2f} "
              f"{row['width_median_x1e3']:>9.2f} "
              f"{row['coverage_median_pct']:>9.1f} "
              f"{row['coverage_mean_pct']:>10.1f} "
              f"{row['width_ratio_to_oracle']:>10.3f} "
              f"{int(row['diverged_total']):>7}")

    pcts = [10, 25, 50, 75, 90]
    print(f"\n{'=' * 88}")
    print("COVERAGE PERCENTILES (%) across problems")
    print("=" * 88)
    print(f"{'Method':<24}" + "".join(f"{'p' + str(p):>8}" for p in pcts))
    print("-" * (24 + 8 * len(pcts)))
    for method in METHODS_ORDER:
        sub = df[df['method'] == method]
        vals = sub['coverage_pct'].to_numpy()
        ps = np.nanpercentile(vals, pcts)
        print(f"{METHOD_LABELS[method]:<24}" + "".join(f"{v:>8.1f}" for v in ps))

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
        description="Compare RR intervals with oracle and estimated variances",
    )
    parser.add_argument('--n-problems', type=int, default=10)
    parser.add_argument('--n-traj', type=int, default=50)
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--n-states', type=int, default=10)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-workers', type=int, default=None)
    parser.add_argument('--n-bootstrap', type=int, default=500)
    parser.add_argument('--direction-coord', type=int, default=None)
    parser.add_argument('--eig-min', type=float, default=0.25)
    parser.add_argument('--eig-max', type=float, default=0.60)
    parser.add_argument('--noise-target', type=float, default=0.35)
    parser.add_argument('--rr-alphas', type=float, nargs=2,
                        default=[0.2, 0.1])
    parser.add_argument('--out', type=str,
                        default='results/oracle_variance_comparison.csv')
    args = parser.parse_args()

    run_experiment(
        args.n_problems, args.n_traj, args.T,
        args.n_states, args.d, args.seed,
        args.n_workers, args.n_bootstrap, args.direction_coord,
        eig_min=args.eig_min, eig_max=args.eig_max,
        noise_target=args.noise_target,
        rr_alphas=tuple(args.rr_alphas),
        out=args.out,
    )


if __name__ == '__main__':
    main()
