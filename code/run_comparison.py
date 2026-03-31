"""Unified comparison of inference strategies for Markovian LSA.

Methods compared:
  From Huo et al. 2023:
    1. Constant step alpha=0.2  + batch-mean CI
    2. Constant step alpha=0.02 + batch-mean CI
    3. RR extrapolation (0.2 + 0.02) + batch-mean CI
    4. Diminishing step 0.2/sqrt(k) + CLTZ20 batch-mean CI
    5. Diminishing step 0.02/sqrt(k) + CLTZ20 batch-mean CI

  From Samsonov et al. 2025:
    6. Diminishing step (PR averaging) + OBM CI
    7. Diminishing step (PR averaging) + MSB bootstrap CI

Usage:
    python run_comparison.py                     # Quick smoke test
    python run_comparison.py --n-problems 100 --n-traj 100 --T 100000
"""

import argparse
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

from lsa_inference.markov_chain import generate_transition_matrix, simulate_chains_batch
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.lsa_engine import (
    prepare_arrays, run_lsa_const, run_lsa_diminishing,
    run_lsa_polyak_ruppert, run_rr,
)
from lsa_inference.inference import batch_mean_ci, obm_ci, msb_ci


def generate_problem(n_states, d, rng):
    """Generate a complete LSA problem instance."""
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)
    A_arr, b_arr = prepare_arrays(A_list, b_list)
    return P, pi, A_bar, theta_star, A_arr, b_arr


def run_all_methods(A_arr, b_arr, trajs, K, burn_in, theta_star, T,
                    pr_c0, pr_k0, pr_gamma, b_n, rng):
    """Run all 7 methods on the same trajectories and return metrics.

    Returns:
        dict {method_name: {'l2': ndarray, 'width': ndarray, 'cov': ndarray}}
    """
    results = {}

    # --- Huo et al. 2023 methods ---

    # 1. Constant alpha=0.2
    bm, n = run_lsa_const(A_arr, b_arr, trajs, 0.2, K, burn_in)
    l2, w, c = batch_mean_ci(bm, n, theta_star)
    results['const_0.2'] = {'l2': l2, 'width': w, 'cov': c}

    # 2. Constant alpha=0.02
    bm, n = run_lsa_const(A_arr, b_arr, trajs, 0.02, K, burn_in)
    l2, w, c = batch_mean_ci(bm, n, theta_star)
    results['const_0.02'] = {'l2': l2, 'width': w, 'cov': c}

    # 3. RR extrapolation
    rr_bm, n = run_rr(A_arr, b_arr, trajs, [0.2, 0.02], K, burn_in)
    l2, w, c = batch_mean_ci(rr_bm, n, theta_star)
    results['RR'] = {'l2': l2, 'width': w, 'cov': c}

    # 4. Diminishing 0.2/sqrt(k)
    bm, n_eff = run_lsa_diminishing(A_arr, b_arr, trajs, 0.2, 0.5, K)
    l2, w, c = batch_mean_ci(bm, n_eff, theta_star)
    results['dim_0.2'] = {'l2': l2, 'width': w, 'cov': c}

    # 5. Diminishing 0.02/sqrt(k)
    bm, n_eff = run_lsa_diminishing(A_arr, b_arr, trajs, 0.02, 0.5, K)
    l2, w, c = batch_mean_ci(bm, n_eff, theta_star)
    results['dim_0.02'] = {'l2': l2, 'width': w, 'cov': c}

    # --- Samsonov et al. 2025 methods ---

    # 6 & 7. Polyak-Ruppert with OBM and MSB
    all_thetas, theta_bar = run_lsa_polyak_ruppert(
        A_arr, b_arr, trajs, pr_c0, pr_k0, pr_gamma
    )

    # 6. OBM CI
    l2, w, c = obm_ci(all_thetas, theta_bar, b_n, theta_star)
    results['PR_OBM'] = {'l2': l2, 'width': w, 'cov': c}

    # 7. MSB bootstrap CI
    l2, w, c = msb_ci(all_thetas, theta_bar, b_n, theta_star, rng=rng)
    results['PR_MSB'] = {'l2': l2, 'width': w, 'cov': c}

    return results


METHOD_LABELS = {
    'const_0.2':  'alpha=0.2 (const)',
    'const_0.02': 'alpha=0.02 (const)',
    'RR':         'RR (0.2+0.02)',
    'dim_0.2':    '0.2/sqrt(k) (dim)',
    'dim_0.02':   '0.02/sqrt(k) (dim)',
    'PR_OBM':     'PR + OBM CI',
    'PR_MSB':     'PR + MSB bootstrap',
}
METHODS_ORDER = list(METHOD_LABELS.keys())


def _solve_problem_worker(args):
    """Multiprocessing worker: solve one LSA problem with all 7 methods.

    Accepts a single tuple so it works with pool.imap_unordered.
    """
    (prob_seed, n_traj, T, n_states, d,
     K, burn_in, pr_c0, pr_k0, pr_gamma, b_n) = args

    rng = np.random.default_rng(prob_seed)
    P, pi, A_bar, theta_star, A_arr, b_arr = generate_problem(n_states, d, rng)

    traj_rng = np.random.default_rng(rng.integers(0, 2**31))
    trajs = simulate_chains_batch(P, pi, T, n_traj, traj_rng)

    boot_rng = np.random.default_rng(rng.integers(0, 2**31))
    results = run_all_methods(
        A_arr, b_arr, trajs, K, burn_in, theta_star, T,
        pr_c0, pr_k0, pr_gamma, b_n, boot_rng
    )

    # Collapse per-trajectory arrays to per-problem scalars
    summary = {}
    for m in METHODS_ORDER:
        summary[m] = {
            metric: float(np.nanmean(results[m][metric]))
            for metric in ('l2', 'width', 'cov')
        }
    return summary


def run_experiment(n_problems, n_traj, T, n_states, d, seed=42,
                   n_workers=None):
    """Run the full comparison experiment with multiprocessing."""
    K = max(int(T ** 0.3), 5)
    burn_in = min(1000, T // 10)

    pr_gamma = 0.75
    pr_c0 = 0.1
    pr_k0 = max(100, int(np.log(T) ** (1 / pr_gamma)))

    b_n = max(int(T ** 0.75), 10)
    b_n = min(b_n, T // 2)

    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_problems)

    print(f"Experiment config: {n_problems} problems x {n_traj} traj, T={T}, "
          f"d={d}, |X|={n_states}, workers={n_workers}")
    print(f"  Huo: K={K}, burn_in={burn_in}")
    print(f"  Samsonov PR: c0={pr_c0}, k0={pr_k0}, gamma={pr_gamma}, b_n={b_n}")
    print(flush=True)

    rng_master = np.random.default_rng(seed)
    seeds = [int(rng_master.integers(0, 2**31)) for _ in range(n_problems)]

    task_args = [
        (s, n_traj, T, n_states, d, K, burn_in, pr_c0, pr_k0, pr_gamma, b_n)
        for s in seeds
    ]

    all_results = {m: {'l2': [], 'width': [], 'cov': []}
                   for m in METHODS_ORDER}

    t_start = time.time()
    completed = 0

    with mp.Pool(n_workers) as pool:
        for summary in pool.imap_unordered(_solve_problem_worker, task_args):
            completed += 1
            for m in METHODS_ORDER:
                for metric in ('l2', 'width', 'cov'):
                    all_results[m][metric].append(summary[m][metric])

            if completed % max(1, n_problems // 20) == 0 or completed == 1:
                elapsed = time.time() - t_start
                eta = elapsed / completed * (n_problems - completed)
                rr_cov = summary['RR']['cov'] * 100
                print(f"  [{completed}/{n_problems}] "
                      f"last RR cov={rr_cov:.0f}% | "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                      flush=True)

    t_total = time.time() - t_start

    # --- Print results ---
    print(f"\n{'=' * 80}")
    print(f"RESULTS ({n_problems} problems, T={T}, {n_traj} traj) "
          f"in {t_total:.0f}s ({t_total/60:.1f}min)")
    print("Mean over problems (coverage in %, L2 and CI width x 1e-3)")
    print("=" * 80)
    header = f"{'Method':<25} {'L2 x1e-3':>10} {'Width x1e-3':>12} {'Cov %':>8}"
    print(header)
    print("-" * 58)

    rows = []
    for m in METHODS_ORDER:
        l2_mean = np.mean(all_results[m]['l2']) * 1e3
        w_mean = np.mean(all_results[m]['width']) * 1e3
        cov_mean = np.mean(all_results[m]['cov']) * 100
        print(f"{METHOD_LABELS[m]:<25} {l2_mean:>10.2f} {w_mean:>12.2f} {cov_mean:>8.1f}")
        rows.append({
            'method': m, 'label': METHOD_LABELS[m],
            'l2_mean': l2_mean, 'width_mean': w_mean, 'cov_mean': cov_mean
        })

    pcts = [10, 25, 50, 75, 90]
    pct_header = f"{'Method':<25}" + "".join(f"{'p'+str(p):>8}" for p in pcts)

    print(f"\n{'=' * 80}")
    print("COVERAGE PERCENTILES (%) across problems")
    print("=" * 80)
    print(pct_header)
    print("-" * (25 + 8 * len(pcts)))
    for m in METHODS_ORDER:
        vals = np.array(all_results[m]['cov']) * 100
        ps = np.percentile(vals, pcts)
        print(f"{METHOD_LABELS[m]:<25}" + "".join(f"{v:>8.1f}" for v in ps))

    print(f"\n{'=' * 80}")
    print("BIAS (L2 x 1e-3) PERCENTILES across problems")
    print("=" * 80)
    print(pct_header)
    print("-" * (25 + 8 * len(pcts)))
    for m in METHODS_ORDER:
        vals = np.array(all_results[m]['l2']) * 1e3
        ps = np.percentile(vals, pcts)
        print(f"{METHOD_LABELS[m]:<25}" + "".join(f"{v:>8.2f}" for v in ps))

    df = pd.DataFrame(rows)
    df.to_csv('results_comparison.csv', index=False)
    print(f"\nResults saved to results_comparison.csv")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare LSA inference strategies")
    parser.add_argument('--n-problems', type=int, default=10,
                        help='Number of random LSA problems (default: 10)')
    parser.add_argument('--n-traj', type=int, default=50,
                        help='Trajectories per problem (default: 50)')
    parser.add_argument('--T', type=int, default=10000,
                        help='Trajectory length (default: 10000)')
    parser.add_argument('--n-states', type=int, default=10,
                        help='Markov chain states (default: 10)')
    parser.add_argument('--d', type=int, default=5,
                        help='Dimension (default: 5)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Parallel workers (default: cpu_count)')
    args = parser.parse_args()

    run_experiment(args.n_problems, args.n_traj, args.T,
                   args.n_states, args.d, args.seed, args.n_workers)


if __name__ == '__main__':
    main()
