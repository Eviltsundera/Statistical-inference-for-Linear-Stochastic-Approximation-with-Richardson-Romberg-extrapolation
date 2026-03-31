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


def run_experiment(n_problems, n_traj, T, n_states, d, seed=42):
    """Run the full comparison experiment."""
    K = max(int(T ** 0.3), 5)
    burn_in = min(1000, T // 10)

    # Samsonov PR parameters: c0, k0, gamma
    # Use gamma=3/4 for optimal Berry-Esseen rate.
    # c0 must be small enough; k0 large enough for stability.
    pr_gamma = 0.75
    pr_c0 = 0.1
    pr_k0 = max(100, int(np.log(T) ** (1 / pr_gamma)))

    # OBM block size: b_n ~ T^{3/4} (Corollary 2 of Samsonov et al.)
    b_n = max(int(T ** 0.75), 10)
    b_n = min(b_n, T // 2)  # safety

    print(f"Experiment config: {n_problems} problems x {n_traj} traj, T={T}, "
          f"d={d}, |X|={n_states}")
    print(f"  Huo: K={K}, burn_in={burn_in}")
    print(f"  Samsonov PR: c0={pr_c0}, k0={pr_k0}, gamma={pr_gamma}, b_n={b_n}")
    print()

    # Accumulators: per-problem averages
    all_results = {m: {'l2': [], 'width': [], 'cov': []}
                   for m in METHODS_ORDER}

    rng_master = np.random.default_rng(seed)

    for prob_idx in range(n_problems):
        t0 = time.time()
        prob_seed = rng_master.integers(0, 2**31)
        rng = np.random.default_rng(prob_seed)

        P, pi, A_bar, theta_star, A_arr, b_arr = generate_problem(n_states, d, rng)

        traj_rng = np.random.default_rng(rng.integers(0, 2**31))
        trajs = simulate_chains_batch(P, pi, T, n_traj, traj_rng)

        boot_rng = np.random.default_rng(rng.integers(0, 2**31))
        results = run_all_methods(
            A_arr, b_arr, trajs, K, burn_in, theta_star, T,
            pr_c0, pr_k0, pr_gamma, b_n, boot_rng
        )

        for m in METHODS_ORDER:
            for metric in ('l2', 'width', 'cov'):
                all_results[m][metric].append(float(np.nanmean(results[m][metric])))

        elapsed = time.time() - t0
        if (prob_idx + 1) % max(1, n_problems // 10) == 0 or prob_idx == 0:
            print(f"  Problem {prob_idx + 1}/{n_problems} done in {elapsed:.1f}s")

    # Build summary table: mean over problems
    print("\n" + "=" * 80)
    print("RESULTS: Mean over problems (coverage in %, L2 and CI width x 1e-3)")
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

    # Percentile table for coverage and bias
    print("\n" + "=" * 80)
    print("COVERAGE PERCENTILES (%) across problems")
    print("=" * 80)
    pcts = [10, 25, 50, 75, 90]
    header = f"{'Method':<25}" + "".join(f"{'p'+str(p):>8}" for p in pcts)
    print(header)
    print("-" * (25 + 8 * len(pcts)))
    for m in METHODS_ORDER:
        vals = np.array(all_results[m]['cov']) * 100
        ps = np.percentile(vals, pcts)
        print(f"{METHOD_LABELS[m]:<25}" + "".join(f"{v:>8.1f}" for v in ps))

    print("\n" + "=" * 80)
    print("BIAS (L2 x 1e-3) PERCENTILES across problems")
    print("=" * 80)
    print(header)
    print("-" * (25 + 8 * len(pcts)))
    for m in METHODS_ORDER:
        vals = np.array(all_results[m]['l2']) * 1e3
        ps = np.percentile(vals, pcts)
        print(f"{METHOD_LABELS[m]:<25}" + "".join(f"{v:>8.2f}" for v in ps))

    # Save CSV
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
    args = parser.parse_args()

    run_experiment(args.n_problems, args.n_traj, args.T,
                   args.n_states, args.d, args.seed)


if __name__ == '__main__':
    main()
