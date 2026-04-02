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
    prepare_arrays, run_lsa_diminishing,
    run_lsa_polyak_ruppert, run_rr_full,
)
from lsa_inference.inference import batch_mean_ci, obm_ci, obm_rr_ci, msb_ci


def generate_problem(n_states, d, rng):
    """Generate a complete LSA problem instance."""
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)
    A_arr, b_arr = prepare_arrays(A_list, b_list)
    return P, pi, A_bar, theta_star, A_arr, b_arr


def run_all_methods(A_arr, b_arr, trajs, K, burn_in, theta_star, T,
                    pr_c0, pr_k0, pr_gamma, b_n, rng, direction,
                    n_bootstrap=500):
    """Run all 9 methods on the same trajectories and return metrics.

    Args:
        direction: (d,) unit vector for CI projection.

    Returns:
        dict {method_name: {'l2': ndarray, 'width': ndarray, 'cov': ndarray}}
    """
    results = {}

    # --- Single run of constant-step LSA for both alphas ---
    (rr_proj, rr_theta_bar, rr_bm,
     per_alpha_proj, per_alpha_bm, per_alpha_bar, n) = run_rr_full(
        A_arr, b_arr, trajs, [0.2, 0.02], K, burn_in, direction=direction
    )

    # --- Huo et al. 2023: methods 1-3 (batch-mean CI) ---

    # 1. Constant alpha=0.2 + batch-mean
    l2, w, c = batch_mean_ci(per_alpha_bm[0], n, theta_star, direction=direction)
    results['const_0.2'] = {'l2': l2, 'width': w, 'cov': c}

    # 2. Constant alpha=0.02 + batch-mean
    l2, w, c = batch_mean_ci(per_alpha_bm[1], n, theta_star, direction=direction)
    results['const_0.02'] = {'l2': l2, 'width': w, 'cov': c}

    # 3. RR + batch-mean
    l2, w, c = batch_mean_ci(rr_bm, n, theta_star, direction=direction)
    results['RR'] = {'l2': l2, 'width': w, 'cov': c}

    # --- Huo et al. 2023: methods 4-5 (diminishing stepsize) ---

    # 4. Diminishing 0.2/sqrt(k)
    bm, n_eff = run_lsa_diminishing(A_arr, b_arr, trajs, 0.2, 0.5, K)
    l2, w, c = batch_mean_ci(bm, n_eff, theta_star, direction=direction)
    results['dim_0.2'] = {'l2': l2, 'width': w, 'cov': c}

    # 5. Diminishing 0.02/sqrt(k)
    bm, n_eff = run_lsa_diminishing(A_arr, b_arr, trajs, 0.02, 0.5, K)
    l2, w, c = batch_mean_ci(bm, n_eff, theta_star, direction=direction)
    results['dim_0.02'] = {'l2': l2, 'width': w, 'cov': c}

    # --- Samsonov et al. 2025: methods 6-7 (PR + OBM/MSB) ---

    pr_proj, pr_theta_bar = run_lsa_polyak_ruppert(
        A_arr, b_arr, trajs, pr_c0, pr_k0, pr_gamma, direction=direction
    )

    # 6. PR + OBM CI
    l2, w, c = obm_ci(pr_proj, pr_theta_bar, b_n, theta_star, direction=direction)
    results['PR_OBM'] = {'l2': l2, 'width': w, 'cov': c}

    # 7. PR + MSB bootstrap CI
    l2, w, c = msb_ci(pr_proj, pr_theta_bar, b_n, theta_star,
                      n_bootstrap=n_bootstrap, direction=direction, rng=rng)
    results['PR_MSB'] = {'l2': l2, 'width': w, 'cov': c}

    # --- Combined: methods 8-9 (RR + OBM/MSB) ---

    # 8. RR + OBM CI
    l2, w, c = obm_ci(rr_proj, rr_theta_bar, b_n, theta_star, direction=direction)
    results['RR_OBM'] = {'l2': l2, 'width': w, 'cov': c}

    # 9. RR + MSB bootstrap CI
    l2, w, c = msb_ci(rr_proj, rr_theta_bar, b_n, theta_star,
                      n_bootstrap=n_bootstrap, direction=direction, rng=rng)
    results['RR_MSB'] = {'l2': l2, 'width': w, 'cov': c}

    # --- NEW: methods 10-13 (constant step + OBM/MSB) ---

    # 10. Constant alpha=0.2 + OBM
    l2, w, c = obm_ci(per_alpha_proj[0], per_alpha_bar[0], b_n, theta_star,
                       direction=direction)
    results['const_0.2_OBM'] = {'l2': l2, 'width': w, 'cov': c}

    # 11. Constant alpha=0.2 + MSB
    l2, w, c = msb_ci(per_alpha_proj[0], per_alpha_bar[0], b_n, theta_star,
                       n_bootstrap=n_bootstrap, direction=direction, rng=rng)
    results['const_0.2_MSB'] = {'l2': l2, 'width': w, 'cov': c}

    # 12. Constant alpha=0.02 + OBM
    l2, w, c = obm_ci(per_alpha_proj[1], per_alpha_bar[1], b_n, theta_star,
                       direction=direction)
    results['const_0.02_OBM'] = {'l2': l2, 'width': w, 'cov': c}

    # 13. Constant alpha=0.02 + MSB
    l2, w, c = msb_ci(per_alpha_proj[1], per_alpha_bar[1], b_n, theta_star,
                       n_bootstrap=n_bootstrap, direction=direction, rng=rng)
    results['const_0.02_MSB'] = {'l2': l2, 'width': w, 'cov': c}

    # --- NEW: methods 14-16 (OBM-RR: RR by block size for variance) ---

    # 14. Constant alpha=0.02 + OBM-RR
    l2, w, c = obm_rr_ci(per_alpha_proj[1], per_alpha_bar[1], b_n, theta_star,
                          direction=direction)
    results['const_0.02_OBM_RR'] = {'l2': l2, 'width': w, 'cov': c}

    # 15. PR + OBM-RR
    l2, w, c = obm_rr_ci(pr_proj, pr_theta_bar, b_n, theta_star,
                          direction=direction)
    results['PR_OBM_RR'] = {'l2': l2, 'width': w, 'cov': c}

    # 16. RR + OBM-RR
    l2, w, c = obm_rr_ci(rr_proj, rr_theta_bar, b_n, theta_star,
                          direction=direction)
    results['RR_OBM_RR'] = {'l2': l2, 'width': w, 'cov': c}

    return results


METHOD_LABELS = {
    'const_0.2':          'alpha=0.2 (const)',
    'const_0.02':         'alpha=0.02 (const)',
    'RR':                 'RR (0.2+0.02)',
    'dim_0.2':            '0.2/sqrt(k) (dim)',
    'dim_0.02':           '0.02/sqrt(k) (dim)',
    'PR_OBM':             'PR + OBM',
    'PR_MSB':             'PR + MSB',
    'RR_OBM':             'RR + OBM',
    'RR_MSB':             'RR + MSB',
    'const_0.2_OBM':      'a=0.2 + OBM',
    'const_0.2_MSB':      'a=0.2 + MSB',
    'const_0.02_OBM':     'a=0.02 + OBM',
    'const_0.02_MSB':     'a=0.02 + MSB',
    'const_0.02_OBM_RR':  'a=0.02 + OBM-RR',
    'PR_OBM_RR':          'PR + OBM-RR',
    'RR_OBM_RR':          'RR + OBM-RR',
}
METHODS_ORDER = list(METHOD_LABELS.keys())


def _count_diverged(arr):
    """Count trajectories where any metric is NaN."""
    return int(np.sum(np.isnan(arr)))


# Maximum trajectories per chunk.  Controls peak memory:
# ~8 bytes * chunk_size * T per projection array, x3 arrays for RR = peak.
# chunk=100, T=10^6 → 100*10^6*8*3 ≈ 2.4 GB per worker.
_CHUNK_SIZE = 100


def _solve_problem_worker(args):
    """Multiprocessing worker: solve one LSA problem with all 9 methods.

    Processes trajectories in chunks of _CHUNK_SIZE to bound memory.
    Returns (summary, divergence_info).
    """
    (prob_seed, n_traj, T, n_states, d,
     K, burn_in, pr_c0, pr_k0, pr_gamma, b_n, n_bootstrap,
     direction_coord) = args

    rng = np.random.default_rng(prob_seed)
    P, pi, A_bar, theta_star, A_arr, b_arr = generate_problem(n_states, d, rng)

    # Generate projection direction: random unit vector or e_k
    if direction_coord is None:
        # Random unit direction (one per problem)
        u = rng.standard_normal(d)
        direction = u / np.linalg.norm(u)
    else:
        direction = np.eye(d)[direction_coord]

    traj_rng = np.random.default_rng(rng.integers(0, 2**31))
    boot_rng = np.random.default_rng(rng.integers(0, 2**31))

    # Accumulate per-trajectory metrics across chunks
    all_metrics = {m: {'l2': [], 'width': [], 'cov': []}
                   for m in METHODS_ORDER}

    for chunk_start in range(0, n_traj, _CHUNK_SIZE):
        chunk_end = min(chunk_start + _CHUNK_SIZE, n_traj)
        chunk_n = chunk_end - chunk_start

        trajs = simulate_chains_batch(P, pi, T, chunk_n, traj_rng)

        results = run_all_methods(
            A_arr, b_arr, trajs, K, burn_in, theta_star, T,
            pr_c0, pr_k0, pr_gamma, b_n, boot_rng, direction, n_bootstrap
        )

        for m in METHODS_ORDER:
            for metric in ('l2', 'width', 'cov'):
                all_metrics[m][metric].append(results[m][metric])

        # Free memory before next chunk
        del trajs, results

    # Concatenate chunks and summarize
    summary = {}
    diverged = {}
    for m in METHODS_ORDER:
        l2 = np.concatenate(all_metrics[m]['l2'])
        width = np.concatenate(all_metrics[m]['width'])
        cov = np.concatenate(all_metrics[m]['cov'])
        n_div = _count_diverged(l2)
        n_ok = n_traj - n_div
        diverged[m] = n_div
        summary[m] = {
            'l2':    float(np.nanmean(l2)) if n_ok > 0 else np.nan,
            'width': float(np.nanmean(width)) if n_ok > 0 else np.nan,
            'cov':   float(np.nanmean(cov)) if n_ok > 0 else 0.0,
        }
    return summary, diverged


def run_experiment(n_problems, n_traj, T, n_states, d, seed=42,
                   n_workers=None, n_bootstrap=500, direction_coord=None):
    """Run the full comparison experiment with multiprocessing.

    Args:
        direction_coord: If None, each problem uses a random unit direction.
            If an int k, uses the basis vector e_k for all problems.
    """
    K = max(int(T ** 0.3), 5)
    burn_in = min(1000, T // 10)

    # Samsonov PR parameters.
    # gamma=2/3 as in paper experiments (Section G, Appendix).
    # c0 and k0 chosen so that initial stepsize alpha_0 = c0/(k0^gamma) ~ 0.2,
    # comparable to the diminishing 0.2/sqrt(k) baseline.
    # Reference notebook: c0=200, k0=20000, gamma=0.65 → alpha_0≈0.32 (for TD, d=2).
    # For our generic LSA (d=5): c0=1.0, k0=10 → alpha_0=1/(10^{2/3})≈0.22.
    pr_gamma = 2 / 3
    pr_c0 = 1.0
    pr_k0 = 10

    # OBM block size: b_n ~ T^{0.6}.
    # Paper Table 2: b_n=250 for T=20480, b_n=1200 for T=204800, b_n=3600 for T=1024000
    # — consistent with T^{0.6} scaling.
    # Previous T^{0.75} was too large, causing OBM to oversmooth → underestimate variance.
    b_n = max(int(T ** 0.6), 10)
    b_n = min(b_n, T // 4)

    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_problems)

    dir_desc = f"e_{direction_coord}" if direction_coord is not None else "random"
    print(f"Experiment config: {n_problems} problems x {n_traj} traj, T={T}, "
          f"d={d}, |X|={n_states}, workers={n_workers}")
    print(f"  Huo: K={K}, burn_in={burn_in}")
    print(f"  Samsonov PR: c0={pr_c0}, k0={pr_k0}, gamma={pr_gamma}, "
          f"b_n={b_n}, n_bootstrap={n_bootstrap}")
    print(f"  Direction: {dir_desc}")
    print(flush=True)

    rng_master = np.random.default_rng(seed)
    seeds = [int(rng_master.integers(0, 2**31)) for _ in range(n_problems)]

    task_args = [
        (s, n_traj, T, n_states, d, K, burn_in, pr_c0, pr_k0, pr_gamma, b_n,
         n_bootstrap, direction_coord)
        for s in seeds
    ]

    all_results = {m: {'l2': [], 'width': [], 'cov': []}
                   for m in METHODS_ORDER}
    all_diverged = {m: [] for m in METHODS_ORDER}

    t_start = time.time()
    completed = 0

    with mp.Pool(n_workers) as pool:
        for summary, diverged in pool.imap_unordered(_solve_problem_worker, task_args):
            completed += 1
            for m in METHODS_ORDER:
                for metric in ('l2', 'width', 'cov'):
                    all_results[m][metric].append(summary[m][metric])
                all_diverged[m].append(diverged[m])

            if completed % max(1, n_problems // 20) == 0 or completed == 1:
                elapsed = time.time() - t_start
                eta = elapsed / completed * (n_problems - completed)
                rr_cov = summary['RR']['cov'] * 100
                rr_div = diverged['RR']
                print(f"  [{completed}/{n_problems}] "
                      f"last RR cov={rr_cov:.0f}% div={rr_div}/{n_traj} | "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                      flush=True)

    t_total = time.time() - t_start

    # --- Divergence summary ---
    print(f"\n{'=' * 80}")
    print(f"DIVERGENCE REPORT ({n_problems} problems x {n_traj} traj)")
    print("=" * 80)
    div_header = (f"{'Method':<25} {'Total div':>10} {'Problems w/div':>15} "
                  f"{'Max div/prob':>14} {'Mean div/prob':>14}")
    print(div_header)
    print("-" * 80)
    for m in METHODS_ORDER:
        divs = np.array(all_diverged[m])
        total = int(np.sum(divs))
        n_probs_div = int(np.sum(divs > 0))
        max_div = int(np.max(divs)) if len(divs) > 0 else 0
        mean_div = float(np.mean(divs))
        print(f"{METHOD_LABELS[m]:<25} {total:>10} {n_probs_div:>15} "
              f"{max_div:>14} {mean_div:>14.1f}")

    # --- Main results (use nanmedian for robustness) ---
    print(f"\n{'=' * 80}")
    print(f"RESULTS ({n_problems} problems, T={T}, {n_traj} traj) "
          f"in {t_total:.0f}s ({t_total/60:.1f}min)")
    print("Median over problems (coverage in %, L2 and CI width x 1e-3)")
    print("=" * 80)
    header = (f"{'Method':<25} {'L2 x1e-3':>10} {'Width x1e-3':>12} "
              f"{'Cov %':>8} {'Mean Cov %':>10}")
    print(header)
    print("-" * 68)

    rows = []
    for m in METHODS_ORDER:
        l2_arr = np.array(all_results[m]['l2'])
        w_arr = np.array(all_results[m]['width'])
        cov_arr = np.array(all_results[m]['cov'])
        l2_med = float(np.nanmedian(l2_arr)) * 1e3
        w_med = float(np.nanmedian(w_arr)) * 1e3
        cov_med = float(np.nanmedian(cov_arr)) * 100
        cov_mean = float(np.nanmean(cov_arr)) * 100
        print(f"{METHOD_LABELS[m]:<25} {l2_med:>10.2f} {w_med:>12.2f} "
              f"{cov_med:>8.1f} {cov_mean:>10.1f}")
        rows.append({
            'method': m, 'label': METHOD_LABELS[m],
            'l2_median': l2_med, 'width_median': w_med,
            'cov_median': cov_med, 'cov_mean': cov_mean,
            'diverged_total': int(np.sum(all_diverged[m])),
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
        ps = np.nanpercentile(vals, pcts)
        print(f"{METHOD_LABELS[m]:<25}" + "".join(f"{v:>8.1f}" for v in ps))

    print(f"\n{'=' * 80}")
    print("BIAS (L2 x 1e-3) PERCENTILES across problems")
    print("=" * 80)
    print(pct_header)
    print("-" * (25 + 8 * len(pcts)))
    for m in METHODS_ORDER:
        vals = np.array(all_results[m]['l2']) * 1e3
        ps = np.nanpercentile(vals, pcts)
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
    parser.add_argument('--n-bootstrap', type=int, default=500,
                        help='MSB bootstrap replications (default: 500)')
    parser.add_argument('--direction-coord', type=int, default=None,
                        help='Coordinate index k for e_k direction. '
                             'If omitted, uses a random unit direction per problem.')
    args = parser.parse_args()

    run_experiment(args.n_problems, args.n_traj, args.T,
                   args.n_states, args.d, args.seed, args.n_workers,
                   args.n_bootstrap, args.direction_coord)


if __name__ == '__main__':
    main()
