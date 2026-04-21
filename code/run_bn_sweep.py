"""Sweep block size b_n for OBM-based methods.

Runs a few key methods (const 0.02, PR, RR) with OBM CI across a range of
block sizes to find the right b_n for each.

Usage:
    python run_bn_sweep.py
    python run_bn_sweep.py --n-problems 50 --n-traj 100 --T 100000
"""

import argparse
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

from lsa_inference.markov_chain import generate_transition_matrix, simulate_chains_batch
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.lsa_engine import (
    prepare_arrays, run_lsa_const_full, run_lsa_polyak_ruppert, run_rr_full,
)
from lsa_inference.inference import obm_ci


def _worker(args):
    """Worker: generate problem, run methods, sweep b_n, return coverage."""
    (prob_seed, n_traj, T, n_states, d,
     K, burn_in, pr_c0, pr_k0, pr_gamma, bn_list) = args

    rng = np.random.default_rng(prob_seed)

    # Generate problem
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)
    A_arr, b_arr = prepare_arrays(A_list, b_list)

    # Random direction
    u = rng.standard_normal(d)
    direction = u / np.linalg.norm(u)

    traj_rng = np.random.default_rng(rng.integers(0, 2**31))
    trajs = simulate_chains_batch(P, pi, T, n_traj, traj_rng)

    # --- Run each method ONCE, then sweep b_n for OBM ---

    # Constant alpha=0.02 (proj + theta_bar)
    (rr_proj, rr_theta_bar, rr_bm,
     per_alpha_proj, per_alpha_bm, per_alpha_bar, n) = run_rr_full(
        A_arr, b_arr, trajs, [0.2, 0.02], K, burn_in, direction=direction
    )
    const02_proj = per_alpha_proj[1]
    const02_bar = per_alpha_bar[1]

    # PR averaging
    pr_proj, pr_theta_bar = run_lsa_polyak_ruppert(
        A_arr, b_arr, trajs, pr_c0, pr_k0, pr_gamma, direction=direction
    )

    # Sweep b_n
    rows = []
    for b_n in bn_list:
        for method_name, proj, theta_bar in [
            ('const_0.02', const02_proj, const02_bar),
            ('PR',         pr_proj,      pr_theta_bar),
            ('RR',         rr_proj,      rr_theta_bar),
        ]:
            _, widths, covs = obm_ci(proj, theta_bar, b_n, theta_star,
                                     direction=direction)
            rows.append({
                'method': method_name,
                'b_n': b_n,
                'cov': float(np.nanmean(covs)),
                'width': float(np.nanmean(widths)),
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Sweep b_n for OBM methods")
    parser.add_argument('--n-problems', type=int, default=10)
    parser.add_argument('--n-traj', type=int, default=50)
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--n-states', type=int, default=10)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-workers', type=int, default=None)
    args = parser.parse_args()

    T = args.T
    K = max(int(T ** 0.3), 5)
    burn_in = min(1000, T // 10)
    pr_gamma = 0.65
    pr_c0 = 200.0
    pr_k0 = 20000

    # b_n values: from T^0.3 to T^0.9
    exponents = np.arange(0.3, 0.91, 0.05)
    bn_list = sorted(set(
        max(10, min(int(T ** e), T // 4)) for e in exponents
    ))

    n_workers = args.n_workers or min(mp.cpu_count(), args.n_problems)

    print(f"Sweep config: {args.n_problems} problems x {args.n_traj} traj, T={T}")
    print(f"  b_n values: {bn_list}")
    print(f"  Workers: {n_workers}")
    print(flush=True)

    rng_master = np.random.default_rng(args.seed)
    seeds = [int(rng_master.integers(0, 2**31)) for _ in range(args.n_problems)]

    task_args = [
        (s, args.n_traj, T, args.n_states, args.d,
         K, burn_in, pr_c0, pr_k0, pr_gamma, bn_list)
        for s in seeds
    ]

    # Collect results
    all_rows = []
    t_start = time.time()

    with mp.Pool(n_workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(_worker, task_args), 1):
            all_rows.extend(rows)
            if i % max(1, args.n_problems // 10) == 0 or i == 1:
                elapsed = time.time() - t_start
                print(f"  [{i}/{args.n_problems}] {elapsed:.0f}s", flush=True)

    df = pd.DataFrame(all_rows)

    # Aggregate: median coverage and width across problems
    agg = df.groupby(['method', 'b_n']).agg(
        cov_median=('cov', 'median'),
        cov_mean=('cov', 'mean'),
        width_median=('width', 'median'),
    ).reset_index()

    # Print table
    methods = ['const_0.02', 'PR', 'RR']
    print(f"\n{'='*80}")
    print(f"COVERAGE (median %) vs b_n   ({args.n_problems} problems, T={T})")
    print(f"{'='*80}")

    header = f"{'b_n':>8} {'T^exp':>6}"
    for m in methods:
        header += f"  {m:>12}"
    print(header)
    print("-" * len(header))

    for b_n in bn_list:
        exp = np.log(b_n) / np.log(T) if T > 1 else 0
        line = f"{b_n:>8} {exp:>6.2f}"
        for m in methods:
            row = agg[(agg['method'] == m) & (agg['b_n'] == b_n)]
            if len(row) > 0:
                cov = row['cov_median'].values[0] * 100
                line += f"  {cov:>11.1f}%"
            else:
                line += f"  {'N/A':>12}"
        print(line)

    print(f"\n{'='*80}")
    print(f"CI WIDTH (median x1e-3) vs b_n")
    print(f"{'='*80}")

    header = f"{'b_n':>8} {'T^exp':>6}"
    for m in methods:
        header += f"  {m:>12}"
    print(header)
    print("-" * len(header))

    for b_n in bn_list:
        exp = np.log(b_n) / np.log(T) if T > 1 else 0
        line = f"{b_n:>8} {exp:>6.2f}"
        for m in methods:
            row = agg[(agg['method'] == m) & (agg['b_n'] == b_n)]
            if len(row) > 0:
                w = row['width_median'].values[0] * 1e3
                line += f"  {w:>11.2f}"
            else:
                line += f"  {'N/A':>12}"
        print(line)

    # Save
    out = 'bn_sweep.csv'
    agg.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
