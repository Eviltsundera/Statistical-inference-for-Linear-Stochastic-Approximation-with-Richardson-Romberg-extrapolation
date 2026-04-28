"""Empirical bias/variance of OBM vs OBM-RR (lugsail) variance estimators.

Tests the prediction from Vats & Flegal (2022), Corollary 1: applying a
lugsail construction to a spectral-variance estimator multiplies the
leading bias term by (1 - c_n * r^q) / (1 - c_n), while inflating the
variance by a factor int k_L^2 / int k^2.

Our construction
    sigma^2_RR(b, lam) = lam/(lam-1) * sigma^2(lam*b) - 1/(lam-1) * sigma^2(b)
corresponds to lugsail with c_n = 1/lam, r = lam. For the Bartlett kernel
(q = 1) this gives (1 - c_n * r) / (1 - c_n) = 0, i.e. the leading O(1/b)
bias is cancelled.

Ground truth (for PR averaging under diminishing step, which satisfies the
CLT):
    sigma^2_inf = u^T A_bar^{-1} Gamma_eps A_bar^{-T} u
    Gamma_eps   = E^T (D Z + Z^T D - D) E
    Z = (I - P + 1 pi^T)^{-1}, D = diag(pi), E rows = eps(x) = A(x) theta* + b(x).

Parameters varied (see CLI):
  --T            sample size (check 1/b bias scaling is T-independent)
  --bn-exps      block-size exponents b = floor(T^e), e.g. 0.2..0.9
  --lam-list     RR ratios lambda (2 cancels O(1/b) exactly; 3, 4 inflate var)
  --iterates     PR (clean truth), const (transient bias), RR (our method)
  --alpha-const  constant step size for const / RR iterates
  --eig-min/max, --noise-target   problem conditioning / state-dependent noise
  --n-problems, --n-traj          outer/inner replication for bias-var Monte
                                  Carlo

Outputs:
  <out>.csv       per-problem raw rows
  <out>_agg.csv   aggregated across problems
  Printed summary: best (min-MSE) b per (iterate, estimator).
"""

import argparse
import os
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

from lsa_inference.markov_chain import (
    generate_transition_matrix, simulate_chains_batch,
)
from lsa_inference.lsa_problem import (
    generate_A, generate_b, compute_theta_star, compute_asymptotic_variance,
)
from lsa_inference.lsa_engine import (
    prepare_arrays, run_lsa_polyak_ruppert, run_rr_full,
)
from lsa_inference.inference import _obm_variance_from_proj


def _collect_bn_needed(bn_list, lam_list):
    """All block sizes we need sigma^2_OBM evaluated at (base + lam*base)."""
    needed = set(bn_list)
    for b in bn_list:
        for lam in lam_list:
            needed.add(int(round(lam * b)))
    return sorted(needed)


def _bias_var_for_source(proj, theta_bar, direction, bn_list, lam_list,
                         sigma_true, T, iterate_name, prob_seed):
    """Compute bias/variance of OBM and OBM-RR estimators across b, lam."""
    bar_proj = theta_bar @ direction

    # Trajectories that diverged: theta_bar has NaN. _obm_variance_from_proj
    # would return finite garbage for them via nancumsum, so mask explicitly.
    diverged = np.any(np.isnan(theta_bar), axis=1)

    # Per-b OBM estimates, one value per trajectory
    all_b = [b for b in _collect_bn_needed(bn_list, lam_list) if 0 < b < T]
    sigma_obm = {}
    for b in all_b:
        s = _obm_variance_from_proj(proj, bar_proj, b)
        s[diverged] = np.nan
        sigma_obm[b] = s

    rows = []
    for b in bn_list:
        if b not in sigma_obm:
            continue
        vals = sigma_obm[b]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        rows.append({
            'problem': prob_seed, 'iterate': iterate_name,
            'estimator': 'OBM', 'b': b, 'lam': 0.0, 'T': T,
            'sigma_true': sigma_true,
            'mean': float(finite.mean()),
            'var': float(finite.var(ddof=1)) if finite.size > 1 else np.nan,
            'mse': float(np.mean((finite - sigma_true) ** 2)),
        })
        for lam in lam_list:
            b_large = int(round(lam * b))
            if b_large not in sigma_obm:
                continue
            small = sigma_obm[b]
            large = sigma_obm[b_large]
            rr = (lam / (lam - 1)) * large - (1 / (lam - 1)) * small
            rr_clamped = np.maximum(rr, 0.0)
            finite = rr_clamped[np.isfinite(rr_clamped)]
            if finite.size == 0:
                continue
            rows.append({
                'problem': prob_seed, 'iterate': iterate_name,
                'estimator': 'OBM_RR', 'b': b, 'lam': float(lam), 'T': T,
                'sigma_true': sigma_true,
                'mean': float(finite.mean()),
                'var': float(finite.var(ddof=1)) if finite.size > 1 else np.nan,
                'mse': float(np.mean((finite - sigma_true) ** 2)),
            })
    return rows


def _worker(args):
    (prob_seed, n_traj, T, n_states, d,
     pr_c0, pr_k0, pr_gamma, alpha_const, rr_alphas,
     bn_list, lam_list, iterate_types,
     eig_min, eig_max, noise_target) = args

    rng = np.random.default_rng(prob_seed)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(
        n_states, d, pi, rng,
        eig_min=eig_min, eig_max=eig_max, noise_target=noise_target,
    )
    b_vec = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_vec, pi)
    A_arr, b_arr = prepare_arrays(A_list, b_vec)

    u = rng.standard_normal(d)
    direction = u / np.linalg.norm(u)

    sigma_true = compute_asymptotic_variance(
        A_list, b_vec, P, pi, theta_star, direction,
    )

    traj_rng = np.random.default_rng(rng.integers(0, 2**31))
    trajs = simulate_chains_batch(P, pi, T, n_traj, traj_rng)

    rows = []

    if 'PR' in iterate_types:
        proj, theta_bar = run_lsa_polyak_ruppert(
            A_arr, b_arr, trajs, pr_c0, pr_k0, pr_gamma, direction=direction,
        )
        rows.extend(_bias_var_for_source(
            proj, theta_bar, direction, bn_list, lam_list,
            sigma_true, T, 'PR', prob_seed,
        ))

    if {'const', 'RR'} & set(iterate_types):
        K = max(int(T ** 0.3), 5)
        burn_in = min(1000, T // 10)
        (rr_proj, rr_theta_bar, _,
         per_alpha_proj, _, per_alpha_bar, _) = run_rr_full(
            A_arr, b_arr, trajs, list(rr_alphas), K, burn_in,
            direction=direction,
        )
        alpha_small_idx = int(np.argmin(rr_alphas))
        if 'const' in iterate_types:
            rows.extend(_bias_var_for_source(
                per_alpha_proj[alpha_small_idx],
                per_alpha_bar[alpha_small_idx],
                direction, bn_list, lam_list,
                sigma_true, T, f'const_{rr_alphas[alpha_small_idx]}',
                prob_seed,
            ))
        if 'RR' in iterate_types:
            rows.extend(_bias_var_for_source(
                rr_proj, rr_theta_bar, direction, bn_list, lam_list,
                sigma_true, T, 'RR_iterate', prob_seed,
            ))

    return rows


def _make_bn_list(T, exponents, max_lam):
    """Block sizes floor(T^e), capped so that max_lam * b < T."""
    cap = max(T // (int(np.ceil(max_lam)) + 1), 5)
    return sorted(set(
        max(5, min(int(T ** e), cap)) for e in exponents
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Bias/variance of OBM vs OBM-RR (lugsail)",
    )
    parser.add_argument('--n-problems', type=int, default=20)
    parser.add_argument('--n-traj', type=int, default=200)
    parser.add_argument('--T', type=int, nargs='+', default=[10000])
    parser.add_argument('--n-states', type=int, default=10)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--bn-exps', type=float, nargs='+',
                        default=[0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    parser.add_argument('--lam-list', type=float, nargs='+', default=[2.0, 3.0])
    parser.add_argument('--alpha-const', type=float, default=0.02)
    parser.add_argument('--rr-alphas', type=float, nargs='+',
                        default=[0.2, 0.02])
    parser.add_argument('--pr-c0', type=float, default=200.0)
    parser.add_argument('--pr-k0', type=float, default=20000.0)
    parser.add_argument('--pr-gamma', type=float, default=0.65)
    parser.add_argument('--eig-min', type=float, default=0.25)
    parser.add_argument('--eig-max', type=float, default=0.60)
    parser.add_argument('--noise-target', type=float, default=0.35)
    parser.add_argument('--iterates', type=str, nargs='+',
                        default=['PR', 'const', 'RR'],
                        choices=['PR', 'const', 'RR'])
    parser.add_argument('--out', type=str, default='results/lugsail_bv.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-workers', type=int, default=None)
    cli = parser.parse_args()

    rng_master = np.random.default_rng(cli.seed)
    max_lam = max(cli.lam_list)

    n_workers = cli.n_workers or min(mp.cpu_count(), cli.n_problems)
    print(f"Config: {cli.n_problems} problems x {cli.n_traj} traj, "
          f"d={cli.d}, n_states={cli.n_states}, workers={n_workers}")
    print(f"  iterates: {cli.iterates}")
    print(f"  lam_list: {cli.lam_list}")
    print(f"  PR: c0={cli.pr_c0}, k0={cli.pr_k0}, gamma={cli.pr_gamma}")
    print(f"  rr_alphas: {cli.rr_alphas}  (const uses min)")
    print(f"  problem: eig in [{cli.eig_min},{cli.eig_max}], "
          f"noise_target={cli.noise_target}")
    print(flush=True)

    all_rows = []
    for T in cli.T:
        bn_list = _make_bn_list(T, cli.bn_exps, max_lam)
        seeds = [int(rng_master.integers(0, 2**31))
                 for _ in range(cli.n_problems)]
        task_args = [
            (s, cli.n_traj, T, cli.n_states, cli.d,
             cli.pr_c0, cli.pr_k0, cli.pr_gamma,
             cli.alpha_const, tuple(cli.rr_alphas),
             bn_list, cli.lam_list, cli.iterates,
             cli.eig_min, cli.eig_max, cli.noise_target)
            for s in seeds
        ]

        print(f"T={T}: b_n={bn_list}", flush=True)
        t_start = time.time()
        with mp.Pool(n_workers) as pool:
            for i, rows in enumerate(pool.imap_unordered(_worker, task_args), 1):
                all_rows.extend(rows)
                if i % max(1, cli.n_problems // 10) == 0 or i == 1:
                    elapsed = time.time() - t_start
                    print(f"  [{i}/{cli.n_problems}] {elapsed:.0f}s",
                          flush=True)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No rows collected.")
        return

    df['rel_bias'] = (df['mean'] - df['sigma_true']) / df['sigma_true']
    df['bias'] = df['mean'] - df['sigma_true']

    agg = df.groupby(['T', 'iterate', 'estimator', 'b', 'lam']).agg(
        sigma_true=('sigma_true', 'mean'),
        mean_mean=('mean', 'mean'),
        bias_mean=('bias', 'mean'),
        rel_bias_median=('rel_bias', 'median'),
        var_mean=('var', 'mean'),
        mse_mean=('mse', 'mean'),
        n_probs=('problem', 'nunique'),
    ).reset_index()

    print("\n=== Best-MSE block size per (T, iterate, estimator, lam) ===")
    for (T, it), grp in agg.groupby(['T', 'iterate']):
        print(f"\n T={T}  iterate={it}")
        for (est, lam), sub in grp.groupby(['estimator', 'lam']):
            if sub.empty:
                continue
            best = sub.loc[sub['mse_mean'].idxmin()]
            tag = f"{est}" if est == 'OBM' else f"{est}(lam={lam:g})"
            print(
                f"   {tag:>14}  b*={int(best['b']):>6}   "
                f"rel_bias={best['rel_bias_median']:+.3f}  "
                f"var={best['var_mean']:.3e}  "
                f"mse={best['mse_mean']:.3e}"
            )

    os.makedirs(os.path.dirname(cli.out) or '.', exist_ok=True)
    df.to_csv(cli.out, index=False)
    agg_out = cli.out.replace('.csv', '_agg.csv')
    agg.to_csv(agg_out, index=False)
    print(f"\nSaved raw to     {cli.out}")
    print(f"Saved aggregated {agg_out}")


if __name__ == '__main__':
    main()
