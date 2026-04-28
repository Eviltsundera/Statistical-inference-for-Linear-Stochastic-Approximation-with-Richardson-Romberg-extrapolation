"""Single-problem MSE decomposition of OBM and OBM-RR (lugsail) estimators.

Verifies the working asymptotic expansion from
`research/obm_rr_markov_lsa_report.md`, sections 4.3 and 5:

    E[sigma_hat^2(b)]  = sigma^2_inf + c1/b + c2/b^2 + c3 * b/n + o(1/b^2 + b/n)
    Var[sigma_hat^2(b)] ~ b/n
    MSE_OBM(b)         ~ const / b^2 + const * b/n         (b* ~ T^{1/3}, MSE ~ T^{-2/3})

For OBM-RR with lambda = 2 the leading 1/b term is cancelled, leaving
    E[sigma_hat^2_RR(b)] = sigma^2_inf - c2/(2 b^2) + 3 c3 * b/n + o(...)
    MSE_RR(b)            ~ const / b^4 + const * b/n       (b* ~ T^{1/5}, MSE ~ T^{-4/5})

We pin to **a single problem** and **a single random direction**, then
generate `n_traj` independent trajectories per `T` to estimate

    bias_hat(b)  = mean_traj(sigma_hat^2(b))                - sigma^2_inf
    var_hat(b)   = var_traj(sigma_hat^2(b))
    MSE_hat(b)   = mean_traj((sigma_hat^2(b) - sigma^2_inf)^2)

The script sweeps `b` (log-spaced) and `T` (multi-value), and dumps
per-(T, b, estimator, lam) rows so the decomposition can be checked by
fitting slopes such as

    log|bias|  ~ -1 * log b      for OBM
    log|bias|  ~ -2 * log b      for OBM-RR (after lam cancellation)
    log var    ~ +1 * log b
    log MSE*   ~ -2/3 * log T    for OBM     at optimal b
    log MSE*   ~ -4/5 * log T    for OBM-RR  at optimal b

For large T the trajectory tensor (n_traj, T) does not fit in memory in
one shot. Use `--traj-chunk` to process trajectories in groups; the
per-trajectory sigma_hat^2 values are concatenated across chunks.
"""

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd

from lsa_inference.markov_chain import (
    generate_transition_matrix, simulate_chains_batch,
)
from lsa_inference.lsa_problem import (
    generate_A, generate_b, compute_theta_star, compute_asymptotic_variance,
)
from lsa_inference.lsa_engine import prepare_arrays, run_lsa_polyak_ruppert
from lsa_inference.inference import _obm_variance_from_proj


def _make_bn_list(T, n_points, lo_exp, hi_exp, max_lam):
    """Log-spaced block sizes b = floor(T^e), capped so that max_lam*b < T."""
    cap = max(T // (int(np.ceil(max_lam)) + 1), 5)
    exps = np.linspace(lo_exp, hi_exp, n_points)
    return sorted(set(max(5, min(int(T ** e), cap)) for e in exps))


def _all_b_needed(bn_list, lam_list, T):
    """All block sizes whose OBM variance must be evaluated (b and round(lam*b))."""
    needed = set(bn_list)
    for b in bn_list:
        for lam in lam_list:
            needed.add(int(round(lam * b)))
    return sorted(b for b in needed if 0 < b < T)


def _run_chunk_obm(A_arr, b_arr, P, pi, T, n_traj_chunk, traj_rng,
                   pr_c0, pr_k0, pr_gamma, direction, all_b):
    """Simulate `n_traj_chunk` trajectories, run PR, return per-b OBM estimates.

    Returns:
        sigma_obm: dict b -> (n_traj_chunk,) float array. NaN for divergent
                   trajectories (those whose PR average is NaN).
    """
    trajs = simulate_chains_batch(P, pi, T, n_traj_chunk, traj_rng)
    proj, theta_bar = run_lsa_polyak_ruppert(
        A_arr, b_arr, trajs, pr_c0, pr_k0, pr_gamma, direction=direction,
    )
    del trajs

    bar_proj = theta_bar @ direction
    diverged = np.any(np.isnan(theta_bar), axis=1)

    sigma_obm = {}
    for b in all_b:
        s = _obm_variance_from_proj(proj, bar_proj, b)
        s[diverged] = np.nan
        sigma_obm[b] = s

    del proj
    gc.collect()
    return sigma_obm


def _summarise(arr, sigma_true):
    """Return (mean, bias, var, mse, n_used) over the finite entries of `arr`."""
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), 0
    m = float(finite.mean())
    v = float(finite.var(ddof=1)) if n > 1 else float('nan')
    mse = float(np.mean((finite - sigma_true) ** 2))
    return m, m - sigma_true, v, mse, n


def main():
    p = argparse.ArgumentParser(
        description="Single-problem MSE decomposition: OBM vs OBM-RR",
    )
    p.add_argument('--prob-seed', type=int, default=0)
    p.add_argument('--dir-seed', type=int, default=1)
    p.add_argument('--traj-seed', type=int, default=2)
    p.add_argument('--n-traj', type=int, default=1000)
    p.add_argument('--traj-chunk', type=int, default=0,
                   help='Process trajectories in chunks of this size '
                        '(0 = no chunking). Use ~200 for T >= 3*10^5.')
    p.add_argument('--T-list', type=int, nargs='+',
                   default=[10000, 30000, 100000, 300000, 1000000])
    p.add_argument('--n-states', type=int, default=10)
    p.add_argument('--d', type=int, default=5)
    p.add_argument('--lam-list', type=float, nargs='+', default=[2.0, 3.0])
    p.add_argument('--bn-points', type=int, default=40)
    p.add_argument('--bn-min-exp', type=float, default=0.20)
    p.add_argument('--bn-max-exp', type=float, default=0.90)
    p.add_argument('--pr-c0', type=float, default=200.0)
    p.add_argument('--pr-k0', type=float, default=20000.0)
    p.add_argument('--pr-gamma', type=float, default=0.65)
    p.add_argument('--eig-min', type=float, default=0.1)
    p.add_argument('--eig-max', type=float, default=0.6)
    p.add_argument('--noise-target', type=float, default=0.35)
    p.add_argument('--out', type=str, default='results/lugsail_decomp.csv')
    cli = p.parse_args()

    if cli.traj_chunk < 0:
        raise SystemExit("--traj-chunk must be >= 0")
    chunk = cli.traj_chunk if cli.traj_chunk > 0 else cli.n_traj

    # -- Build problem ONCE --
    rng_prob = np.random.default_rng(cli.prob_seed)
    P, pi = generate_transition_matrix(cli.n_states, rng_prob)
    A_list, A_bar = generate_A(
        cli.n_states, cli.d, pi, rng_prob,
        eig_min=cli.eig_min, eig_max=cli.eig_max,
        noise_target=cli.noise_target,
    )
    b_vec = generate_b(cli.n_states, cli.d, rng_prob)
    theta_star = compute_theta_star(A_list, b_vec, pi)
    A_arr, b_arr = prepare_arrays(A_list, b_vec)

    # -- Pick random direction ONCE --
    rng_dir = np.random.default_rng(cli.dir_seed)
    u = rng_dir.standard_normal(cli.d)
    direction = u / np.linalg.norm(u)

    sigma_true = compute_asymptotic_variance(
        A_list, b_vec, P, pi, theta_star, direction,
    )

    print("=" * 78)
    print("Single-problem decomposition (OBM and OBM-RR)")
    print(f"  prob_seed={cli.prob_seed}  dir_seed={cli.dir_seed}  "
          f"traj_seed={cli.traj_seed}")
    print(f"  d={cli.d}  n_states={cli.n_states}  "
          f"eig in [{cli.eig_min},{cli.eig_max}]  "
          f"noise_target={cli.noise_target}")
    print(f"  n_traj={cli.n_traj}  traj_chunk={chunk}")
    print(f"  PR: c0={cli.pr_c0} k0={cli.pr_k0} gamma={cli.pr_gamma}")
    print(f"  T_list={cli.T_list}  lam_list={cli.lam_list}")
    print(f"  bn_points={cli.bn_points} in T^[{cli.bn_min_exp},{cli.bn_max_exp}]")
    print(f"  sigma^2_inf (analytic) = {sigma_true:.6e}")
    print("=" * 78, flush=True)

    rng_traj = np.random.default_rng(cli.traj_seed)
    rows = []
    max_lam = max(cli.lam_list)

    for T in cli.T_list:
        bn_list = _make_bn_list(
            T, cli.bn_points, cli.bn_min_exp, cli.bn_max_exp, max_lam,
        )
        all_b = _all_b_needed(bn_list, cli.lam_list, T)

        print(f"\n=== T={T} ===")
        print(f"  bn_list ({len(bn_list)} pts): "
              f"{bn_list[0]}..{bn_list[-1]}")
        print(f"  all_b ({len(all_b)} OBM evals): "
              f"{all_b[0]}..{all_b[-1]}", flush=True)

        # -- Per-b accumulator: list of arrays from each chunk --
        sigma_per_b = {b: [] for b in all_b}
        n_done = 0
        t_T = time.time()
        while n_done < cli.n_traj:
            this_chunk = min(chunk, cli.n_traj - n_done)
            t_c = time.time()
            sigma_obm_chunk = _run_chunk_obm(
                A_arr, b_arr, P, pi, T, this_chunk, rng_traj,
                cli.pr_c0, cli.pr_k0, cli.pr_gamma, direction, all_b,
            )
            for b, s in sigma_obm_chunk.items():
                sigma_per_b[b].append(s)
            n_done += this_chunk
            dt = time.time() - t_c
            print(f"    chunk {n_done:>5}/{cli.n_traj}  ({this_chunk} traj)  "
                  f"{dt:6.1f}s",
                  flush=True)

        sigma_obm = {b: np.concatenate(arrs) for b, arrs in sigma_per_b.items()}
        del sigma_per_b
        gc.collect()

        # -- Aggregate per-b stats --
        for b in bn_list:
            arr = sigma_obm[b]
            m, bias, var, mse, n_used = _summarise(arr, sigma_true)
            rows.append({
                'T': T, 'b': b, 'estimator': 'OBM', 'lam': 0.0,
                'sigma_true': sigma_true,
                'mean': m, 'bias': bias, 'var': var, 'mse': mse,
                # OBM has no clamping; keep columns NaN for schema compatibility
                'mean_clamped': float('nan'), 'mse_clamped': float('nan'),
                'n_traj_used': n_used,
            })
            for lam in cli.lam_list:
                bL = int(round(lam * b))
                if bL not in sigma_obm:
                    continue
                rr = ((lam / (lam - 1)) * sigma_obm[bL]
                      - (1 / (lam - 1)) * sigma_obm[b])
                m, bias, var, mse, n_used = _summarise(rr, sigma_true)
                rr_c = np.maximum(rr, 0.0)
                m_c, _, _, mse_c, _ = _summarise(rr_c, sigma_true)
                rows.append({
                    'T': T, 'b': b, 'estimator': 'OBM_RR', 'lam': float(lam),
                    'sigma_true': sigma_true,
                    'mean': m, 'bias': bias, 'var': var, 'mse': mse,
                    'mean_clamped': m_c, 'mse_clamped': mse_c,
                    'n_traj_used': n_used,
                })

        print(f"  T={T} done in {time.time() - t_T:.1f}s", flush=True)

        del sigma_obm
        gc.collect()

    df = pd.DataFrame(rows)

    # -- Print summary table per T --
    print("\n" + "=" * 78)
    print("BEST-MSE (over swept b) per (T, estimator)")
    print("=" * 78)
    print(f"{'T':>10} {'method':>14} {'b*':>7} {'b/T^e':>7} "
          f"{'bias':>11} {'var':>11} {'mse':>11}")
    print("-" * 78)
    for T in cli.T_list:
        sub = df[df['T'] == T]
        for est, lam in [('OBM', 0.0)] + [('OBM_RR', l) for l in cli.lam_list]:
            ss = sub[(sub['estimator'] == est) & (sub['lam'] == lam)]
            ss = ss.dropna(subset=['mse'])
            if ss.empty:
                continue
            best = ss.loc[ss['mse'].idxmin()]
            tag = est if est == 'OBM' else f"OBM_RR(λ={lam:g})"
            exp = (np.log(best['b']) / np.log(T)) if T > 1 else 0
            print(f"{T:>10} {tag:>14} {int(best['b']):>7} {exp:>7.3f} "
                  f"{best['bias']:>+11.3e} {best['var']:>11.3e} "
                  f"{best['mse']:>11.3e}")

    os.makedirs(os.path.dirname(cli.out) or '.', exist_ok=True)
    df.to_csv(cli.out, index=False)
    print(f"\nSaved {len(df)} rows to {cli.out}")


if __name__ == '__main__':
    main()
