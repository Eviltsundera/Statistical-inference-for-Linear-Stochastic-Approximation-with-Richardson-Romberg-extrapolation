"""Empirical CDF diagnostics for the balanced RR normal approximation.

This runner tests the theorem-facing statistic

    Z_n^RR(u) = sqrt(n) u^T (theta_bar_RR - theta*) / sigma_inf(u)

with alpha_n = c / sqrt(n) and the two-level RR pair (2 alpha_n, alpha_n).
It uses the analytic finite-state long-run variance, not OBM, so the measured
Kolmogorov error targets the normal approximation rather than variance
estimation.
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lsa_inference.markov_chain import generate_transition_matrix
from lsa_inference.lsa_problem import (
    compute_asymptotic_variance,
    compute_theta_star,
    generate_A,
    generate_b,
    problem_diagnostics,
)
from lsa_inference.lsa_engine import prepare_arrays


METHODS = ("RR", "single_alpha", "single_2alpha")


def _draw_initial_states(pi, n_traj, rng, start_state):
    if start_state is not None:
        return np.full(n_traj, int(start_state), dtype=np.int32)
    return np.searchsorted(np.cumsum(pi), rng.uniform(size=n_traj)).astype(
        np.int32
    )


def _simulate_chunk(args):
    (
        P,
        pi,
        A_arr,
        b_arr,
        theta_star,
        direction,
        sigma_true,
        n,
        n0,
        alpha,
        n_traj,
        seed,
        start_state,
    ) = args

    rng = np.random.default_rng(seed)
    n_states = len(pi)
    d = b_arr.shape[1]
    cum_P = np.cumsum(P, axis=1)

    states = _draw_initial_states(pi, n_traj, rng, start_state)
    theta_alpha = np.zeros((n_traj, d))
    theta_2alpha = np.zeros((n_traj, d))
    sum_alpha = np.zeros((n_traj, d))
    sum_2alpha = np.zeros((n_traj, d))

    for t in range(n):
        A_t = A_arr[states]
        b_t = b_arr[states]

        drift_alpha = np.einsum("nij,nj->ni", A_t, theta_alpha) + b_t
        drift_2alpha = np.einsum("nij,nj->ni", A_t, theta_2alpha) + b_t
        theta_alpha += alpha * drift_alpha
        theta_2alpha += (2.0 * alpha) * drift_2alpha

        if t % 100 == 99:
            bad_alpha = (
                ~np.isfinite(theta_alpha) | (np.abs(theta_alpha) > 1e6)
            )
            bad_2alpha = (
                ~np.isfinite(theta_2alpha) | (np.abs(theta_2alpha) > 1e6)
            )
            bad_rows = np.any(bad_alpha | bad_2alpha, axis=1)
            if np.any(bad_rows):
                theta_alpha[bad_rows] = np.nan
                theta_2alpha[bad_rows] = np.nan

        if t >= n0:
            sum_alpha += theta_alpha
            sum_2alpha += theta_2alpha

        if t + 1 < n:
            u = rng.uniform(size=n_traj)
            states = (u[:, None] < cum_P[states]).argmax(axis=1).astype(
                np.int32
            )

    m = max(n - n0, 1)
    bar_alpha = sum_alpha / m
    bar_2alpha = sum_2alpha / m
    bar_rr = 2.0 * bar_alpha - bar_2alpha

    scale = np.sqrt(n) / np.sqrt(sigma_true)
    star_proj = float(theta_star @ direction)
    z_alpha = scale * ((bar_alpha @ direction) - star_proj)
    z_2alpha = scale * ((bar_2alpha @ direction) - star_proj)
    z_rr = scale * ((bar_rr @ direction) - star_proj)

    return {
        "RR": z_rr,
        "single_alpha": z_alpha,
        "single_2alpha": z_2alpha,
    }


def _ks_distance(z):
    z = np.sort(np.asarray(z, dtype=float))
    z = z[np.isfinite(z)]
    m = z.size
    if m == 0:
        return np.nan

    phi = stats.norm.cdf(z)
    upper = np.arange(1, m + 1, dtype=float) / m
    lower = np.arange(0, m, dtype=float) / m
    return float(np.max(np.maximum(np.abs(upper - phi), np.abs(phi - lower))))


def _summarize_z(method, z, n, n0, alpha, sigma_true, dkw_eps):
    z = np.asarray(z, dtype=float)
    finite = z[np.isfinite(z)]
    m = finite.size
    if m == 0:
        return {
            "n": int(n),
            "n0": int(n0),
            "m_eff": int(max(n - n0, 1)),
            "alpha": float(alpha),
            "method": method,
            "n_samples": 0,
            "diverged": int(z.size),
            "sigma_true": float(sigma_true),
            "ks_D": np.nan,
            "ks_D_minus_dkw": np.nan,
            "dkw_eps_95": float(dkw_eps),
            "mean_Z": np.nan,
            "var_Z": np.nan,
            "skew_Z": np.nan,
            "excess_kurt_Z": np.nan,
            "coverage_95": np.nan,
            "q025_error": np.nan,
            "q975_error": np.nan,
        }

    ks_d = _ks_distance(finite)
    q025, q975 = np.quantile(finite, [0.025, 0.975])
    return {
        "n": int(n),
        "n0": int(n0),
        "m_eff": int(max(n - n0, 1)),
        "alpha": float(alpha),
        "method": method,
        "n_samples": int(m),
        "diverged": int(z.size - m),
        "sigma_true": float(sigma_true),
        "ks_D": ks_d,
        "ks_D_minus_dkw": float(max(ks_d - dkw_eps, 0.0)),
        "dkw_eps_95": float(dkw_eps),
        "mean_Z": float(np.mean(finite)),
        "var_Z": float(np.var(finite, ddof=1)) if m > 1 else np.nan,
        "skew_Z": float(stats.skew(finite, bias=False)) if m > 2 else np.nan,
        "excess_kurt_Z": (
            float(stats.kurtosis(finite, fisher=True, bias=False))
            if m > 3
            else np.nan
        ),
        "coverage_95": float(np.mean(np.abs(finite) <= 1.959963984540054)),
        "q025_error": float(q025 - stats.norm.ppf(0.025)),
        "q975_error": float(q975 - stats.norm.ppf(0.975)),
    }


def _cdf_error_rows(method, z, n, grid):
    z = np.asarray(z, dtype=float)
    finite = np.sort(z[np.isfinite(z)])
    if finite.size == 0:
        return []

    counts = np.searchsorted(finite, grid, side="right")
    f_hat = counts / finite.size
    phi = stats.norm.cdf(grid)
    return [
        {
            "n": int(n),
            "method": method,
            "x": float(x),
            "F_hat": float(f),
            "Phi": float(p),
            "F_minus_Phi": float(f - p),
        }
        for x, f, p in zip(grid, f_hat, phi)
    ]


def _make_problem(seed, n_states, d, eig_min, eig_max, noise_target):
    rng = np.random.default_rng(seed)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(
        n_states,
        d,
        pi,
        rng,
        eig_min=eig_min,
        eig_max=eig_max,
        noise_target=noise_target,
    )
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)
    direction_raw = rng.standard_normal(d)
    direction = direction_raw / np.linalg.norm(direction_raw)
    sigma_true = compute_asymptotic_variance(
        A_list, b_list, P, pi, theta_star, direction
    )
    a_proxy = float(np.min(np.linalg.eigvalsh(-A_bar)))
    diagnostics = problem_diagnostics(A_list)
    return {
        "P": P,
        "pi": pi,
        "A_list": A_list,
        "b_list": b_list,
        "A_arr": prepare_arrays(A_list, b_list)[0],
        "b_arr": prepare_arrays(A_list, b_list)[1],
        "theta_star": theta_star,
        "direction": direction,
        "sigma_true": sigma_true,
        "a_proxy": a_proxy,
        "diagnostics": diagnostics,
    }


def _burn_in_length(n, alpha, a_proxy, kappa, cap_frac):
    raw = int(np.floor(kappa * (alpha * a_proxy) ** -1 * np.log(n) ** 2))
    cap = int(np.floor(cap_frac * n))
    return max(0, min(raw, cap)), raw


def run_experiment(
    n_values,
    n_traj,
    chunk_size,
    c_alpha,
    burn_kappa,
    burn_cap_frac,
    n_states,
    d,
    seed,
    problem_seed,
    n_workers,
    start_state,
    eig_min,
    eig_max,
    noise_target,
    out,
    cdf_out,
    z_out,
    grid_min,
    grid_max,
    grid_size,
):
    n_values = [int(n) for n in n_values]
    if n_workers is None:
        n_workers = max(1, min(mp.cpu_count(), len(n_values)))

    problem = _make_problem(
        problem_seed, n_states, d, eig_min, eig_max, noise_target
    )
    master_rng = np.random.default_rng(seed)
    grid = np.linspace(grid_min, grid_max, grid_size)

    print("RR empirical CDF experiment")
    print(f"  n_values={n_values}")
    print(f"  n_traj={n_traj}, chunk_size={chunk_size}, workers={n_workers}")
    print(f"  alpha_n={c_alpha}/sqrt(n), burn_kappa={burn_kappa}")
    print(f"  burn_cap_frac={burn_cap_frac}, start_state={start_state}")
    print(
        f"  problem_seed={problem_seed}, sigma_true={problem['sigma_true']:.6g}, "
        f"a_proxy={problem['a_proxy']:.6g}"
    )
    print(f"  diagnostics={problem['diagnostics']}")
    print(flush=True)

    summary_rows = []
    cdf_rows = []
    z_frames = []
    t_all = time.time()

    for n in n_values:
        alpha = float(c_alpha / np.sqrt(n))
        n0, n0_raw = _burn_in_length(
            n, alpha, problem["a_proxy"], burn_kappa, burn_cap_frac
        )
        chunk_sizes = []
        remaining = n_traj
        while remaining > 0:
            size = min(chunk_size, remaining)
            chunk_sizes.append(size)
            remaining -= size

        task_args = []
        for size in chunk_sizes:
            task_args.append(
                (
                    problem["P"],
                    problem["pi"],
                    problem["A_arr"],
                    problem["b_arr"],
                    problem["theta_star"],
                    problem["direction"],
                    problem["sigma_true"],
                    n,
                    n0,
                    alpha,
                    size,
                    int(master_rng.integers(0, 2**31)),
                    start_state,
                )
            )

        print(
            f"n={n}: alpha={alpha:.6g}, 2alpha={2 * alpha:.6g}, "
            f"n0={n0} (raw={n0_raw}), chunks={len(task_args)}",
            flush=True,
        )
        t_start = time.time()
        collected = {method: [] for method in METHODS}
        completed = 0

        with mp.Pool(n_workers) as pool:
            for chunk_result in pool.imap_unordered(_simulate_chunk, task_args):
                completed += 1
                for method in METHODS:
                    collected[method].append(chunk_result[method])
                if completed == 1 or completed % max(1, len(task_args) // 10) == 0:
                    elapsed = time.time() - t_start
                    eta = elapsed / completed * (len(task_args) - completed)
                    print(
                        f"  n={n} [{completed}/{len(task_args)}] "
                        f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                        flush=True,
                    )

        dkw_eps = float(np.sqrt(np.log(2.0 / 0.05) / (2.0 * n_traj)))
        for method in METHODS:
            z = np.concatenate(collected[method])
            summary_rows.append(
                _summarize_z(
                    method, z, n, n0, alpha, problem["sigma_true"], dkw_eps
                )
            )
            cdf_rows.extend(_cdf_error_rows(method, z, n, grid))
            z_frames.append(
                pd.DataFrame(
                    {
                        "n": int(n),
                        "method": method,
                        "sample": np.arange(z.size, dtype=np.int64),
                        "Z": z,
                    }
                )
            )

        print(f"Finished n={n} in {time.time() - t_start:.0f}s", flush=True)

    summary = pd.DataFrame(summary_rows)
    cdf = pd.DataFrame(cdf_rows)
    z_samples = pd.concat(z_frames, ignore_index=True)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    cdf_path = Path(cdf_out)
    cdf_path.parent.mkdir(parents=True, exist_ok=True)
    cdf.to_csv(cdf_path, index=False)

    z_path = Path(z_out)
    z_path.parent.mkdir(parents=True, exist_ok=True)
    z_samples.to_csv(z_path, index=False)

    print("\nSummary:")
    cols = [
        "n",
        "method",
        "n0",
        "alpha",
        "ks_D",
        "ks_D_minus_dkw",
        "mean_Z",
        "var_Z",
        "coverage_95",
        "q025_error",
        "q975_error",
    ]
    print(summary[cols].to_string(index=False))
    print(f"\nSaved summary to {out_path}")
    print(f"Saved CDF grid to {cdf_path}")
    print(f"Saved Z samples to {z_path}")
    print(f"Total runtime: {time.time() - t_all:.0f}s")
    return summary, cdf, z_samples


def main():
    parser = argparse.ArgumentParser(
        description="Empirical CDF/KS diagnostic for balanced RR statistic",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[50_000, 100_000, 300_000, 1_000_000],
    )
    parser.add_argument("--n-traj", type=int, default=10_000)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--c-alpha", type=float, default=20.0)
    parser.add_argument("--burn-kappa", type=float, default=1.0)
    parser.add_argument("--burn-cap-frac", type=float, default=0.25)
    parser.add_argument("--n-states", type=int, default=10)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--problem-seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument(
        "--start-state",
        type=int,
        default=0,
        help="Fixed initial Markov state. Use -1 to start from stationarity.",
    )
    parser.add_argument("--eig-min", type=float, default=0.25)
    parser.add_argument("--eig-max", type=float, default=0.60)
    parser.add_argument("--noise-target", type=float, default=0.35)
    parser.add_argument(
        "--out",
        type=str,
        default="results/cdf/rr_cdf_summary.csv",
    )
    parser.add_argument(
        "--cdf-out",
        type=str,
        default="results/cdf/rr_cdf_grid.csv",
    )
    parser.add_argument(
        "--z-out",
        type=str,
        default="results/cdf/rr_cdf_z_samples.csv",
    )
    parser.add_argument("--grid-min", type=float, default=-3.0)
    parser.add_argument("--grid-max", type=float, default=3.0)
    parser.add_argument("--grid-size", type=int, default=601)
    args = parser.parse_args()

    start_state = None if args.start_state < 0 else args.start_state
    run_experiment(
        n_values=args.n_values,
        n_traj=args.n_traj,
        chunk_size=args.chunk_size,
        c_alpha=args.c_alpha,
        burn_kappa=args.burn_kappa,
        burn_cap_frac=args.burn_cap_frac,
        n_states=args.n_states,
        d=args.d,
        seed=args.seed,
        problem_seed=args.problem_seed,
        n_workers=args.n_workers,
        start_state=start_state,
        eig_min=args.eig_min,
        eig_max=args.eig_max,
        noise_target=args.noise_target,
        out=args.out,
        cdf_out=args.cdf_out,
        z_out=args.z_out,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()
