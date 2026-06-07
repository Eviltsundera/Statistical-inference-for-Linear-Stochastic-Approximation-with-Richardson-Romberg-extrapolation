"""Plot empirical CDF diagnostics from ``run_rr_cdf_experiment.py`` outputs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RR_COLOR = "#1f77b4"


def _setup_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.constrained_layout.use": True,
        }
    )


def plot_ks(summary, out_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    rr = summary[summary["method"] == "RR"].sort_values("n")
    ax.plot(
        rr["n"],
        rr["ks_D"],
        marker="o",
        linewidth=2.2,
        color=RR_COLOR,
        label="RR empirical KS",
    )
    if not rr.empty:
        ax.plot(
            rr["n"],
            rr["dkw_eps_95"],
            linestyle="--",
            linewidth=1.8,
            color="#444444",
            label="DKW 95% MC band",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("horizon n")
    ax.set_ylabel("Kolmogorov distance")
    ax.set_title("Empirical CDF distance to N(0,1)")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "rr_cdf_dense_ks_distance.svg")
    plt.close(fig)


def plot_theory_proxy(summary, out_dir):
    rr = summary[summary["method"] == "RR"].sort_values("n").copy()
    first_emp = float(rr["ks_D"].iloc[0])
    n = rr["n"].to_numpy(dtype=float)
    n0 = float(n[0])
    power_benchmark = first_emp * (n / n0) ** -0.25
    be_shape = np.log(n) ** 0.75 * n ** -0.25
    be_benchmark = first_emp * be_shape / be_shape[0]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(
        rr["n"],
        rr["ks_D"],
        marker="o",
        linewidth=2.2,
        color=RR_COLOR,
        label="RR empirical KS",
    )
    ax.plot(
        n,
        power_benchmark,
        linestyle="-",
        linewidth=1.9,
        color="#2ca02c",
        label=r"$n^{-1/4}$ benchmark (scaled)",
    )
    ax.plot(
        n,
        be_benchmark,
        linestyle="-.",
        linewidth=1.7,
        color="#ff7f0e",
        label=r"$\log^{3/4}(n)n^{-1/4}$ benchmark (scaled)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("horizon n")
    ax.set_ylabel("Kolmogorov distance")
    ax.set_title("Empirical KS versus theorem-rate benchmarks")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out_dir / "rr_cdf_dense_theory_proxy.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(
        rr["n"],
        rr["ks_D"],
        marker="o",
        linewidth=2.2,
        color=RR_COLOR,
        label="RR empirical KS",
    )
    ax.plot(
        n,
        power_benchmark,
        linestyle="-",
        linewidth=1.9,
        color="#2ca02c",
        label=r"$n^{-1/4}$ benchmark (scaled)",
    )
    ax.plot(
        n,
        be_benchmark,
        linestyle="-.",
        linewidth=1.7,
        color="#ff7f0e",
        label=r"$\log^{3/4}(n)n^{-1/4}$ benchmark (scaled)",
    )
    dkw = rr["dkw_eps_95"].to_numpy(dtype=float)
    ax.plot(
        rr["n"],
        dkw,
        linestyle=":",
        linewidth=1.8,
        color="#444444",
        label="DKW 95% MC band",
    )
    ax.set_xscale("log")
    ax.set_xlabel("horizon n")
    ax.set_ylabel("Kolmogorov distance")
    ax.set_title("Empirical KS versus theorem-rate benchmarks")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out_dir / "rr_cdf_dense_theory_proxy_linear.svg")
    plt.close(fig)


def plot_rr_cdf_error(cdf, summary, out_dir):
    rr = cdf[cdf["method"] == "RR"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ns = sorted(rr["n"].unique())
    colors = plt.cm.viridis([i / max(len(ns) - 1, 1) for i in range(len(ns))])
    for n, color in zip(ns, colors):
        group = rr[rr["n"] == n].sort_values("x")
        ax.plot(
            group["x"],
            group["F_minus_Phi"],
            linewidth=1.9,
            color=color,
            label=f"n={n:,}",
        )

    dkw = float(summary["dkw_eps_95"].dropna().iloc[0])
    ax.axhline(dkw, color="#444444", linestyle="--", linewidth=1.2)
    ax.axhline(-dkw, color="#444444", linestyle="--", linewidth=1.2)
    ax.axhline(0.0, color="#111111", linewidth=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\hat F_n(x)-\Phi(x)$")
    ax.set_title("RR signed empirical CDF error")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out_dir / "rr_cdf_dense_error_by_n.svg")
    plt.close(fig)


def _parse_selected_n(raw, available):
    if not raw:
        return [int(available[0]), int(available[len(available) // 2]), int(available[-1])]
    requested = [int(x) for x in raw.split(",") if x.strip()]
    available_set = set(int(x) for x in available)
    return [n for n in requested if n in available_set]


def plot_cdf_density(z_samples, out_dir, selected_n):
    rr = z_samples[z_samples["method"] == "RR"].copy()
    x_grid = np.linspace(-4.0, 4.0, 801)
    phi_cdf = stats.norm.cdf(x_grid)
    phi_pdf = stats.norm.pdf(x_grid)

    fig, axes = plt.subplots(
        len(selected_n),
        2,
        figsize=(9.0, 3.0 * len(selected_n)),
        squeeze=False,
    )

    for row, n in enumerate(selected_n):
        z = rr[rr["n"] == n]["Z"].to_numpy(dtype=float)
        z = np.sort(z[np.isfinite(z)])
        f_hat = np.searchsorted(z, x_grid, side="right") / z.size

        ax_cdf = axes[row, 0]
        ax_pdf = axes[row, 1]

        ax_cdf.plot(x_grid, f_hat, color=RR_COLOR, linewidth=2.0)
        ax_cdf.plot(x_grid, phi_cdf, color="#111111", linestyle="--", linewidth=1.6)
        ax_cdf.set_title(f"CDF, n={n:,}")
        ax_cdf.set_xlabel("x")
        ax_cdf.set_ylabel("distribution function")

        ax_pdf.hist(
            z,
            bins=60,
            range=(-4.0, 4.0),
            density=True,
            color=RR_COLOR,
            alpha=0.35,
            edgecolor="none",
        )
        ax_pdf.plot(x_grid, phi_pdf, color="#111111", linestyle="--", linewidth=1.8)
        ax_pdf.set_title(f"Density, n={n:,}")
        ax_pdf.set_xlabel("x")
        ax_pdf.set_ylabel("density")

    handles = [
        plt.Line2D([0], [0], color=RR_COLOR, linewidth=2.0),
        plt.Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.6),
    ]
    fig.legend(handles, ["empirical RR", "N(0,1)"], frameon=False, loc="upper center", ncol=2)
    fig.savefig(out_dir / "rr_cdf_dense_cdf_density_selected_n.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot RR empirical CDF experiment outputs",
    )
    parser.add_argument(
        "--summary",
        default="results/cdf/rr_cdf_balanced_M10000_T50k_1M_summary.csv",
    )
    parser.add_argument(
        "--cdf",
        default="results/cdf/rr_cdf_balanced_M10000_T50k_1M_grid.csv",
    )
    parser.add_argument(
        "--z",
        default=None,
        help="Optional raw Z sample CSV from run_rr_cdf_experiment.py.",
    )
    parser.add_argument(
        "--out-dir",
        default="../figures/experiments",
    )
    parser.add_argument(
        "--selected-n",
        default=None,
        help="Comma-separated horizons for CDF/density panels.",
    )
    args = parser.parse_args()

    _setup_style()
    summary = pd.read_csv(args.summary)
    cdf = pd.read_csv(args.cdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_ks(summary, out_dir)
    plot_theory_proxy(summary, out_dir)
    plot_rr_cdf_error(cdf, summary, out_dir)
    if args.z is not None:
        z_samples = pd.read_csv(args.z)
        selected_n = _parse_selected_n(args.selected_n, sorted(z_samples["n"].unique()))
        if selected_n:
            plot_cdf_density(z_samples, out_dir, selected_n)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
