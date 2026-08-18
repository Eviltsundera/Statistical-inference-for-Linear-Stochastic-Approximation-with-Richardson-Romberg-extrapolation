"""Build the main distributional figure from the saved RR CDF experiment.

The script only visualizes existing results.  It does not rerun the LSA
simulation.  Run it from any directory with ``python3``; the input and output
paths are resolved relative to the repository root.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "code/results/cdf/rr_cdf_dense_M10000_T20k_1M_summary.csv"
SAMPLES = ROOT / "code/results/cdf/rr_cdf_dense_M10000_T20k_1M_z.csv"
OUTPUT = Path(__file__).with_name("rr_main_experiment.pdf")

METHODS = ("single_2alpha", "single_alpha", "RR")
LABELS = {
    "single_2alpha": r"PR, $2\alpha_n$",
    "single_alpha": r"PR, $\alpha_n$",
    "RR": "RR",
}
COLORS = {
    "single_2alpha": "#b22222",
    "single_alpha": "#d97706",
    "RR": "#1f5aa6",
}
LINESTYLES = {
    "single_2alpha": (0, (5, 2)),
    "single_alpha": (0, (2, 1.5)),
    "RR": "-",
}
MARKERS = {"single_2alpha": "s", "single_alpha": "^", "RR": "o"}


def read_summary() -> dict[str, list[dict[str, float]]]:
    rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    with SUMMARY.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            method = row["method"]
            if method not in METHODS:
                continue
            rows[method].append(
                {
                    "n": float(row["n"]),
                    "ks": float(row["ks_D"]),
                }
            )
    for method in METHODS:
        rows[method].sort(key=lambda item: item["n"])
    return rows


def read_endpoint_samples() -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = defaultdict(list)
    with SAMPLES.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["n"] == "1000000" and row["method"] in METHODS:
                value = float(row["Z"])
                if np.isfinite(value):
                    values[row["method"]].append(value)
    return {method: np.asarray(values[method], dtype=float) for method in METHODS}


def normal_density(x: np.ndarray) -> np.ndarray:
    return np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.7,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    summary = read_summary()
    samples = read_endpoint_samples()
    fig, (ax_density, ax_ks) = plt.subplots(2, 1, figsize=(4.85, 6.15))

    bins = np.linspace(-3.5, 4.0, 61)
    for method in METHODS:
        ax_density.hist(
            samples[method],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.45,
            linestyle=LINESTYLES[method],
            color=COLORS[method],
            label=LABELS[method],
        )
    x_grid = np.linspace(-3.5, 4.0, 700)
    ax_density.plot(
        x_grid,
        normal_density(x_grid),
        color="#111111",
        linewidth=1.2,
        linestyle=(0, (7, 2)),
        label=r"$N(0,1)$",
    )
    ax_density.axvline(0.0, color="#555555", linewidth=0.7, alpha=0.55)
    ax_density.set_xlim(-3.5, 4.0)
    ax_density.set_xlabel(r"нормированная статистика $Z_n$")
    ax_density.set_ylabel("эмпирическая плотность")
    ax_density.set_title(r"(а) Распределение при $n=10^6$")
    ax_density.legend(frameon=False, ncol=2, loc="upper right")

    for method in METHODS:
        n_values = np.asarray([row["n"] for row in summary[method]])
        ks_values = np.asarray([row["ks"] for row in summary[method]])
        ax_ks.plot(
            n_values,
            ks_values,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            markersize=3.8,
            linewidth=1.45,
            label=LABELS[method],
        )
    n_trajectories = len(samples["RR"])
    mc_threshold = np.sqrt(np.log(2.0 / 0.05) / (2.0 * n_trajectories))
    ax_ks.axhline(
        mc_threshold,
        color="#222222",
        linewidth=1.0,
        linestyle=(0, (7, 2)),
        label="выборочная погрешность (95%)",
    )
    ax_ks.set_xscale("log")
    ax_ks.set_yscale("log")
    ax_ks.set_xlabel(r"горизонт $n$")
    ax_ks.set_ylabel("расстояние Колмогорова")
    ax_ks.set_title("(б) Отклонение от стандартного нормального закона")
    ax_ks.legend(frameon=False, ncol=2, loc="center right")

    fig.tight_layout(h_pad=1.25, pad=0.35)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.025)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
