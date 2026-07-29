"""Estimate separate bias and variance rates for OBM / OBM-LW.

This script tests the refined finite-T hypothesis:

  * variance should already follow Var(sigma_hat^2) ~= const * b / T,
    so with b = T^eta the rate is 1 - eta;
  * bias should follow |Bias| ~= b^{-1} for OBM and b^{-2} for OBM-LW
    only after the block size has entered the asymptotic truncation regime.

Inputs are CSVs from `run_lugsail_decomposition.py` or aggregated outputs from
`run_lugsail_bias_variance.py`.  For every method and every eta in the grid,
the script selects the closest swept b to T^eta and fits component power laws:

    |bias|   ~= C T^{-q_abs}
    bias^2   ~= C T^{-q_sq}
    variance ~= C T^{-s}
    MSE      ~= C T^{-r}
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_grid(spec: str) -> list[float]:
    if ":" in spec:
        start, stop, step = [float(x) for x in spec.split(":")]
        if step <= 0:
            raise argparse.ArgumentTypeError("grid step must be positive")
        n = int(math.floor((stop - start) / step + 0.5)) + 1
        return [round(start + i * step, 10) for i in range(max(n, 0))]
    return [float(x) for x in spec.split(",") if x.strip()]


def _label(row_or_key) -> str:
    if isinstance(row_or_key, pd.Series):
        estimator = row_or_key["estimator"]
        lam = float(row_or_key["lam"])
        iterate = row_or_key.get("iterate", "")
        source = row_or_key.get("source", "")
    else:
        names, values = row_or_key
        data = dict(zip(names, values))
        estimator = data["estimator"]
        lam = float(data["lam"])
        iterate = data.get("iterate", "")
        source = data.get("source", "")

    bits = []
    if source:
        bits.append(str(source))
    if iterate:
        bits.append(str(iterate))
    if estimator == "OBM":
        bits.append("OBM")
    else:
        bits.append(f"OBM-LW(lambda={lam:g})")
    return " / ".join(bits)


def _load_inputs(paths: list[Path], split_source: bool) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if split_source:
            df["source"] = path.stem
        frames.append(df)
    df = pd.concat(frames, ignore_index=True, sort=False)

    if "mse" in df.columns:
        df["mse_selected"] = df["mse"]
    elif "mse_mean" in df.columns:
        df["mse_selected"] = df["mse_mean"]
    else:
        raise SystemExit("input CSV must contain `mse` or `mse_mean`")

    if "bias" not in df.columns and "bias_mean" in df.columns:
        df["bias"] = df["bias_mean"]
    if "var" not in df.columns and "var_mean" in df.columns:
        df["var"] = df["var_mean"]
    if "bias" not in df.columns:
        raise SystemExit("input CSV must contain `bias` or `bias_mean`")
    if "var" not in df.columns:
        raise SystemExit("input CSV must contain `var` or `var_mean`")
    if "lam" not in df.columns:
        df["lam"] = 0.0

    needed = ["T", "b", "estimator", "lam", "bias", "var", "mse_selected"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise SystemExit(f"input CSV missing required columns: {missing}")
    return df


def _group_columns(df: pd.DataFrame, split_source: bool) -> list[str]:
    cols = []
    if split_source and "source" in df.columns:
        cols.append("source")
    if "iterate" in df.columns:
        cols.append("iterate")
    cols.extend(["estimator", "lam"])
    return cols


def _aggregate_grid(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols + ["T", "b"], dropna=False)
        .agg(
            bias=("bias", "mean"),
            var=("var", "mean"),
            mse_selected=("mse_selected", "mean"),
        )
        .reset_index()
    )


def _fit_power_law(T: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = C T^kappa. Returns (kappa, C, log-scale R2)."""
    mask = (T > 0) & (y > 0) & np.isfinite(T) & np.isfinite(y)
    T = T[mask]
    y = y[mask]
    if len(T) < 2:
        return float("nan"), float("nan"), float("nan")
    x_log = np.log(T)
    y_log = np.log(y)
    kappa, log_c = np.polyfit(x_log, y_log, 1)
    pred = kappa * x_log + log_c
    ss_res = float(np.sum((y_log - pred) ** 2))
    ss_tot = float(np.sum((y_log - y_log.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(kappa), float(np.exp(log_c)), r2


def _theory_abs_bias_rate(estimator: str, eta: float) -> float:
    return eta if estimator == "OBM" else 2.0 * eta


def _theory_bias_sq_rate(estimator: str, eta: float) -> float:
    return 2.0 * eta if estimator == "OBM" else 4.0 * eta


def _theory_var_rate(eta: float) -> float:
    return 1.0 - eta


def _select_eta_rows(grid: pd.DataFrame, group_cols: list[str],
                     eta_grid: list[float],
                     max_log_b_error: float | None) -> pd.DataFrame:
    rows = []
    for key, grp in grid.groupby(group_cols + ["T"], dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols + ["T"], key_values))
        T = float(base["T"])
        grp = grp[
            np.isfinite(grp["bias"])
            & np.isfinite(grp["var"])
            & np.isfinite(grp["mse_selected"])
            & (grp["var"] > 0)
            & (grp["mse_selected"] > 0)
        ]
        if grp.empty:
            continue
        for eta in eta_grid:
            target = T ** eta
            dist = np.abs(np.log(grp["b"].to_numpy(float) / target))
            j = int(np.argmin(dist))
            if max_log_b_error is not None and dist[j] > max_log_b_error:
                continue
            row = grp.iloc[j].copy()
            row["eta"] = eta
            row["target_b"] = target
            row["eta_actual"] = math.log(float(row["b"])) / math.log(T)
            row["log_b_error"] = float(dist[j])
            row["abs_bias"] = abs(float(row["bias"]))
            row["bias_sq"] = float(row["bias"]) ** 2
            row["bias_sq_plus_var"] = row["bias_sq"] + float(row["var"])
            rows.append(row)
    if not rows:
        raise SystemExit("no rows selected")
    return pd.DataFrame(rows).reset_index(drop=True)


def _rate_summary(values: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    components = [
        ("abs_bias", "abs_bias"),
        ("bias_sq", "bias_sq"),
        ("var", "var"),
        ("mse", "mse_selected"),
        ("bias_sq_plus_var", "bias_sq_plus_var"),
    ]
    for key, grp in values.groupby(group_cols + ["eta"], dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols + ["eta"], key_values))
        grp = grp.sort_values("T")
        T = grp["T"].to_numpy(float)
        estimator = str(base["estimator"])
        eta = float(base["eta"])
        row = {
            **base,
            "label": _label((group_cols, key_values[:len(group_cols)])),
            "n_T": int(len(grp)),
            "T_min": int(np.min(T)),
            "T_max": int(np.max(T)),
            "theory_abs_bias_rate": _theory_abs_bias_rate(estimator, eta),
            "theory_bias_sq_rate": _theory_bias_sq_rate(estimator, eta),
            "theory_var_rate": _theory_var_rate(eta),
            "mean_abs_log_b_error": float(np.mean(grp["log_b_error"])),
            "max_abs_log_b_error": float(np.max(grp["log_b_error"])),
        }
        for name, col in components:
            kappa, c_value, r2 = _fit_power_law(T, grp[col].to_numpy(float))
            row[f"{name}_kappa"] = kappa
            row[f"{name}_rate"] = -kappa
            row[f"{name}_C"] = c_value
            row[f"{name}_r2"] = r2
        row["abs_bias_rate_minus_theory"] = (
            row["abs_bias_rate"] - row["theory_abs_bias_rate"]
        )
        row["bias_sq_rate_minus_theory"] = (
            row["bias_sq_rate"] - row["theory_bias_sq_rate"]
        )
        row["var_rate_minus_theory"] = row["var_rate"] - row["theory_var_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def _method_groups(df: pd.DataFrame, group_cols: list[str]):
    for key, grp in df.groupby(group_cols, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        yield key_values, grp


def _plot_rate(summary: pd.DataFrame, group_cols: list[str],
               component: str, theory_col: str, title: str, ylabel: str,
               out: Path) -> None:
    methods = list(_method_groups(summary, group_cols))
    n = len(methods)
    ncols = min(2, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.5 * nrows),
                             squeeze=False, sharex=True, sharey=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for idx, (key_values, grp) in enumerate(methods):
        ax = axes.ravel()[idx]
        grp = grp.sort_values("eta")
        ax.plot(grp["eta"], grp[f"{component}_rate"], "o-",
                label="empirical")
        ax.plot(grp["eta"], grp[theory_col], "--", label="theory")
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.4)
        ax.set_title(_label((group_cols, key_values)))
        ax.set_xlabel("block exponent eta")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_component_contrib(values: pd.DataFrame, group_cols: list[str],
                            plot_eta: list[float], out: Path) -> None:
    methods = list(_method_groups(values, group_cols))
    n = len(methods)
    ncols = min(2, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.8 * nrows),
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    for idx, (key_values, grp) in enumerate(methods):
        ax = axes.ravel()[idx]
        available = np.sort(grp["eta"].unique().astype(float))
        etas = []
        for eta in plot_eta:
            val = float(available[int(np.argmin(np.abs(available - eta)))])
            if val not in etas:
                etas.append(val)
        ax.set_title(_label((group_cols, key_values)))
        for j, eta in enumerate(etas):
            color = cmap(j / max(1, len(etas) - 1))
            sub = grp[np.isclose(grp["eta"], eta)].sort_values("T")
            frac = sub["bias_sq"] / sub["mse_selected"]
            ax.semilogx(sub["T"], frac, "o-", color=color,
                        label=f"eta={eta:g}")
        ax.set_xlabel("T")
        ax.set_ylabel("bias^2 / MSE")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Bias contribution to MSE", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Estimate separate bias/variance asymptotic rates.",
    )
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument("--outdir", type=Path,
                   default=Path("../reports/figures/lugsail_component_rates"))
    p.add_argument("--prefix", default="lugsail_components")
    p.add_argument("--eta-grid", type=_parse_grid,
                   default=_parse_grid("0.15:0.75:0.025"))
    p.add_argument("--plot-eta", type=_parse_grid,
                   default=_parse_grid("0.20,0.25,0.333333,0.40,0.45,0.50,0.60"))
    p.add_argument("--max-log-b-error", type=float, default=None)
    p.add_argument("--split-source", action="store_true")
    args = p.parse_args()

    df = _load_inputs(args.csv, args.split_source)
    group_cols = _group_columns(df, args.split_source)
    grid = _aggregate_grid(df, group_cols)
    values = _select_eta_rows(
        grid, group_cols, args.eta_grid, args.max_log_b_error,
    )
    summary = _rate_summary(values, group_cols)

    args.outdir.mkdir(parents=True, exist_ok=True)
    values_out = args.outdir / f"{args.prefix}_eta_values.csv"
    summary_out = args.outdir / f"{args.prefix}_component_rates.csv"
    values.to_csv(values_out, index=False)
    summary.to_csv(summary_out, index=False)

    _plot_rate(
        summary, group_cols, "abs_bias", "theory_abs_bias_rate",
        "Absolute bias decay rate by fixed block exponent",
        "rate q in |bias| ~ T^{-q}",
        args.outdir / f"{args.prefix}_abs_bias_rate_vs_eta.png",
    )
    _plot_rate(
        summary, group_cols, "bias_sq", "theory_bias_sq_rate",
        "Squared-bias decay rate by fixed block exponent",
        "rate q in bias^2 ~ T^{-q}",
        args.outdir / f"{args.prefix}_bias_sq_rate_vs_eta.png",
    )
    _plot_rate(
        summary, group_cols, "var", "theory_var_rate",
        "Variance decay rate by fixed block exponent",
        "rate s in Var ~ T^{-s}",
        args.outdir / f"{args.prefix}_var_rate_vs_eta.png",
    )
    _plot_component_contrib(
        values, group_cols, args.plot_eta,
        args.outdir / f"{args.prefix}_bias_fraction.png",
    )

    show_cols = [
        "label", "eta",
        "abs_bias_rate", "theory_abs_bias_rate", "abs_bias_rate_minus_theory",
        "var_rate", "theory_var_rate", "var_rate_minus_theory",
        "bias_sq_rate", "mse_rate",
    ]
    best_mse = (
        summary.sort_values("mse_rate", ascending=False)
        .groupby(group_cols, dropna=False)
        .head(1)
    )
    print(best_mse[show_cols].to_string(index=False))
    print(f"\nSaved {values_out}")
    print(f"Saved {summary_out}")


if __name__ == "__main__":
    main()
