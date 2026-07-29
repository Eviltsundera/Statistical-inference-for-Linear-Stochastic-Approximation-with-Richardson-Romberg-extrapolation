"""Estimate MSE decay rates for OBM / OBM-LW at fixed b = T^eta.

This is complementary to `analyze_lugsail_optimal_b.py`.

Instead of first minimizing over b and then fitting b*(T), this script fixes a
candidate block exponent eta, selects the closest swept block size b ~= T^eta
for every T, and fits

    MSE(T, eta) ~= C(eta) * T^kappa(eta)

on the log scale.  The reported positive rate is r_hat(eta) = -kappa(eta).

Theory templates used for comparison:
  OBM:    MSE ~= b^{-2} + b/T  => r(eta) = min(2 eta, 1 - eta)
  OBM-LW: MSE ~= b^{-4} + b/T  => r(eta) = min(4 eta, 1 - eta)

The input is typically produced by `run_lugsail_decomposition.py`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_grid(spec: str) -> list[float]:
    """Parse either `a:b:step` or a comma-separated float list."""
    if ":" in spec:
        parts = [float(x) for x in spec.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "grid range must have form start:stop:step"
            )
        start, stop, step = parts
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


def _load_inputs(paths: list[Path], split_source: bool,
                 use_clamped: bool) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if split_source:
            df["source"] = path.stem
        frames.append(df)
    df = pd.concat(frames, ignore_index=True, sort=False)

    if use_clamped and "mse_clamped" in df.columns:
        df["mse_selected"] = df["mse_clamped"].where(
            df["mse_clamped"].notna(), df.get("mse")
        )
    elif "mse" in df.columns:
        df["mse_selected"] = df["mse"]
    elif "mse_mean" in df.columns:
        df["mse_selected"] = df["mse_mean"]
    else:
        raise SystemExit("input CSV must contain `mse`, `mse_mean`, or `mse_clamped`")

    if "lam" not in df.columns:
        df["lam"] = 0.0
    needed = ["T", "b", "estimator", "lam", "mse_selected"]
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
    agg_spec = {"mse_selected": "mean"}
    for optional in ["bias", "bias_mean", "var", "var_mean", "sigma_true"]:
        if optional in df.columns:
            agg_spec[optional] = "mean"
    return (
        df.groupby(group_cols + ["T", "b"], dropna=False)
        .agg(agg_spec)
        .reset_index()
    )


def _fit_power_law(T: np.ndarray, mse: np.ndarray) -> tuple[float, float, float]:
    """Fit mse = C T^kappa. Returns (kappa, C, r2) on the log scale."""
    mask = (T > 0) & (mse > 0) & np.isfinite(T) & np.isfinite(mse)
    T = T[mask]
    mse = mse[mask]
    if len(T) < 2:
        return float("nan"), float("nan"), float("nan")
    x = np.log(T)
    y = np.log(mse)
    kappa, log_c = np.polyfit(x, y, 1)
    pred = kappa * x + log_c
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(kappa), float(np.exp(log_c)), r2


def _theory_rate(estimator: str, eta: float) -> float:
    if estimator == "OBM":
        return min(2.0 * eta, 1.0 - eta)
    return min(4.0 * eta, 1.0 - eta)


def _select_eta_rows(grid: pd.DataFrame, group_cols: list[str],
                     eta_grid: list[float],
                     max_log_b_error: float | None) -> pd.DataFrame:
    rows = []
    for key, grp in grid.groupby(group_cols + ["T"], dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols + ["T"], key_values))
        grp = grp[np.isfinite(grp["mse_selected"]) & (grp["mse_selected"] > 0)]
        if grp.empty:
            continue
        T = float(base["T"])
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
            rows.append(row)
    if not rows:
        raise SystemExit("no rows selected for eta grid")
    return pd.DataFrame(rows).reset_index(drop=True)


def _rate_summary(values: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, grp in values.groupby(group_cols + ["eta"], dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols + ["eta"], key_values))
        grp = grp.sort_values("T")
        T = grp["T"].to_numpy(float)
        mse = grp["mse_selected"].to_numpy(float)
        kappa, c_mse, r2 = _fit_power_law(T, mse)
        estimator = str(base["estimator"])
        eta = float(base["eta"])
        rows.append({
            **base,
            "label": _label((group_cols, key_values[:len(group_cols)])),
            "n_T": int(len(grp)),
            "T_min": int(np.min(T)),
            "T_max": int(np.max(T)),
            "kappa_hat": kappa,
            "rate_hat": -kappa,
            "C_hat_mse": c_mse,
            "r2_log_mse": r2,
            "theory_rate": _theory_rate(estimator, eta),
            "rate_minus_theory": -kappa - _theory_rate(estimator, eta),
            "mean_abs_log_b_error": float(np.mean(grp["log_b_error"])),
            "max_abs_log_b_error": float(np.max(grp["log_b_error"])),
        })
    return pd.DataFrame(rows)


def _method_groups(df: pd.DataFrame, group_cols: list[str]):
    for key, grp in df.groupby(group_cols, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        yield key_values, grp


def _plot_rate_vs_eta(summary: pd.DataFrame, group_cols: list[str],
                      out: Path) -> None:
    methods = list(_method_groups(summary, group_cols))
    n = len(methods)
    ncols = min(2, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.5 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    for idx, (key_values, grp) in enumerate(methods):
        ax = axes.ravel()[idx]
        grp = grp.sort_values("eta")
        label = _label((group_cols, key_values))
        ax.plot(grp["eta"], grp["rate_hat"], "o-", label="empirical")
        ax.plot(grp["eta"], grp["theory_rate"], "--", label="theory")
        best = grp.loc[grp["rate_hat"].idxmax()]
        ax.scatter(best["eta"], best["rate_hat"], marker="*", s=120,
                   edgecolor="black", zorder=3, label="best empirical")
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.4)
        ax.set_title(label)
        ax.set_xlabel("block exponent eta")
        ax.set_ylabel("MSE decay rate r in MSE ~ T^{-r}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("MSE decay rate by fixed block exponent", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _nearest_plot_etas(available: np.ndarray, requested: list[float]) -> list[float]:
    out = []
    for eta in requested:
        j = int(np.argmin(np.abs(available - eta)))
        val = float(available[j])
        if val not in out:
            out.append(val)
    return out


def _plot_mse_vs_t(values: pd.DataFrame, summary: pd.DataFrame,
                   group_cols: list[str], plot_eta: list[float],
                   out: Path) -> None:
    methods = list(_method_groups(values, group_cols))
    n = len(methods)
    ncols = min(2, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.7 * nrows),
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    for idx, (key_values, grp) in enumerate(methods):
        ax = axes.ravel()[idx]
        label = _label((group_cols, key_values))
        available = np.sort(grp["eta"].unique().astype(float))
        etas = _nearest_plot_etas(available, plot_eta)
        ax.set_title(label)
        for j, eta in enumerate(etas):
            color = cmap(j / max(1, len(etas) - 1))
            sub = grp[np.isclose(grp["eta"], eta)].sort_values("T")
            ax.loglog(sub["T"], sub["mse_selected"], "o-",
                      color=color, label=f"eta={eta:g}")
            row = summary
            for col, value in zip(group_cols, key_values):
                row = row[row[col] == value]
            row = row[np.isclose(row["eta"], eta)]
            if not row.empty and np.isfinite(row.iloc[0]["kappa_hat"]):
                kappa = row.iloc[0]["kappa_hat"]
                c_mse = row.iloc[0]["C_hat_mse"]
                tt = np.geomspace(sub["T"].min(), sub["T"].max(), 100)
                ax.loglog(tt, c_mse * tt ** kappa, "--",
                          color=color, alpha=0.75)
        ax.set_xlabel("T")
        ax.set_ylabel("MSE")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle("MSE versus T at fixed b = T^eta", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scaled_mse(values: pd.DataFrame, group_cols: list[str],
                     plot_eta: list[float], out: Path) -> None:
    methods = list(_method_groups(values, group_cols))
    n = len(methods)
    ncols = min(2, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.7 * nrows),
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    for idx, (key_values, grp) in enumerate(methods):
        ax = axes.ravel()[idx]
        label = _label((group_cols, key_values))
        estimator = str(key_values[group_cols.index("estimator")])
        available = np.sort(grp["eta"].unique().astype(float))
        etas = _nearest_plot_etas(available, plot_eta)
        ax.set_title(label)
        for j, eta in enumerate(etas):
            color = cmap(j / max(1, len(etas) - 1))
            sub = grp[np.isclose(grp["eta"], eta)].sort_values("T")
            rate = _theory_rate(estimator, eta)
            scaled = sub["mse_selected"] * (sub["T"].astype(float) ** rate)
            ax.loglog(sub["T"], scaled, "o-", color=color,
                      label=f"eta={eta:g}, r_th={rate:.2f}")
        ax.set_xlabel("T")
        ax.set_ylabel("MSE * T^{r_theory(eta)}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=1)

    fig.suptitle("Theory-scaled MSE diagnostics", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Estimate MSE asymptotic rates at fixed b=T^eta.",
    )
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument("--outdir", type=Path,
                   default=Path("../reports/figures/lugsail_mse_asymptotics"))
    p.add_argument("--prefix", default="lugsail_mse")
    p.add_argument("--eta-grid", type=_parse_grid,
                   default=_parse_grid("0.15:0.75:0.025"))
    p.add_argument("--plot-eta", type=_parse_grid,
                   default=_parse_grid("0.20,0.25,0.333333,0.40,0.50,0.60"))
    p.add_argument("--max-log-b-error", type=float, default=None,
                   help="Drop selected rows with |log(b/T^eta)| above this.")
    p.add_argument("--split-source", action="store_true")
    p.add_argument("--use-clamped", action="store_true",
                   help="Use mse_clamped for OBM-LW if present.")
    args = p.parse_args()

    df = _load_inputs(args.csv, args.split_source, args.use_clamped)
    group_cols = _group_columns(df, args.split_source)
    grid = _aggregate_grid(df, group_cols)
    values = _select_eta_rows(
        grid, group_cols, args.eta_grid, args.max_log_b_error,
    )
    summary = _rate_summary(values, group_cols)

    args.outdir.mkdir(parents=True, exist_ok=True)
    values_out = args.outdir / f"{args.prefix}_eta_values.csv"
    summary_out = args.outdir / f"{args.prefix}_eta_rates.csv"
    values.to_csv(values_out, index=False)
    summary.to_csv(summary_out, index=False)

    _plot_rate_vs_eta(
        summary, group_cols, args.outdir / f"{args.prefix}_rate_vs_eta.png",
    )
    _plot_mse_vs_t(
        values, summary, group_cols, args.plot_eta,
        args.outdir / f"{args.prefix}_mse_vs_T.png",
    )
    _plot_scaled_mse(
        values, group_cols, args.plot_eta,
        args.outdir / f"{args.prefix}_scaled_mse.png",
    )

    best = (
        summary.sort_values("rate_hat", ascending=False)
        .groupby(group_cols, dropna=False)
        .head(1)
    )
    show_cols = [
        "label", "eta", "rate_hat", "theory_rate", "rate_minus_theory",
        "r2_log_mse", "n_T", "max_abs_log_b_error",
    ]
    print(best[show_cols].to_string(index=False))
    print(f"\nSaved {values_out}")
    print(f"Saved {summary_out}")


if __name__ == "__main__":
    main()
