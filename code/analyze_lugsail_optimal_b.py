"""Analyze optimal OBM / OBM-LW block-size scaling from experiment CSVs.

Inputs can be either:
  - `run_lugsail_decomposition.py` output, with columns `T,b,estimator,lam,mse`.
  - `run_lugsail_bias_variance.py` aggregated output, with columns
    `T,iterate,estimator,lam,mse_mean`.

For every method group the script:
  1. picks the swept block size b* minimizing MSE at each T;
  2. fits b* = C T^eta and MSE* = D T^kappa on the available T values;
  3. scans candidate normalizations b* / T^x and records which x makes the
     ratio most stable across T;
  4. writes CSV summaries and compact diagnostic figures.
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


def _method_label(row_or_key) -> str:
    """Human-readable method label from a Series or tuple-like key."""
    if isinstance(row_or_key, pd.Series):
        iterate = row_or_key.get("iterate", "")
        estimator = row_or_key["estimator"]
        lam = float(row_or_key["lam"])
        source = row_or_key.get("source", "")
    else:
        names, values = row_or_key
        data = dict(zip(names, values))
        iterate = data.get("iterate", "")
        estimator = data["estimator"]
        lam = float(data["lam"])
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


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = C x^eta. Returns (eta, C, r2) on the log scale."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    lx = np.log(x)
    ly = np.log(y)
    eta, log_c = np.polyfit(lx, ly, 1)
    pred = eta * lx + log_c
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(eta), float(np.exp(log_c)), r2


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
        raise SystemExit("input CSV must contain either `mse` or `mse_mean`")

    if "bias" not in df.columns and "bias_mean" in df.columns:
        df["bias"] = df["bias_mean"]
    if "rel_bias" not in df.columns and "rel_bias_median" in df.columns:
        df["rel_bias"] = df["rel_bias_median"]
    if "lam" not in df.columns:
        df["lam"] = 0.0
    if "estimator" not in df.columns:
        raise SystemExit("input CSV must contain `estimator`")

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
    """Average duplicate rows on the same (method, T, b) grid point."""
    agg_spec = {"mse_selected": "mean"}
    for optional in ["bias", "rel_bias", "var", "var_mean", "sigma_true"]:
        if optional in df.columns:
            agg_spec[optional] = "mean"
    return (
        df.groupby(group_cols + ["T", "b"], dropna=False)
        .agg(agg_spec)
        .reset_index()
    )


def _best_by_t(grid: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, grp in grid.groupby(group_cols + ["T"], dropna=False):
        sub = grp[np.isfinite(grp["mse_selected"]) & (grp["mse_selected"] > 0)]
        if sub.empty:
            continue
        best = sub.loc[sub["mse_selected"].idxmin()].copy()
        best["eta_eff"] = math.log(best["b"]) / math.log(best["T"])
        rows.append(best)
    if not rows:
        raise SystemExit("no finite positive MSE rows found")
    return pd.DataFrame(rows).reset_index(drop=True)


def _scan_x(best: pd.DataFrame, group_cols: list[str],
            x_grid: list[float]) -> pd.DataFrame:
    rows = []
    for key, grp in best.groupby(group_cols, dropna=False):
        grp = grp.sort_values("T")
        T = grp["T"].to_numpy(float)
        b = grp["b"].to_numpy(float)
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key_values))
        for x in x_grid:
            ratio = b / (T ** x)
            eta_ratio, _, _ = _fit_power_law(T, ratio)
            rows.append({
                **base,
                "x": x,
                "ratio_mean": float(np.mean(ratio)),
                "ratio_std": float(np.std(ratio, ddof=1))
                if len(ratio) > 1 else float("nan"),
                "ratio_cv": float(np.std(ratio, ddof=1) / np.mean(ratio))
                if len(ratio) > 1 and np.mean(ratio) != 0 else float("nan"),
                "ratio_min": float(np.min(ratio)),
                "ratio_max": float(np.max(ratio)),
                "log_ratio_slope": eta_ratio,
                "abs_log_ratio_slope": abs(eta_ratio)
                if np.isfinite(eta_ratio) else float("nan"),
            })
    return pd.DataFrame(rows)


def _scaling_summary(best: pd.DataFrame, x_scan: pd.DataFrame,
                     group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, grp in best.groupby(group_cols, dropna=False):
        grp = grp.sort_values("T")
        T = grp["T"].to_numpy(float)
        b = grp["b"].to_numpy(float)
        mse = grp["mse_selected"].to_numpy(float)
        eta, c_b, r2_b = _fit_power_law(T, b)
        kappa, c_mse, r2_mse = _fit_power_law(T, mse)

        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key_values))
        scan = x_scan
        for col, value in base.items():
            scan = scan[scan[col] == value]
        stable = scan.dropna(subset=["ratio_cv"])
        best_cv = stable.loc[stable["ratio_cv"].idxmin()] if not stable.empty else None
        stable_slope = scan.dropna(subset=["abs_log_ratio_slope"])
        best_slope = (
            stable_slope.loc[stable_slope["abs_log_ratio_slope"].idxmin()]
            if not stable_slope.empty else None
        )

        rows.append({
            **base,
            "label": _method_label((group_cols, key_values)),
            "n_T": int(len(grp)),
            "T_min": int(np.min(T)),
            "T_max": int(np.max(T)),
            "eta_hat_bstar": eta,
            "C_hat_bstar": c_b,
            "r2_log_bstar": r2_b,
            "mse_power_kappa": kappa,
            "D_hat_mse": c_mse,
            "r2_log_mse": r2_mse,
            "best_x_by_cv": float(best_cv["x"]) if best_cv is not None else float("nan"),
            "best_cv": float(best_cv["ratio_cv"]) if best_cv is not None else float("nan"),
            "best_x_by_slope": float(best_slope["x"])
            if best_slope is not None else float("nan"),
            "best_abs_slope": float(best_slope["abs_log_ratio_slope"])
            if best_slope is not None else float("nan"),
        })
    return pd.DataFrame(rows)


def _plot_bstar(best: pd.DataFrame, summary: pd.DataFrame,
                group_cols: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cmap = plt.get_cmap("tab10")
    for idx, (key, grp) in enumerate(best.groupby(group_cols, dropna=False)):
        grp = grp.sort_values("T")
        key_values = key if isinstance(key, tuple) else (key,)
        label = _method_label((group_cols, key_values))
        color = cmap(idx % 10)
        ax.loglog(grp["T"], grp["b"], "o-", color=color, label=label)

        row = summary
        for col, value in zip(group_cols, key_values):
            row = row[row[col] == value]
        if not row.empty and np.isfinite(row.iloc[0]["eta_hat_bstar"]):
            eta = row.iloc[0]["eta_hat_bstar"]
            c_b = row.iloc[0]["C_hat_bstar"]
            tt = np.geomspace(grp["T"].min(), grp["T"].max(), 200)
            ax.loglog(tt, c_b * tt ** eta, "--", color=color, alpha=0.75)

    ax.set_xlabel("sample size T")
    ax.set_ylabel("optimal swept block size b*")
    ax.set_title("Empirical optimal block-size scaling")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ratio_panels(best: pd.DataFrame, group_cols: list[str],
                       plot_x: list[float], out: Path) -> None:
    methods = list(best.groupby(group_cols, dropna=False))
    if not methods:
        return
    n = len(methods)
    ncols = min(2, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.3 * nrows),
        squeeze=False, sharex=False,
    )
    cmap = plt.get_cmap("viridis")
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    for idx, (key, grp) in enumerate(methods):
        ax = axes.ravel()[idx]
        grp = grp.sort_values("T")
        T = grp["T"].to_numpy(float)
        b = grp["b"].to_numpy(float)
        key_values = key if isinstance(key, tuple) else (key,)
        ax.set_title(_method_label((group_cols, key_values)), fontsize=10)
        for j, x in enumerate(plot_x):
            color = cmap(j / max(1, len(plot_x) - 1))
            ax.loglog(T, b / (T ** x), "o-", color=color,
                      linewidth=1.2, markersize=3.5, label=f"x={x:g}")
        ax.set_xlabel("T")
        ax.set_ylabel("b* / T^x")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Candidate normalizations for b*", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_x_scan(x_scan: pd.DataFrame, group_cols: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    cmap = plt.get_cmap("tab10")
    for idx, (key, grp) in enumerate(x_scan.groupby(group_cols, dropna=False)):
        grp = grp.sort_values("x")
        key_values = key if isinstance(key, tuple) else (key,)
        label = _method_label((group_cols, key_values))
        y = grp["ratio_cv"]
        if y.isna().all():
            y = grp["abs_log_ratio_slope"]
            ylabel = "|slope of log(b*/T^x)|"
        else:
            ylabel = "CV of b*/T^x over T"
        ax.plot(grp["x"], y, "-", color=cmap(idx % 10), label=label)
        finite = np.isfinite(y)
        if finite.any():
            j = y[finite].idxmin()
            ax.scatter(grp.loc[j, "x"], y.loc[j], color=cmap(idx % 10),
                       edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_xlabel("candidate exponent x")
    ax.set_ylabel(ylabel)
    ax.set_title("Stability scan for b* / T^x")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze asymptotic scaling of optimal OBM block sizes.",
    )
    p.add_argument("csv", nargs="+", type=Path,
                   help="CSV files from lugsail decomposition or BV runs.")
    p.add_argument("--outdir", type=Path,
                   default=Path("../reports/figures/lugsail_optimal_b"))
    p.add_argument("--prefix", default="lugsail_optimal_b")
    p.add_argument("--x-grid", type=_parse_grid, default=_parse_grid("0.20:0.70:0.025"),
                   help="Candidate x grid, either start:stop:step or comma list.")
    p.add_argument("--plot-x", type=_parse_grid,
                   default=_parse_grid("0.20,0.25,0.333333,0.40,0.50,0.60"),
                   help="Candidate x values to show in ratio-panel figure.")
    p.add_argument("--split-source", action="store_true",
                   help="Keep input files as separate method families.")
    args = p.parse_args()

    df = _load_inputs(args.csv, args.split_source)
    group_cols = _group_columns(df, args.split_source)
    grid = _aggregate_grid(df, group_cols)
    best = _best_by_t(grid, group_cols)
    x_scan = _scan_x(best, group_cols, args.x_grid)
    summary = _scaling_summary(best, x_scan, group_cols)

    args.outdir.mkdir(parents=True, exist_ok=True)
    best_out = args.outdir / f"{args.prefix}_bstar.csv"
    scan_out = args.outdir / f"{args.prefix}_x_scan.csv"
    summary_out = args.outdir / f"{args.prefix}_scaling_summary.csv"
    best.to_csv(best_out, index=False)
    x_scan.to_csv(scan_out, index=False)
    summary.to_csv(summary_out, index=False)

    _plot_bstar(best, summary, group_cols,
                args.outdir / f"{args.prefix}_bstar_scaling.png")
    _plot_ratio_panels(best, group_cols, args.plot_x,
                       args.outdir / f"{args.prefix}_bstar_over_Tx.png")
    _plot_x_scan(x_scan, group_cols,
                 args.outdir / f"{args.prefix}_x_scan.png")

    show_cols = [
        "label", "n_T", "eta_hat_bstar", "C_hat_bstar", "r2_log_bstar",
        "mse_power_kappa", "best_x_by_cv", "best_cv", "best_x_by_slope",
    ]
    print(summary[show_cols].to_string(index=False))
    print(f"\nSaved {best_out}")
    print(f"Saved {scan_out}")
    print(f"Saved {summary_out}")


if __name__ == "__main__":
    main()
