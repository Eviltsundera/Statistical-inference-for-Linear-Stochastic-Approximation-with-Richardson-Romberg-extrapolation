"""Plot MSE curves and parametric fits from `run_lugsail_decomposition` output.

Reads a CSV produced by `run_lugsail_decomposition.py` and emits:
  - mse_vs_b.png            log-log MSE vs b, faceted by T (overview, all methods)
  - mse_fit_OBM.png         per-method fit + model formula
  - mse_fit_OBM_RR_lam2.png
  - mse_fit_OBM_RR_lam3.png
  - mse_fit_OBM_RR_lam4.png

The parametric model pooled across T:
  OBM:    MSE(b, T) = A / b^2 + C * b / T     (leading bias^2 + variance)
  OBM_RR: MSE(b, T) = A / b^4 + C * b / T     (after 1/b cancellation)

Fit via least squares on log(MSE) (so all decades weigh equally).
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------------
# Plot styling helpers
# ----------------------------------------------------------------------------

def _label(est, lam):
    if est == 'OBM':
        return 'OBM'
    return f'OBM-RR (λ={int(lam) if float(lam).is_integer() else lam})'


def _color_for_method(est, lam):
    if est == 'OBM':
        return 'tab:blue'
    return {2.0: 'tab:orange', 3.0: 'tab:green', 4.0: 'tab:red'}.get(
        float(lam), 'tab:purple',
    )


def _color_for_T(T, T_list):
    """Distinct color per T using viridis, ordered by T."""
    cmap = plt.get_cmap('viridis')
    idx = sorted(T_list).index(T)
    return cmap(idx / max(1, len(T_list) - 1) * 0.85)


def _series_iter(df_sub):
    """Yield (label, color, sub_df) for each (estimator, lam) group."""
    for (est, lam), grp in df_sub.groupby(['estimator', 'lam']):
        yield _label(est, lam), _color_for_method(est, lam), grp.sort_values('b')


# ----------------------------------------------------------------------------
# Overview: MSE vs b, faceted by T (kept as the high-level comparison)
# ----------------------------------------------------------------------------

def plot_mse_vs_b(df, out_path):
    """log-log MSE vs b, faceted by T, all methods on each panel."""
    Ts = sorted(df['T'].unique())
    n = len(Ts)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, T in zip(axes, Ts):
        sub = df[df['T'] == T]
        for label, color, grp in _series_iter(sub):
            mask = (grp['mse'] > 0) & np.isfinite(grp['mse'])
            ax.loglog(grp['b'][mask], grp['mse'][mask], 'o-', ms=3, lw=1,
                      label=label, color=color, alpha=0.85)
            if mask.any():
                idx = grp.loc[mask, 'mse'].idxmin()
                ax.scatter(grp.loc[idx, 'b'], grp.loc[idx, 'mse'],
                           marker='*', s=120, color=color, zorder=5,
                           edgecolor='black', linewidth=0.5)
        ax.set_xlabel(r'block size $b$')
        ax.set_title(rf'$T = {T:,}$')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel(r'$\mathrm{MSE}[\hat\sigma^2(b)]$')
    axes[-1].legend(fontsize=8, loc='upper left')
    fig.suptitle(r'OBM estimator MSE vs block size $b$ (★ = optimum)',
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Per-method parametric fit
# ----------------------------------------------------------------------------

def _make_model(est, p):
    """Return (model_fn(X, *params), formula_latex, param_names).

    Structural model: MSE = bias(b, T)^2 + Var(b, T).

    From the asymptotic expansion in `obm_rr_markov_lsa_report.md`:

        bias_OBM(b)    ~ c_lead / b   + c_fs * b / T
        bias_OBM_RR(b) ~ c_lead / b^2 + c_fs * b / T   (cross-sign -> zero crossing)
        Var(b)          ~ c_var * b / T

    `c_lead` and `c_fs` keep their physical sign (negative or positive),
    so the (always-positive) bias^2 and the model are entirely
    theory-driven rather than polynomial-in-1/b. This is what gives
    OBM-RR its observed bias-crossing.
    """
    if est == 'OBM':
        def model(X, c_lead, c_fs, c_var):
            bb, TT = X
            bias = c_lead / bb + c_fs * bb / TT
            var = c_var * bb / TT
            return np.log(bias ** 2 + var)
        formula = (r"$\mathrm{MSE} = "
                   r"\left(\dfrac{c_1}{b} + c_3\,\dfrac{b}{T}\right)^2 "
                   r"+ c_4\,\dfrac{b}{T}$")
        return model, formula, ('c_1', 'c_3', 'c_4')
    else:  # OBM_RR
        def model(X, c_lead, c_fs, c_var):
            bb, TT = X
            bias = c_lead / bb ** 2 + c_fs * bb / TT
            var = c_var * bb / TT
            return np.log(bias ** 2 + var)
        formula = (r"$\mathrm{MSE} = "
                   r"\left(\dfrac{c_2}{b^{2}} + c_3'\,\dfrac{b}{T}\right)^2 "
                   r"+ c_4'\,\dfrac{b}{T}$")
        return model, formula, ('c_2', "c_3'", "c_4'")


def _fit_model(b, T, mse, bias, est, p, sigma_true):
    """Fit MSE(b, T) on the asymptotic window.

    Restricts to the regime where the decomposition applies:
        |bias| < 0.5 * sigma_true   (skip saturation at small b)
        b      < T / 3              (skip finite-sample tail at large b)

    Returns:
        params: dict {param_name: value}
        resid_std_log: std of log(MSE) - log(model) on the fit set.
        b_range_per_T: dict T -> (b_min_fit, b_max_fit) for plotting.
        model_fn, formula_latex
    """
    b = np.asarray(b, dtype=float)
    T = np.asarray(T, dtype=float)
    mse = np.asarray(mse, dtype=float)
    bias = np.asarray(bias, dtype=float)

    asymp = (
        (mse > 0) & np.isfinite(mse) & (b > 0)
        & (np.abs(bias) < 0.5 * sigma_true)
        & (b < T / 3.0)
    )
    b_f, T_f, mse_f = b[asymp], T[asymp], mse[asymp]

    model_fn, formula, names = _make_model(est, p)

    # Initial guesses (signs from theory: bias is empirically negative -> c_lead < 0)
    # |c_lead|^2 / b^p ~ MSE in small-b end (after squaring)
    abs_lead0 = float(np.sqrt(np.median(mse_f * b_f ** (2 * p))))
    c_lead0 = -abs_lead0   # bias starts negative
    c_var0 = float(np.median(mse_f * T_f / b_f))
    c_fs0 = -np.sqrt(c_var0) * 1e-3  # small initial cross-term

    bounds = ([-np.inf, -np.inf, 1e-30], [np.inf, np.inf, np.inf])

    popt, _ = curve_fit(model_fn, (b_f, T_f), np.log(mse_f),
                        p0=[c_lead0, c_fs0, c_var0],
                        bounds=bounds, maxfev=40000)
    pred = model_fn((b_f, T_f), *popt)
    resid_std = float(np.std(np.log(mse_f) - pred))

    b_range_per_T = {}
    for T_val in np.unique(T_f):
        bb = b_f[T_f == T_val]
        if len(bb) > 0:
            b_range_per_T[float(T_val)] = (float(bb.min()), float(bb.max()))

    params = dict(zip(names, [float(x) for x in popt]))
    return params, resid_std, b_range_per_T, model_fn, formula


def _fmt(coef):
    """Format coefficient for display."""
    if coef == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(coef))))
    if -3 <= exp <= 3:
        return f"{coef:.3g}"
    return f"{coef:.3e}"


def plot_mse_per_method(df, est, lam, sigma_true, out_path):
    """Per-method MSE(b, T) for all swept T, with overlaid parametric fit.

    Empirical points outside the asymptotic fit window are drawn faded;
    fit-window points are drawn solid. Model curve is drawn over the
    fit window only.
    """
    sub = df[(df['estimator'] == est) & (df['lam'] == lam)].copy()
    if sub.empty:
        return None

    Ts = sorted(sub['T'].unique())
    p = 2 if est == 'OBM' else 4

    params, resid, b_range_per_T, model_fn, formula = _fit_model(
        sub['b'], sub['T'], sub['mse'], sub['bias'],
        est, p, sigma_true,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for T in Ts:
        ggrp = sub[sub['T'] == T].sort_values('b')
        valid = (ggrp['mse'] > 0) & np.isfinite(ggrp['mse'])
        color = _color_for_T(T, Ts)

        in_window = (
            valid
            & (ggrp['bias'].abs() < 0.5 * sigma_true)
            & (ggrp['b'] < T / 3.0)
        )
        out_window = valid & ~in_window
        ax.loglog(ggrp.loc[out_window, 'b'], ggrp.loc[out_window, 'mse'],
                  'o', ms=3, color=color, alpha=0.20)
        ax.loglog(ggrp.loc[in_window, 'b'], ggrp.loc[in_window, 'mse'],
                  'o', ms=4, color=color, alpha=0.95,
                  label=rf'$T = {T:,}$')

        if valid.any():
            idx = ggrp.loc[valid, 'mse'].idxmin()
            ax.scatter(ggrp.loc[idx, 'b'], ggrp.loc[idx, 'mse'],
                       marker='*', s=130, color=color, zorder=5,
                       edgecolor='black', linewidth=0.5)

        rng = b_range_per_T.get(float(T))
        if rng is None:
            continue
        b_grid = np.geomspace(rng[0], rng[1], 200)
        mse_pred = np.exp(model_fn(
            (b_grid, np.full_like(b_grid, T)), *params.values(),
        ))
        ax.loglog(b_grid, mse_pred, '-', lw=1.6, color=color, alpha=0.85)

    coef_str = ",  ".join(f"${k} = {_fmt(v)}$" for k, v in params.items())
    note = (r"fit on $|\mathrm{bias}| < 0.5\,\sigma^2_\infty$"
            r" and $b < T/3$  (faded points excluded)")
    annotation = (
        formula + "\n" + coef_str
        + rf"$\quad$ ($\sigma_{{\log}} = {resid:.2f}$)" + "\n" + note
    )
    ax.text(0.02, 0.97, annotation,
            transform=ax.transAxes, fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', edgecolor='gray', alpha=0.9))

    ax.set_xlabel(r'block size $b$')
    ax.set_ylabel(r'$\mathrm{MSE}[\hat\sigma^2(b)]$')
    ax.set_title(_label(est, lam))
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='lower left', ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return params, resid


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def _slug(est, lam):
    if est == 'OBM':
        return 'OBM'
    return f'OBM_RR_lam{int(lam) if float(lam).is_integer() else lam}'


def main():
    p = argparse.ArgumentParser(
        description="Plot lugsail decomposition MSE curves with parametric fits",
    )
    p.add_argument('--csv', type=str,
                   default='results/lugsail_decomp_lab.csv')
    p.add_argument('--outdir', type=str,
                   default='../reports/figures')
    p.add_argument('--clean', action='store_true',
                   help='Delete previous lugsail decomposition figures first.')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    sigma_true = df['sigma_true'].iloc[0]
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"sigma^2_inf = {sigma_true:.4e}")
    print(f"T values    = {sorted(df['T'].unique())}")
    print(f"estimators  = {sorted(df['estimator'].unique())}")
    print(f"lams        = {sorted(df['lam'].unique())}")

    os.makedirs(args.outdir, exist_ok=True)

    if args.clean:
        for stale in ['bias_vs_b.png', 'variance_vs_b.png',
                      'mse_T_scaling.png', 'var_b_over_T.png']:
            path = os.path.join(args.outdir, stale)
            if os.path.exists(path):
                os.remove(path)
                print(f"  removed stale {stale}")

    # 1. Overview (faceted by T)
    plot_mse_vs_b(df, os.path.join(args.outdir, 'mse_vs_b.png'))
    print(f"  wrote mse_vs_b.png")

    # 2. Per-method fits (structural model: MSE = bias^2 + var)
    print("\n=== Per-method parametric fits "
          "(structural model: MSE = bias^2 + var) ===")
    print(f"{'method':>14}  {'c_lead':>12} {'c_fs':>12} {'c_var':>12} "
          f"{'sigma_log':>10}")
    print("-" * 66)
    for est, lam in [('OBM', 0.0),
                     ('OBM_RR', 2.0),
                     ('OBM_RR', 3.0),
                     ('OBM_RR', 4.0)]:
        if df[(df['estimator'] == est) & (df['lam'] == lam)].empty:
            continue
        out_path = os.path.join(args.outdir, f"mse_fit_{_slug(est, lam)}.png")
        result = plot_mse_per_method(df, est, lam, sigma_true, out_path)
        if result is None:
            continue
        params, resid = result
        vals = list(params.values())
        print(f"{_label(est, lam):>14}  "
              f"{vals[0]:>12.4g} {vals[1]:>12.4g} {vals[2]:>12.4g} "
              f"{resid:>10.3f}")
        print(f"  wrote {os.path.basename(out_path)}")


if __name__ == '__main__':
    main()
