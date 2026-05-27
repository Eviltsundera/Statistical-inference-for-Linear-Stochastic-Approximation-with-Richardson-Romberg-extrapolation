"""Generate compact thesis figures for the numerical experiments chapter."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'figures' / 'experiments'
RESULTS = ROOT / 'code' / 'results'


COLORS = {
    'blue': '#2563eb',
    'green': '#059669',
    'orange': '#d97706',
    'red': '#dc2626',
    'gray': '#6b7280',
    'dark': '#111827',
}


def _style():
    plt.rcParams.update({
        'figure.dpi': 140,
        'savefig.dpi': 180,
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#e5e7eb',
        'grid.linewidth': 0.8,
        'grid.alpha': 1.0,
    })


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    text = out.read_text(encoding='utf-8')
    out.write_text('\n'.join(line.rstrip() for line in text.splitlines()) + '\n',
                   encoding='utf-8')
    print(out)


def main_methods_figure():
    df = pd.read_csv(ROOT / 'code' / 'results_comparison.csv')
    keep = [
        ('const_0.2', 'const 0.2'),
        ('const_0.02', 'const 0.02'),
        ('RR', 'RR center'),
        ('dim_0.2', r'$0.2/\sqrt{k}$'),
        ('PR_OBM', 'PR + OBM'),
        ('RR_OBM', 'RR + OBM'),
        ('RR_OBM_RR', 'RR + OBM-LW'),
    ]
    sub = (
        df.set_index('method')
        .loc[[key for key, _ in keep]]
        .assign(short_label=[label for _, label in keep])
    )

    x = np.arange(len(sub))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1), sharex=True)

    axes[0].bar(x, sub['l2_median'], color=COLORS['blue'], width=0.72)
    axes[0].set_ylabel(r'L2 error, $10^{-3}$')
    axes[0].set_title('Point-estimator error')
    axes[0].set_ylim(0, max(sub['l2_median']) * 1.18)
    for i, value in enumerate(sub['l2_median']):
        axes[0].text(i, value + 0.5, f'{value:.1f}', ha='center', va='bottom',
                     fontsize=7)

    axes[1].bar(x, sub['cov_median'], color=COLORS['green'], width=0.72)
    axes[1].axhline(95, color=COLORS['red'], linestyle='--', linewidth=1.1,
                    label='target 95%')
    axes[1].set_ylabel('median coverage, %')
    axes[1].set_title('Scalar CI coverage')
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc='lower right', frameon=False)
    for i, value in enumerate(sub['cov_median']):
        axes[1].text(i, value + 2.2, f'{value:.0f}', ha='center', va='bottom',
                     fontsize=7)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(sub['short_label'], rotation=35, ha='right')
        ax.grid(axis='x', visible=False)

    fig.suptitle('Main comparison at T=1e6', y=1.04, fontsize=11)
    _save(fig, 'main_methods_comparison.svg')


def blocksize_lugsail_figure():
    block = pd.read_csv(
        RESULTS / 'blocksize_coverage'
        / 'rr_blocksize_T20k_100k_1M_pair0p20_0p10_w24_summary.csv'
    )
    block = block[(block['T'] == 100000) & block['eta'].notna()]
    obm = block[block['estimator'] == 'OBM'].sort_values('eta')
    obm_rr = block[block['estimator'] == 'OBM_RR'].sort_values('eta')

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.05), sharex=True)
    axes[0].plot(obm['eta'], obm['coverage_median_pct'], marker='o',
                 color=COLORS['gray'], label='OBM coverage')
    axes[0].plot(obm_rr['eta'], obm_rr['coverage_median_pct'], marker='o',
                 color=COLORS['green'], label='OBM-LW coverage')
    axes[0].axhline(95, color=COLORS['red'], linestyle='--', linewidth=1.0)
    axes[0].set_xlabel(r'block exponent $\eta$')
    axes[0].set_ylabel('median coverage, %')
    axes[0].set_title('Coverage')
    axes[0].set_ylim(70, 100)
    axes[0].legend(frameon=False, loc='lower center')

    axes[1].plot(obm['eta'], obm['rel_bias_raw_median'], marker='s',
                 color=COLORS['gray'], label='OBM')
    axes[1].plot(obm_rr['eta'], obm_rr['rel_bias_raw_median'], marker='s',
                 color=COLORS['green'], label='OBM-LW')
    axes[1].axhline(0, color=COLORS['red'], linestyle='--', linewidth=1.0)
    axes[1].set_xlabel(r'block exponent $\eta$')
    axes[1].set_ylabel('median relative variance bias')
    axes[1].set_title('Variance-estimator bias')
    axes[1].set_ylim(-0.75, 0.12)
    axes[1].legend(frameon=False, loc='lower left')

    for ax in axes:
        ax.set_xticks(obm['eta'])

    fig.suptitle(r'Block-size tuning for OBM and OBM-LW, $T=10^5$',
                 y=1.04, fontsize=11)
    _save(fig, 'blocksize_lugsail_diagnostics.svg')


def mixing_stress_figure():
    mix = pd.read_csv(
        RESULTS / 'stress'
        / 'rr_mixing_lazy_T100k_1M_pair0p20_0p10_w24_summary.csv'
    )
    mix_eta = mix[(mix['T'] == 1000000) &
                  ((mix['estimator'] == 'ORACLE') |
                   ((mix['estimator'] == 'OBM_RR') & (mix['eta'] == 0.5)))]
    oracle = mix_eta[mix_eta['estimator'] == 'ORACLE'].sort_values(
        'spectral_gap_median', ascending=False
    )
    rr = mix_eta[mix_eta['estimator'] == 'OBM_RR'].sort_values(
        'spectral_gap_median', ascending=False
    )

    fig, ax = plt.subplots(figsize=(5.2, 3.15))
    ax.plot(rr['spectral_gap_median'],
            rr['coverage_median_pct'], marker='o',
            color=COLORS['orange'], linewidth=1.8,
            label=r'OBM-LW, $\eta=0.5$')
    ax.plot(oracle['spectral_gap_median'],
            oracle['coverage_median_pct'], marker='o', fillstyle='none',
            markersize=7, linestyle='--', linewidth=1.4,
            color=COLORS['blue'], label='oracle variance')
    ax.axhline(95, color=COLORS['red'], linestyle='--', linewidth=1.0)
    ax.invert_xaxis()
    ax.set_xlabel('median spectral gap')
    ax.set_ylabel('median coverage, %')
    ax.set_title(r'Mixing stress, $T=10^6$')
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc='lower left')
    for _, row in oracle.iterrows():
        label = row['scenario'].replace('lazy_', '')
        ax.annotate(label, (row['spectral_gap_median'],
                    row['coverage_median_pct']), textcoords='offset points',
                    xytext=(4, 6), fontsize=7)

    _save(fig, 'mixing_stress_diagnostics.svg')


def main():
    _style()
    main_methods_figure()
    blocksize_lugsail_figure()
    mixing_stress_figure()


if __name__ == '__main__':
    main()
