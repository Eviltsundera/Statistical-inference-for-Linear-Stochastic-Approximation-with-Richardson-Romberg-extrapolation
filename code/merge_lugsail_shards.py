"""Merge sharded outputs of `run_lugsail_decomposition.py` into one CSV.

Shards are runs with the same problem (`--prob-seed`, `--dir-seed`) and the
same (T, b) grid but different `--traj-seed`. Per-(T, b, estimator, lam) rows
are combined exactly:

    N     = sum_i n_i                      (n_i = n_traj_used of shard i)
    mean  = sum_i n_i * mean_i / N
    mse   = sum_i n_i * mse_i  / N
    bias  = mean - sigma_true
    var   = (mse - bias^2) * N / (N - 1)   (pooled ddof=1 sample variance)

`mean_clamped` / `mse_clamped` are combined the same way as `mean` / `mse`
(NaN rows, e.g. plain OBM, stay NaN).

Usage:
    python merge_lugsail_shards.py shard1.csv shard2.csv ... --out merged.csv
"""

import argparse

import numpy as np
import pandas as pd

KEY = ['T', 'b', 'estimator', 'lam']
COLS = ['T', 'b', 'estimator', 'lam', 'sigma_true', 'mean', 'bias', 'var',
        'mse', 'mean_clamped', 'mse_clamped', 'n_traj_used']


def _weighted_mean(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & (w > 0)
    if not ok.any():
        return float('nan')
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def merge(frames):
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for key, g in df.groupby(KEY, sort=True):
        sigma_true = g['sigma_true'].iloc[0]
        if not np.allclose(g['sigma_true'], sigma_true, rtol=1e-12):
            raise ValueError(f"sigma_true mismatch in group {key}")
        n = g['n_traj_used'].to_numpy(dtype=float)
        N = int(n.sum())
        mean = _weighted_mean(g['mean'], n)
        mse = _weighted_mean(g['mse'], n)
        bias = mean - sigma_true
        var = (mse - bias ** 2) * N / (N - 1) if N > 1 else float('nan')
        rows.append({
            'T': key[0], 'b': key[1], 'estimator': key[2], 'lam': key[3],
            'sigma_true': sigma_true,
            'mean': mean, 'bias': bias, 'var': var, 'mse': mse,
            'mean_clamped': _weighted_mean(g['mean_clamped'], n),
            'mse_clamped': _weighted_mean(g['mse_clamped'], n),
            'n_traj_used': N,
        })
    return pd.DataFrame(rows, columns=COLS)


def main():
    p = argparse.ArgumentParser(description="Merge lugsail decomposition shards")
    p.add_argument('csvs', nargs='+')
    p.add_argument('--out', required=True)
    cli = p.parse_args()

    frames = [pd.read_csv(path) for path in cli.csvs]
    merged = merge(frames)
    merged.to_csv(cli.out, index=False)
    n_shards = len(frames)
    print(f"Merged {n_shards} shards -> {len(merged)} rows -> {cli.out}")
    print(merged.groupby('T')['n_traj_used'].max().rename('max n_traj_used'))


if __name__ == '__main__':
    main()
