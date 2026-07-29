# High-T check of the OBM / OBM-LW bias branch (T up to 2×10⁷)

**Date:** 2026-07-29
**Machine:** `train-4` (first run on this host; 20 parallel single-core workers, ~21% of the 96-core cgroup quota)
**Runner:** `code/run_lugsail_decomposition.py` (sharded), merge: `code/merge_lugsail_shards.py`
**Analyzers:** `code/analyze_lugsail_mse_asymptotics.py`, `code/analyze_lugsail_component_rates.py`, `code/analyze_lugsail_optimal_b.py`

## Purpose

Follow-up to `reports/2026-06-10_lugsail_component_rates.md` ("Next check") and the
unresolved gaps of `conversations/2026-06-10_obm-mse-hypothesis-refinement.md`:
extend the fixed-η rate experiment beyond T = 10⁶ and see whether the bias
branch finally bends toward the classical rates η (OBM) / 2η (OBM-LW), or
whether the saturation is driven by the growing correlation length of the
diminishing-step iterates, ℓ(t) ≈ 1/(c·α_t) = (t+k₀)^γ/(c·c₀), γ = 0.65 —
in which case it should get *worse* with T, not better.

## Run

Same problem and direction as the June runs (`prob-seed 0`, `dir-seed 1`,
σ²_∞ = 5.530043), PR schedule c₀=200, k₀=20000, γ=0.65, λ ∈ {2,3,4},
70 block points in T^[0.15, 0.75].

New T values: **2×10⁶, 3×10⁶, 5×10⁶, 10⁷, 2×10⁷**, each with **800
trajectories** = 4 shards × 200 (traj-seeds 1000–1019), run as 20 independent
single-core processes (`OMP_NUM_THREADS=1`, `nice -10`). Wall clock: ~41 min
(T=2×10⁶ shards) to ~6.6 h (T=2×10⁷ shards). No divergent trajectories:
`n_traj_used = 800` for every (T, b).

Shards were merged exactly (weighted means for `mean`/`mse`, variance
recovered as `(mse − bias²)·N/(N−1)`; self-test reproduces a single run to
float precision):

```bash
python merge_lugsail_shards.py results/highT_2026-07-29/lugsail_highT_T*_seed*.csv \
  --out results/highT_2026-07-29/lugsail_highT_2026-07-29_merged.csv
```

Analysis was run in two windows, pooling with the June CSV:

- **pooled**: `lugsail_mse_asymptotics_2026-06-09.csv` + merged (14 T values, 10⁴…2×10⁷);
- **late**: merged only (5 T values, 2×10⁶…2×10⁷).

## Outputs

- Raw shards + merged CSV + logs: `code/results/highT_2026-07-29/`
- Figures/tables: `reports/figures/lugsail_mse_asymptotics_2026-07-29_{highT,lateT}/`,
  `reports/figures/lugsail_component_rates_2026-07-29_{highT,lateT}/`,
  `reports/figures/lugsail_optimal_b_2026-07-29_highT/`

## Results

### 1. The bias branch did not bend — the saturation front moves right

Fixed-η MSE rates (MSE ≈ C·T^(−r(η)), star = best η) by fitting window:

| window | OBM | LW(λ=2) | LW(λ=3) | LW(λ=4) |
|---|---|---|---|---|
| T ≤ 10⁶ (June) | η=0.600, r=0.395 | η=0.450, r=0.549 | η=0.425, r=0.567 | η=0.425, r=0.568 |
| pooled 10⁴…2×10⁷ | η=0.625, r=0.336 | η=0.500, r=0.503 | η=0.475, r=0.519 | η=0.475, r=0.527 |
| late 2×10⁶…2×10⁷ | **η=0.675, r=0.324** | **η=0.500, r=0.481** | η=0.500, r=0.474 | η=0.475, r=0.477 |

The negative valley at small η **deepens** instead of closing: late-window
minima are r = −0.29 (OBM, η≈0.45–0.475) and r ≈ −0.74…−0.77 (LW, η≈0.40–0.425),
versus −0.06 / −0.23…−0.30 in the June window. Fixed-η MSE at η = 0.40–0.45
now *grows* with T for LW — in June those η were near-optimal.

### 2. Component check: variance still exact, bias frontier at η ≈ γ

Late-window component rates: the variance branch still tracks 1−η closely
(e.g. OBM at η=0.675: 0.333 vs 0.325; LW2 at η=0.5: 0.492 vs 0.5). The |bias|
rate for OBM crosses zero almost exactly at **η ≈ 0.65 = γ**:

| η | 0.55 | 0.60 | 0.65 | 0.675 |
|---|---|---|---|---|
| OBM |bias| rate (late) | −0.102 | −0.051 | **+0.001** | +0.035 |

For LW the frontier sits between 0.50 and 0.55 (λ=2: −0.634 at η=0.50,
+0.053 at 0.55) — lower than OBM's because the λb window reaches 2–4× deeper,
but clearly drifting right as well (in June LW was clean at η=0.45).

### 3. b\* keeps drifting — no fixed power law

Effective exponent of the empirical optimum (eta_eff = log b\*/log T):

| T | 10⁴ | 10⁶ | 2×10⁶ | 10⁷ | 2×10⁷ |
|---|---|---|---|---|---|
| OBM | 0.554 | 0.571 | 0.585 | 0.602 | **0.611** |
| LW(λ=2) | 0.383 | 0.437 | 0.454 | 0.472 | **0.489** |

The `b*/T^x` normalization scan confirms no x in [0.15, 0.70] flattens the
curves over the full range. The 14-point global fits (b\* ~ T^0.694 for OBM,
T^0.63 for LW; MSE\* ~ T^−0.29 / T^−0.36) are averages over a drifting local
slope, not genuine power laws.

### 4. Lugsail advantage grows in absolute terms

At T = 2×10⁷: MSE\*(OBM) = 0.0943 vs MSE\*(LW λ=2) = **0.0322** — a 2.9× win
(was 2.5× at 10⁶), with λ=2 best among {2,3,4} at every T, and at ~8× smaller
blocks (b\* = 3725 vs 28838).

## Interpretation

The experiment cleanly discriminates the two scenarios posed in June:

1. **Not classical pre-asymptotics.** With 3.3 decades of T, none of the
   fixed-η rates below the transition moved toward η / 2η; they moved away
   (deeper negative). The left branch of min(2η, 1−η) / min(4η, 1−η) is not
   late — under this step schedule it does not exist at fixed η < γ.
2. **Consistent with growing correlation length.** The PR schedule
   α_t = c₀/(t+k₀)^γ makes the iterate correlation length grow as
   ℓ(t) ≈ (t+k₀)^γ/(c·c₀). The truncation-bias expansion needs b ≫ ℓ(T),
   i.e. asymptotically η > γ. Three independent signatures match:
   the OBM bias-rate zero crossing lands at η ≈ 0.65 = γ in the late window;
   the safe-η frontier moves right in T for every method; the empirical b\*
   exponent climbs toward γ (0.611 at 2×10⁷ and rising).

Consequences:

- **Rate ceiling.** If bias-safety requires η > γ, the best achievable MSE
  rate under this schedule is capped near 1−γ = 0.35 — exactly where the late
  OBM optimum sits (r = 0.324 at η = 0.675). LW still shows r ≈ 0.47–0.48
  because its frontier lags OBM's, but it should converge to the same ceiling.
  Under diminishing-step PR, lugsail buys a large **constant-factor** MSE
  improvement (~3× here) and smaller blocks, not a better asymptotic rate.
- **The field-standard b ~ T^0.6** (Samsonov et al. 2025, Table 2) reads as an
  ℓ-tracking choice for γ = 0.65, not a bias-variance optimum in the classical
  sense; it is close to optimal for OBM on practical T but will keep creeping.
- Safe picks on this problem at T ~ 10⁶–2×10⁷: OBM η ≈ 0.65–0.68,
  LW(λ=2) η ≈ 0.55–0.60 (η = 0.45–0.50 is already bias-contaminated in trend,
  even though its MSE level is still fine at 2×10⁷).

## Next

1. **Constant-step control** (`run_lsa_const` exists): with α fixed, ℓ is
   constant, so the classical branches η / 2η should reappear at fixed η.
   This would pin the mechanism beyond doubt.
2. **γ-sweep** (e.g. γ ∈ {0.55, 0.65, 0.75} with matched c₀): the bias
   frontier should track γ.
3. Theory: formalize the saturated-bias regime b ≲ ℓ(T) for LSA-PR and the
   resulting effective rate cap 1−γ; candidate thesis statement replacing the
   June cautious wording.
