# FLUX LayerNorm speedup campaign

**Goal:** make fused Distributed LayerNorm speedup-over-composite on FLUX shapes approach the
fused RMSNorm speedup on the same shapes. All optimizations correctness-preserving (no dtype/
fidelity lowering). Baseline commit: `13ffe34e586` (bench committed).

## Baseline numbers (WH galaxy, links=4, dim=6144)

FLUX **RMSNorm** fused-vs-composite (target): TP4 1.4-1.7x, TP8 2.0-2.6x.
FLUX **LayerNorm** fused-vs-composite (start): below.

| TP | seq | base µs | fused µs | LN speedup | (RMS speedup same shape) |
|--:|--:|--:|--:|--:|--:|
| 4 | 64   | 105 | 172 | 0.61x | (~1.13-1.6x) |
| 4 | 512  | 114 | 112 | 1.02x | 1.69x |
| 4 | 2048 | 182 | 180 | 1.01x | 1.41x |
| 4 | 8192 | 570 | 440 | 1.31x | 1.63x |
| 8 | 128  | 109 | 89  | 1.23x | 1.98x |
| 8 | 1024 | 141 | 110 | 1.28x | 2.02x |
| 8 | 4096 | 306 | 210 | 1.46x | 2.07x |
| 8 | 16384| 986 | 683 | 1.44x | 2.34x |

**Key diagnosis (same-shape absolute, tp8/N16384/dim6144):** composite baselines are similar
(RMS 951µs vs LN 986µs), but **fused LN 683µs is ~68% slower than fused RMS 407µs**. So the gap
is in our fused LN compute, not the baseline. Optimize the LN op.

## Structural gap (LN vs RMS compute kernel)

FLUX LN shapes are RESIDENT path (TP4=48 tiles, TP8=24 tiles; both < 56 block-major threshold).
- **LN PRE** = per-tile Welford update loop (mean + M2), heavier than RMS's `mul_tiles(x,x)` +
  matmul-reduce sum-of-squares.
- **LN POST** = 4 full-row passes, each a pack/unpack L1 round-trip:
  (x-mean) -> *(1/std) -> *weight -> +bias. RMS POST = 2 passes (x*1/rms -> *weight).
- Prior profiling (post recip-LUT, non-FLUX shape): PRE ~40%, POST ~60%.

## PROFILE (device zones, FLUX TP8 N2368 dim6144, eager, cyc)

| zone | %compute | avg cyc/instance | what |
|--|--:|--:|--|
| **LN_MERGE** | **62.8%** | 49679 | cross-shard `combine_welford_partials` (sequential per-shard) |
| LN_PRE_WELFORD | 22.8% | 18018 | per-tile Welford (mean+M2) over the shard |
| LN_MUL_INVSTD | 3.8% | 3005 | POST: (x-mean)*1/std |
| LN_WEIGHT | 3.7% | 2899 | POST: *weight |
| LN_SUB_MEAN | 3.5% | 2795 | POST: x-mean |
| LN_BIAS | 3.4% | 2709 | POST: +bias |

**THE bottleneck is LN_MERGE (62.8%), not POST (14% total).** `combine_welford_partials`
(kernel_util/compute/combine_welford.h) does a sequential pairwise Welford merge: per shard
(ring_size=8), ~11 init-heavy SFPU/binary tile ops => ~90 dependent ops/row inside one
tile_regs block. RMS avoids this entirely (1-stat partial sums combined by a matmul-reduce).
The composite dit_layernorm baseline uses the SAME shared combine -> optimize OUR kernel's
merge only (don't touch the shared fn: keeps baseline honest + 4 other consumers safe).

## Plan
1. [done] Profile -> merge dominates.
2. [in progress] Replace the per-shard sequential merge with an equal-count combine. All n_i
   are equal (= reduce_width), so the exact Welford combine is:
     mean_g = mean(mean_i);  var_g = mean(var_i) + (mean(mean_i^2) - mean_g^2)
   -- numerically IDENTICAL to the pairwise merge (no fidelity change), far fewer ops. Prefer a
   matmul-reduce over the gathered tiles if the gather layout allows.
3. If more is needed: cheaper PRE, worker/block sweeps.
4. After each change: test_layernorm_corr (PCC>=99.9%, det) + the FLUX LN bench; log deltas.

## Results log
- Profiling checkpoint: merge = 62.8% (commit b1e78b39b20).
- **Opt 1 — equal-count merge** (numerically stable, EXACT Welford for equal counts).
  Replaced the sequential pairwise combine_welford_partials with FPU pairwise sums +
  the between-shard term computed in Welford's STABLE deviation form
  Σ(mean_i - mean_g)^2 (NOT Σmean_i^2 - mean_g^2 -- squares only small deviations, no
  cancellation). Per-shard PRE Welford unchanged. PCC unchanged (100.0003-100.0010%,
  det=OK, 5/5 LN corr). FLUX LN speedup (delta form):

  | shape | before | after |
  |--|--:|--:|
  | tp8 N16384 | 1.44x | **1.87x** |
  | tp8 N4096  | 1.46x | **1.79x** |
  | tp8 N1024  | 1.28x | **1.55x** |
  | tp8 N128   | 1.23x | **1.58x** |
  | tp4 N8192  | 1.31x | **1.42x** |
  | tp4 N2048  | 1.01x | 1.05x |
  | tp4 N512   | 1.02x | 1.11x |
  | tp4 N64    | 0.61x | 0.68x |

  TP8 now 1.55-1.87x (RMS target 2.0-2.6x). TP4 gains smaller (ring_size=4 -> merge less
  dominant). Next: re-profile; likely PRE Welford (was 22.8%) is now the top zone, and
  TP4/small-N still lag. Candidates: cheaper PRE, POST-pass fusion, worker sweep.
