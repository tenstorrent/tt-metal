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
  dominant).

  **Same-shape fused TIME gap (the real metric): tp8/N16384/dim6144 fused LN 683 -> 526 us
  vs fused RMS 407 us. LN was 68% slower than RMS, now 29% slower -- the fused LN/RMS time
  gap MORE THAN HALVED.**

## Re-profile after Opt 1 (FLUX TP8 N2368)

| zone | before | after | note |
|--|--:|--:|--|
| LN_MERGE | 62.8% | **50.7%** | 49679 -> 30199 cyc; still #1 |
| LN_PRE_WELFORD | 22.8% | 30.2% | unchanged absolute (18014 cyc) |
| POST (4 passes) | 14% | ~19% | unchanged absolute |
| TOTAL | 421M | 318M cyc | -25% |

## Why the residual gap to RMS is Welford-inherent (analysis)

- **Merge (50.7%) is at its stable floor.** The equal-count combine still needs the
  between-shard correction Σ(mean_i-mean_g)^2 = one deviation-square per shard (K squares,
  full-tile fp32 SFPU). Tried batching the loop to cut SFPU init churn -> **no measurable
  change** (the tile OPS, not inits, are the cost), so reverted. Dropping the correction
  term or using sum/sumsq (E[x^2]-E[x]^2) would be faster but LOWERS fidelity/stability ->
  disallowed. RMS avoids this entirely: 1 stat (sum of squares), combined by pairwise adds.
- **PRE (30.2%) is LLK-bound.** welford_update reduces over tile ROWS, so each input tile
  needs a transpose first; there is no column-reduce Welford variant. RMS uses mul_tiles +
  matmul-reduce (no transpose, no per-tile iterative update).
- **POST (19%) is 4 CB round-trips** (x-mean -> *1/std -> *w -> +b). The FPU broadcast ops
  read only from CBs (not DST), so the passes can't be fused into one tile_regs cycle.

Net: preserving Welford numerical stability (required) means LN keeps 2-stat Welford
overhead that RMS's 1-stat sum-of-squares does not have. Opt 1 removed the biggest
non-inherent cost (the O(ring_size) sequential pairwise merge).

## Further headroom (higher risk, not done)
- **32-row stat batching:** each stat tile uses only row 0 (32 of 1024 elements); packing
  ~num_tile_rows tile-rows' stats into full tiles would amortize the merge + stat handling
  up to 32x. Big rework of the PRE finalize + gather cadence + shared forwarder/writer.
- POST fusion would need an SFPU op that multiplies a DST tile by a col/row-broadcast tile.
