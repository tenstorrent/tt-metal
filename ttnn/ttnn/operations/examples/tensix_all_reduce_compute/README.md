# All-reduce compute accumulation - FPU destination reuse

**Problem type:** compute-bound: reduction  ·  **Concept:** reduction accumulation in DST registers
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · Wormhole B0 · 1000 MHz · 2026-07-10 · `5f0ad060667`
**Measured on:** WH B0 n150 (regimes, 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, then read the code if needed.

## The problem

A core has several equal tile blocks in L1 and must add corresponding tiles. Reinitializing the
SFPU binary add path for every contributor is legal but expensive. This example removes all NoC
traffic and runs on one core so the measured difference is only the reduction kernel.

## Variants

| Variant | Accumulation strategy |
|---|---|
| `sfpu_serial_bf16` | Copy the first tile to DST, then call `add_binary_tile_init()` and `add_binary_tile()` for every additional block. This preserves the former all-reduce reducer as the baseline. |
| `fpu_dest_reuse_bf16` | Pair contributors with FPU `add_tiles(..., acc_to_dest=true)` and keep up to eight output tiles live in BF16 DST. |
| `fpu_dest_reuse_fp32` | Use the same pairwise FPU accumulation with FP32 DST and a JIT-derived DST-capacity batch. This is the collective implementation. |

Odd block counts copy one contributor into DST once, then accumulate the remaining contributor
pairs. Even block counts need no seed copy.

## Measured result

Six BF16 tiles, one L1-sharded core, 100 reductions per launch, median of five trials:

| Blocks | SFPU serial | FPU reuse BF16 | FPU reuse FP32 | FP32 speedup |
|---:|---:|---:|---:|---:|
| 2 | 2066.2 ns | 826.6 ns | **766.1 ns** | **2.70x** |
| 4 | 5099.0 ns | 1228.4 ns | **1112.6 ns** | **4.58x** |
| 8 | 11171.9 ns | 2006.8 ns | **1888.1 ns** | **5.92x** |
| 16 | 23313.8 ns | 3569.0 ns | **3455.4 ns** | **6.75x** |

Repeated SFPU setup grows once per added contributor, while the FPU variant initializes once per
DST batch and handles two contributors per instruction. On this Wormhole B0 box FP32 DST was also
slightly faster; the smaller four-tile batch did not offset its accumulation advantage. That does not
hold on BH p150b, where FP32 DST is 2–8% slower than BF16 DST from 4 blocks up (see below).

## Run

```bash
python -m ttnn.operations.examples.tensix_all_reduce_compute \
  --num-blocks 2 4 8 16 --num-tiles 6 --kernel-iters 100 --trials 5
```

The CLI accepts `--variant`, `--num-blocks`, `--num-tiles`, `--kernel-iters`, `--trials`, and
`--report`. Correctness is checked against a Torch sum before profiling.

```bash
ARC_COMPUTE_KERNEL_ITERS=100 ARC_COMPUTE_TRIALS=5 \
ARC_COMPUTE_REPORT=ttnn/ttnn/operations/examples/tensix_all_reduce_compute/report.md \
scripts/run_safe_pytest.sh --run-all \
tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce_compute.py::test_tensix_all_reduce_compute_device_perf
```

## When it helps / when it does not
**Rule:** always replace the per-contributor SFPU add with FPU `add_tiles(acc_to_dest=true)`; the gain grows
with the number of contributors (blocks) and never drops below 1.0× (`num_blocks >= 2` is enforced). Same
verdicts on both boards, larger factors on BH p150b (9.37× vs 5.89× at 8 blocks). The FP32-vs-BF16 DST
choice is not a speed lever: ~flat on WH B0 n150, slightly slower on BH p150b.

One Tensix core, bf16 tiles; "blocks" = reduced tiles per output tile; ns per reduction. Baseline
`sfpu_serial_bf16` → lever `fpu_dest_reuse_fp32` unless the row says otherwise; parentheses = min–max over
the retained trials.

| Regime (blocks × output tiles, reductions per launch) | WH B0 n150 · 8×8 = 64 Tensix cores (1 used) | BH p150b · 12×10 = 120 Tensix cores (1 used) |
|---|---|---|
| **Helps:** 2 blocks × 6 tiles, 100 per launch | 2 066 → 766 ns **2.70×** | 1 290 → 330 ns **3.91×** |
| **Helps:** 4 blocks × 6 tiles, 100 per launch | 5 099 → 1 113 ns **4.58×** | 3 174 → 471 ns **6.74×** |
| **Helps:** 8 blocks × 6 tiles, 100 per launch (steady state) | 11 178 (11 169–11 180) → 1 899 (1 869–1 912) ns **5.89×** | 6 937 (6 936–6 937) → 740 (740–740) ns **9.37×** |
| **Helps:** 16 blocks × 6 tiles, 100 per launch | 23 314 → 3 455 ns **6.75×** | 14 478 → 1 264 ns **11.45×** |
| **Helps weakly:** 2 blocks × 1 tile, ONE reduction per launch (launch fixed cost not amortized) | 783 (775–803) → 568 (559–583) ns **1.38×** | 509 (482–521) → 392 (374–412) ns **1.30×** |
| **Does not help:** DST precision, `fpu_dest_reuse_bf16` → `fpu_dest_reuse_fp32`, 16 blocks × 6 tiles | 3 611 (3 580–3 660) → 3 492 (3 455–3 500) ns **1.03–1.04×** | 1 165 (1 165–1 166) → 1 241 (1 241–1 241) ns **0.94× — slower on BH** |
| **Boundary** | none: smallest input still 1.38× (1-launch) / 1.90× (steady) | none: smallest input still 1.30× (1-launch); 2 × 1 steady state not measured, 2 × 6 is 3.91× |

**Why the boards differ:**
- The FPU path speeds up more than the SFPU path on BH p150b: at 8 blocks `fpu_dest_reuse_fp32` is 2.6× faster
  than on WH B0 n150 (740 vs 1 888 ns) while `sfpu_serial_bf16` is only 1.6× faster (6 937 vs 11 172 ns);
  AICLK (1350 vs 1000 MHz) explains 1.35× of each. The rest is not isolated by this example.
- FP32 DST holds 4 tiles per half-sync batch vs 8 for BF16 DST on both boards (same DST capacity). On BH
  p150b FP32 DST is slower from 4 blocks up (FP32-DST ns ÷ BF16-DST ns: 0.98 at 2 blocks, 1.02 at 4, 1.06
  at 8, 1.08 at 16); on WH B0 it was 3–8% faster. The smaller batch is the likely cost; this example does
  not isolate it.

The FPU win grows with the number of SFPU calls it removes: the baseline pays one
`add_binary_tile_init()` plus one SFPU add per contributor beyond the first, so its cost scales with
`(blocks - 1) x tiles` while the FPU variant pays one init per DST batch and handles two contributors
per instruction. Further WH B0 sweep points (`fpu_dest_reuse_fp32` vs `sfpu_serial_bf16`, 100 reductions
per launch): 2 blocks x 1 tile 1.90x, 2 x 2 2.15x, 2 x 6 2.66x, 3 x 1 2.37x, 4 x 1 3.01x, 8 x 1 4.30x;
not measured on BH. The speedup never reaches 1.0x on this API: on WH B0 even the smallest input is 1.9x
in steady state and 1.38x when the launch's fixed cost (single reduction, ~570 ns of which most is kernel
start/stop) is not amortized (`kernel_iters` 1 / 2 / 10 at 2x1: 1.38x / 1.78x / 1.98x). Do not expect a
blocks-vs-1x crossover; the only regime where the lever buys nothing is the sub-choice of DST precision,
where FP32 DST is only ~3% faster than BF16 DST at 16 blocks on WH B0 (up to ~8% at 2 blocks) and 6%
slower on BH p150b.

*How measured:*
- WH B0 n150: `test_tensix_all_reduce_compute_regimes` (interleaved A/B, median of 11 retained trials,
  three runs; ranges in [`report.md`](report.md)), 2026-10-12. The 2/4/16-block rows are the 2026-07-10
  `test_tensix_all_reduce_compute_device_perf` sweep (`bgd-lab-t3003`, N=5, kernel-iters 100).
- BH p150b: the same regimes test (8 × 6, 2 × 1, DST precision) plus
  `test_tensix_all_reduce_compute_device_perf` (N=5, kernel-iters 100) for the 2/4/16-block rows;
  `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19.

Reproduce (same regimes on any board; asserts only what held on every measured board, `measure` regimes only log):
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce_compute.py::test_tensix_all_reduce_compute_regimes
```
