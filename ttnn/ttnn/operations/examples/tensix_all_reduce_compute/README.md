# All-reduce compute accumulation - FPU destination reuse

**Problem type:** compute-bound: reduction  ·  **Concept:** reduction accumulation in DST registers
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · Wormhole B0 · 1000 MHz · 2026-07-10 · `5f0ad060667`

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
DST batch and handles two contributors per instruction. FP32 DST was also slightly faster in this
sweep; the smaller four-tile batch did not offset its accumulation advantage.

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

Board: **Wormhole B0 n150**; numbers from `test_tensix_all_reduce_compute_regimes` (one Tensix core,
bf16 tiles, interleaved A/B, median of 11 retained trials, 2026-10-12). `kernel_iters` is the number
of reductions per launch; ns are per reduction.

| Condition | Shape(s) | Baseline ns | Lever ns | Speedup | Trials | Board |
|---|---|---:|---:|---:|---:|---|
| **Helps:** 8 blocks (reduced tiles per output tile), 6 output tiles, steady state (100 reductions per launch) | 8 blocks x 6 tiles | 11178 (11169-11180) | 1899 (1869-1912) | **5.89x** | 11 | WH B0 n150 |
| **Helps weakly:** smallest input the API accepts, ONE reduction per launch | 2 blocks x 1 tile, `kernel_iters=1` | 783 (775-803) | 568 (559-583) | **1.38x** | 11 | WH B0 n150 |
| **Does not help:** DST precision choice, FP32 DST vs BF16 DST (both already FPU `acc_to_dest`) | 16 blocks x 6 tiles | 3611 (3580-3660) | 3492 (3455-3500) | **1.03-1.04x** | 11 | WH B0 n150 |

The FPU win grows with the number of SFPU calls it removes: the baseline pays one
`add_binary_tile_init()` plus one SFPU add per contributor beyond the first, so its cost scales with
`(blocks - 1) x tiles` while the FPU variant pays one init per DST batch and handles two contributors
per instruction. Sweep (WH B0, `fpu_dest_reuse_fp32` vs `sfpu_serial_bf16`, 100 reductions per launch):
2 blocks x 1 tile 1.90x, 2 x 2 2.15x, 2 x 6 2.66x, 3 x 1 2.37x, 4 x 1 3.01x, 8 x 1 4.30x. The speedup
never reaches 1.0x on this API (`num_blocks >= 2` is enforced): even the smallest input is 1.9x in
steady state and 1.38x when the launch's fixed cost (single reduction, ~570 ns of which most is
kernel start/stop) is not amortized (`kernel_iters` 1 / 2 / 10 at 2x1: 1.38x / 1.78x / 1.98x). Do not
expect a blocks-vs-1x crossover; the only regime where the lever buys nothing is the sub-choice of
DST precision, where FP32 DST is only ~3% faster than BF16 DST at 16 blocks (up to ~8% at 2 blocks).
