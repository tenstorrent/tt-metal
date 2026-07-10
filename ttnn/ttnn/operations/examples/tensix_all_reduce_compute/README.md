# All-reduce compute accumulation - FPU destination reuse

**Difficulty:** ⭐⭐ T2  ·  **Concept:** reduction accumulation in DST registers
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
