# MXFP8 optimization harness

MXFP8 uses the shared quantized-decoder harness `scripts/quant_harness.py`. The
principles, isolation model, evaluation stages and promotion rules are the same as
for NVFP4; see [the NVFP4 harness documentation](../dequantize_nvfp4/HARNESS.md).
This file lists only what differs.

```bash
source source-env.sh
python scripts/quant_harness.py --format mxfp8 prepare --path /tmp/mxfp8-baseline
python scripts/quant_harness.py --format mxfp8 prepare --path /tmp/mxfp8-candidate

python scripts/quant_harness.py --format mxfp8 evaluate \
    --candidate /tmp/mxfp8-baseline --output generated/mxfp8_loop/baseline --math-isolate

# Edit only kernel files in /tmp/mxfp8-candidate, then:
python scripts/quant_harness.py --format mxfp8 evaluate \
    --candidate /tmp/mxfp8-candidate --output generated/mxfp8_loop/candidate-001 --math-isolate

python scripts/quant_harness.py --format mxfp8 compare \
    --baseline generated/mxfp8_loop/baseline/result.json \
    --candidate generated/mxfp8_loop/candidate-001/result.json

python scripts/quant_loop_stats.py generated/mxfp8_loop --output generated/mxfp8-loop-stats.json
```

## Differences from NVFP4

| | NVFP4 | MXFP8 |
|---|---|---|
| Editable scope | `dequantize_nvfp4/device/kernels` | `dequantize_mxfp8/device/kernels` |
| Worktree manifest | `.nvfp4-worktree.json` | `.mxfp8-worktree.json` (records `format`; a worktree can't be evaluated as another format) |
| Accuracy gate | Max/mean/median BF16 ULP and SNR limits | **Exact**: zero mismatches against `E4M3 * 2^(E8M0-127)`. ≥2^128 must be ∞; values <2^-126 may flush to zero |
| Corpus | 127 scales × NVFP4 code/BFP contexts | Every finite (code, scale) pair, 254 × 255, each code at a different column per row, plus random blocks |
| Accuracy hash binding | `kernel_sha256` of `nvfp4_sfpu.hpp` | `kernel_hashes` of **every** kernel file |
| Full accuracy suite | Survey + op tests + transpose suite; 6 permitted skips | `test_mxfp8_accuracy.py` + `test_dequantize_mxfp8.py`; **no** permitted skips |
| Screen | SFPU corpus × 12 tensor scales | Exhaustive corpus in the primary configuration |
| Benchmark | UINT16, BF16 DEST, tensor scale 0.8125 | Primary configuration (`PRIMARY` in `mxfp8_reference.py`: UINT16, BF16 DEST, BF16 TILE), L1-capacity-sized; `MXFP8_BENCH_PACKED_TILES_PER_CORE` overrides |
| Profiler op code / compute kernel | `Nvfp4DeviceOperation` / `dequantize_fast` | `Mxfp8DeviceOperation` / `dequantize` |
| Math diagnostic sweep | Located by NVFP4 source anchors | Code between `// MXFP8_MATH_SWEEP_BEGIN` and `// MXFP8_MATH_SWEEP_END` |
| Input bit permutation | Supported | Not supported |

Environment variables seen by the tests are `MXFP8_KERNEL_ROOT`, `MXFP8_ACCURACY_REPORT` and `MXFP8_BENCH_REPORT`. The generic `QUANT_KERNEL_ROOT` and `QUANT_PYTEST_METRICS` are consumed by `scripts/quant_harness_plugin.py`.

## Adding another format

Add a `FormatSpec` to `FORMATS` in `scripts/quant_harness.py` with these fields:
- kernel directory and manifest name
- trusted files
- accuracy, screen and perf test lists
- permitted skips
- profiler op code
- compute kernel stem
- math-sweep locator
- accuracy hash binding

The tests must:
- write `<PREFIX>_ACCURACY_REPORT` with an `accuracy_gate.passed` field and the kernel hash binding;
- write `<PREFIX>_BENCH_REPORT` workload metadata (`calls`, `warmup_calls`, `cores`, `maximum_output_tiles_per_core`).
