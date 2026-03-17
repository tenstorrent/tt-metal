# layer_norm_rm — Pipeline Report

## Summary

**Operation**: `layer_norm_rm` — Row-major layer normalization
**Result**: SUCCESS — All 5 TDD stages passed on first attempt
**Total test cases**: 25 (5 shapes × 5 stages)
**Retries used**: 0

### What it does
Normalizes each row (last dimension W) of a row-major interleaved bfloat16 tensor:
```
mean = (1/W) * sum(x)
var  = (1/W) * sum((x - mean)²)
output = gamma * (x - mean) / sqrt(var + eps) + beta
```
Gamma and beta are optional affine parameters. All I/O in ROW_MAJOR_LAYOUT — tilize/untilize happens in-kernel.

---

## Pipeline Execution

| Phase | Agent | Duration | Output |
|-------|-------|----------|--------|
| 0: Discovery | orchestrator | ~30s | 3 references identified |
| 1: Analysis | 3× ttnn-operation-analyzer (parallel) | ~11min | tilize, untilize, batch_norm analyses |
| 2: Design | ttnn-operation-architect | ~8min | op_design.md, .tdd_state.json (5 stages) |
| 3: Build | ttnn-generic-op-builder | ~9min | Python infra, stub kernels, test files |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~22min | All 5 stages passed |
| 5: Report | orchestrator | — | This file |
| 6: Self-Reflection | ttnn-self-reflection | — | self_reflection.md |

---

## Agent Summaries

### Phase 1: Analyzers (3 parallel)

**Tilize (input_stage)**:
- Reader uses `TensorAccessor.get_noc_addr(page_id)` for interleaved DRAM page addressing
- Reads 32 sticks per batch (one TILE_HEIGHT), pushes Wt tiles per `cb_push_back`
- Work distribution via `split_blocks_for_tilize` — 1D linearized grid, contiguous page ranges

**Untilize (output_stage)**:
- Writer uses `TensorAccessor` for output DRAM writes
- Reads untilized sticks from output CB, writes 32 sticks per tile-row
- Single-buffered output CB with Wt pages

**Batch Norm (compute_core)**:
- Two variants (FPU vs SFPU), selected by `fp32_dest_acc_en`
- `binary_dest_reuse_tiles` optimization for chained operations
- Multi-pass CB persistence pattern — stats loaded once, consumed across tiles
- CB aliasing for conditional affine paths
- Key difference: batch norm receives pre-computed mean/var; layer norm must compute them via reduction

### Phase 2: Architect

Designed hybrid architecture combining tilize input stage, untilize output stage, and adapted batch_norm compute:

**CB Layout** (15 circular buffers):
- c_0: input RM sticks, c_1: tilized input (2×Wt for reuse)
- c_2/c_3: gamma/beta RM sticks, c_8: reduce scaler, c_9: epsilon
- c_16: output RM sticks
- c_24-c_31: intermediates (mean, centered, squared, variance, rstd, normalized, tilized gamma/beta)

**5 TDD Stages**: data_pipeline → reduce_mean → subtract_mean → variance_rsqrt → affine

### Phase 3: Builder

Created:
- `layer_norm_rm.py`: Entry point with validation (dtype, layout, shape alignment, gamma/beta width match)
- `layer_norm_rm_program_descriptor.py`: ProgramDescriptor with 15 CBs, 3 kernels, work distribution via `split_work_to_cores`
- Stub kernels (3 `.cpp` files)
- Integration test + 5 TDD stage tests

### Phase 4: TDD Kernel Writer

Implemented all 5 stages incrementally:

| Stage | Description | Attempts | Result |
|-------|-------------|----------|--------|
| 1. data_pipeline | Reader reads RM→tilize→untilize→writer writes RM | 1 | PASSED |
| 2. reduce_mean | Row-wise reduce for mean, broadcast back to full shape | 1 | PASSED |
| 3. subtract_mean | Sub(input, mean) with COL broadcast | 1 | PASSED |
| 4. variance_rsqrt | Square, reduce variance, add eps, rsqrt, multiply | 1 | PASSED |
| 5. affine | Optional gamma mul (ROW) + beta add (ROW) | 1 | PASSED |

**Key implementation details**:
- Reader: `TensorAccessor` for DRAM reads, `calculate_and_prepare_reduce_scaler` for 1/W, manual epsilon tile generation, zero-padded gamma/beta sticks for tilize
- Compute: `tilize` → `reduce(REDUCE_ROW)` → `sub(COL)` → `square` → `reduce(REDUCE_ROW)` → `add(SCALAR)+rsqrt` → `mul(COL)` → optional `mul(ROW)` → optional `add(ROW)` → `untilize`
- Writer: `TensorAccessor` for DRAM writes, 32 sticks per tile-row
- CB persistence: `WaitUpfrontNoPop` for multi-use buffers, manual `cb_pop_front` after final use

---

## Git History

```
250328133a2 [ttnn-kernel-writer-tdd] stage affine: passed - ALL STAGES COMPLETE
e0eced8a753 [ttnn-kernel-writer-tdd] stage variance_rsqrt: passed
26e752ebc9b [ttnn-kernel-writer-tdd] stage subtract_mean: passed
b3a45fed68f [ttnn-kernel-writer-tdd] stage reduce_mean: passed
ffa5516e587 [ttnn-kernel-writer-tdd] stage data_pipeline: passed
d9791e86eb4 [ttnn-generic-op-builder] breadcrumbs: layer_norm_rm completion event
6759ba86023 [ttnn-generic-op-builder] stubs: layer_norm_rm
f4e5f5135cd [ttnn-operation-architect] finalize: breadcrumb completion event
3fd58e7f614 [ttnn-operation-architect] design: layer_norm_rm
eab6422e4e8 [ttnn-operation-analyzer] analysis: batch_norm (compute_core reference)
```

---

## Files Produced

### Operation code
```
ttnn/ttnn/operations/layer_norm_rm/
├── __init__.py
├── layer_norm_rm.py                        # Entry point + validation
├── layer_norm_rm_program_descriptor.py     # ProgramDescriptor
├── kernels/
│   ├── layer_norm_rm_reader.cpp            # Read RM sticks, tilize, scalers
│   ├── layer_norm_rm_compute.cpp           # Tilize/reduce/normalize/untilize
│   └── layer_norm_rm_writer.cpp            # Write RM sticks to DRAM
├── op_design.md                            # Operation design document
├── .tdd_state.json                         # TDD pipeline state
├── REPORT.md                               # This file
└── agent_logs/                             # Breadcrumbs and analysis files
```

### Tests
```
tests/ttnn/unit_tests/operations/layer_norm_rm/
├── test_layer_norm_rm.py                   # Integration test
├── test_stage_data_pipeline.py
├── test_stage_reduce_mean.py
├── test_stage_subtract_mean.py
├── test_stage_variance_rsqrt.py
└── test_stage_affine.py
```

---

## Decisions and Deviations

1. **Compute kernel syntax**: The design specified `void kernel_main() {}` but the kernel writer used `namespace NAMESPACE { void MAIN {} }` for the compute kernel. This is the standard pattern for compute kernels (only reader/writer use `void kernel_main()`).

2. **CB untilize source selection**: Compile-time constexpr logic selects which CB to untilize from based on gamma/beta presence, avoiding runtime branching.

3. **Gamma/beta zero-padding**: Reader manually zero-pads 31 sticks after the single gamma/beta stick to fill a complete tile-row for tilize. This is simpler than using padding helpers.

4. **Program descriptor `total_rows` calculation**: Fixed off-by-one in initial calculation — total_rows = product of all dims except the last (W).

5. **TensorAccessor args chaining**: All three tensor accessors (input, gamma, beta) are chained via `next_compile_time_args_offset()` regardless of presence — placeholder args used when absent.

---

## Infrastructure Issues

- **None encountered**: No device hangs, build failures, or environment issues during the entire pipeline run.

---

## Suggestions for Pipeline Improvement

1. **Analyzer role-scoping worked well**: The focused directives cut analysis size and kept architect context clean.
2. **Zero retries**: Clean TDD progression suggests the design-to-implementation handoff was accurate.
3. **Consider adding a validation stage**: A pre-TDD stage that just runs `import` + `pytest --collect-only` to catch Python syntax errors before device tests.
