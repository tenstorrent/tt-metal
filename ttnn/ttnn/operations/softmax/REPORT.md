# Softmax Operation — Pipeline Report

## Summary

**Operation**: softmax
**Math**: `exp(x_i - max(x)) / sum(exp(x_j - max(x)))` along dim=-1 (width) or dim=-2 (height)
**Overall Result**: ALL 4 TDD STAGES PASSED
**Total Pipeline Duration**: ~65 minutes (analysis: ~13min, design: ~5min, build: ~14min, TDD: ~36min)

## Pipeline Execution

| Phase | Agent | Duration | Output | Status |
|-------|-------|----------|--------|--------|
| 0: Discovery | orchestrator | ~1min | 3 references selected | DONE |
| 1: Analysis | 3x ttnn-operation-analyzer | ~13min (parallel) | 3 analysis.md files | DONE |
| 2: Design | ttnn-operation-architect | ~5min | op_design.md + .tdd_state.json | DONE |
| 3: Build | ttnn-generic-op-builder | ~14min | Python infra + stubs + tests (15/15 passed) | DONE |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~36min | 4/4 stages passed | DONE |
| 5: Report | orchestrator | — | REPORT.md | DONE |

## Agent Summaries

### Phase 1: Analyzers (parallel)

**tt-train softmax analyzer**:
- Discovered 5-phase pipeline: find_max → reduce_max → sum_exp → reduce_sum+recip → final_output
- 11 circular buffers with Float32 intermediates for precision
- Dual-path L1 strategy (cache vs stream based on L1 capacity)
- matmul_tiles replaces reduce_tile<SUM> for precision
- NaN-safe masking with -inf for padding

**reduce_w analyzer**:
- Two compute paths: matmul-based (SUM/AVG) and reduce_tile (MAX)
- `compute_kernel_lib::reduce<>()` helper wraps boilerplate
- `WaitUpfrontNoPop` input policy for tile reuse
- Scaler tile construction: `prepare_reduce_scaler` vs `generate_mm_scaler`

**reduce_h analyzer**:
- Column-major tile ordering for REDUCE_COL
- Stride-by-Wt DRAM access pattern
- ROW broadcast for column reduction results

### Phase 2: Architect

- Designed 7 CBs: c_0 (input), c_1 (scaler), c_2 (mm_scaler), c_3 (max), c_4 (exp_sum), c_5 (recip_sum), c_16 (output)
- 3-pass reader pattern per row/column
- Single-core work distribution
- fp32_dest_acc_en=true for precision
- Compile-time defines: DIM_W, DIM_H, NUMERIC_STABLE, REDUCE_OP, REDUCE_DIM
- 4 TDD stages: data_pipeline → exp_only → softmax_dim_w → softmax_dim_h

### Phase 3: Builder

- Created entry point with full input validation (dtype, layout, rank, dim)
- Program descriptor with 7 CBs, 3 kernel paths, preprocessor defines
- Stub kernels (empty `void kernel_main() {}`)
- Integration test: 15/15 tests passed (shape verification + validation tests)
- 5 TDD stage test files generated

### Phase 4: TDD Kernel Writer

Implemented all 3 kernels through 4 incremental stages.

## TDD Pipeline Results

| Stage | Name | Result | Hard Attempts | Free Retries | Failure Classifications |
|-------|------|--------|---------------|--------------|------------------------|
| 1 | data_pipeline | PASS | 0 | 0 | — |
| 2 | exp_only | PASS | 0 | 1 | compilation_error (wrong include path) |
| 3 | softmax_dim_w | PASS | 3 | 0 | numerical_mismatch, hang_cb_deadlock, unknown |
| 4 | softmax_dim_h | PASS | 0 | 0 | — |

### Stage Details

**Stage 1 (data_pipeline)**: Passed first try. Reader reads tiles, compute copies c_0→c_16, writer writes output. Required upstream fix: reader needed additional compile-time args (num_rows_or_cols, num_tiles) and DIM_W/DIM_H defines.

**Stage 2 (exp_only)**: 1 free retry for wrong include path (`api/compute/eltwise_unary/sfpu/exp.h` → `api/compute/eltwise_unary/exp.h`). Passed after fix.

**Stage 3 (softmax_dim_w)**: Most challenging stage. 3 hard attempts:
1. Numerical mismatch (max diff 0.51) — incorrect reduction accumulation
2. Device hang (cb_wait_front deadlock) — matmul acquire_dst/release_dst conflict with reduce helper
3. Test failure after restructuring — CB sizing issue (cb_exp_sum needed inner_dim tiles, not 2)

Key design deviation: Replaced matmul-based sum with `reduce<SUM>` helper due to DST register deadlock between matmul accumulator and reduce helper cycles. The reduce<SUM> approach achieves sufficient precision within tolerances (rtol=0.05, atol=0.2).

**Stage 4 (softmax_dim_h)**: Passed first try. The dim=-2 path reused the same compute structure with REDUCE_COL and ROW broadcasts, plus column-major tile ordering in the reader.

## Files Produced

### Operation Files (`ttnn/ttnn/operations/softmax/`)
```
├── __init__.py                         # Re-export softmax()
├── softmax.py                          # Entry point with validation
├── softmax_program_descriptor.py       # CB config, kernel args, defines
├── kernels/
│   ├── reader_softmax.cpp              # 3-pass reader with dim branching
│   ├── compute_softmax.cpp             # 3-phase softmax (max, sub+exp+sum, mul)
│   └── writer_softmax.cpp              # Sequential tile writer
├── op_design.md                        # Operation design document
├── .tdd_state.json                     # TDD pipeline state (all passed)
├── REPORT.md                           # This file
└── agent_logs/                         # All breadcrumbs and analyses
```

### Test Files (`tests/ttnn/unit_tests/operations/softmax/`)
```
├── test_softmax.py                     # Integration test (15 tests)
├── test_stage_data_pipeline.py         # TDD stage 1
├── test_stage_exp_only.py              # TDD stage 2
├── test_stage_softmax_dim_w.py         # TDD stage 3
└── test_stage_softmax_dim_h.py         # TDD stage 4
```

## Git History

```
329f5a79d4 [ttnn-kernel-writer-tdd] stage softmax_dim_h: passed (all stages complete)
7a4d812f80 [ttnn-kernel-writer-tdd] stage softmax_dim_w: passed
23cb7ce397 [ttnn-kernel-writer-tdd] stage exp_only: passed
adeea77856 [ttnn-kernel-writer-tdd] stage data_pipeline: passed
6d8b001615 [ttnn-generic-op-builder] logs: softmax execution log and breadcrumbs
51d95a2417 [ttnn-generic-op-builder] stubs: softmax
ee048d220d [ttnn-operation-architect] finalize: breadcrumb completion events
0c2ded4212 [ttnn-operation-architect] design: softmax
9a296f433a [ttnn-operation-analyzer] analysis: softmax (tt-train reference)
46c138a105 [ttnn-operation-analyzer] analysis: reduce_h
```

## Decisions and Deviations

### Decisions Made
1. **3 references** instead of 1: Used tt-train softmax + reduce_w + reduce_h to cover both dimension patterns
2. **Single-core**: Initial implementation is single-core for simplicity
3. **3-pass streaming**: Each row/column reads from DRAM 3 times (no L1 caching optimization)
4. **bf16 intermediate CBs** with fp32 DST: Rather than fp32 CBs (as tt-train uses), kept CBs as bf16 but enabled fp32_dest_acc_en for DST precision

### Deviations from Design
1. **reduce<SUM> instead of matmul_tiles**: The design specified matmul-based sum for precision, but acquire_dst/release_dst caused deadlocks with the reduce helper. The reduce<SUM> helper achieves acceptable precision.
2. **cb_mm_scaler unused**: Since matmul path was replaced, CB c_2 (mm_scaler) is generated by reader but never consumed by compute (harmless overhead)
3. **cb_exp_sum sizing**: Increased from 2 tiles to inner_dim tiles to buffer all exp tiles before reduction

## Infrastructure Issues

- **No device hangs during final tests**: All 4 passing stage tests ran cleanly
- **1 device hang during development** (stage 3, attempt 2): DST register deadlock from matmul + reduce helper conflict. Resolved by switching to reduce<SUM>.
- **No build failures**: Metal build not needed (kernels compile at runtime)
- **No venv issues**: Python environment worked correctly throughout

## Suggestions for Improving the Agent Pipeline

1. **Include path database**: The exp.h include path error (sfpu/ subdir doesn't exist) is a common mistake. A curated list of correct kernel API include paths would prevent free retries.
2. **CB sizing validation in architect**: The architect designed cb_exp_sum with 2 tiles, but the actual implementation needed inner_dim tiles for buffering. The architect should validate CB sizes against the data flow (how many tiles must coexist).
3. **matmul + reduce helper interaction warning**: The deadlock from mixing acquire_dst/release_dst (matmul) with tile_regs_acquire/release (reduce helper) should be documented as a known incompatibility.
4. **Scaler tile generation API**: The reader kernel needs to generate scaler tiles — this is common boilerplate that could be a standard reader utility.
