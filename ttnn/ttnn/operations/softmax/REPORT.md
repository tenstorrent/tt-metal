# Softmax Operation - Pipeline Report

## Summary

**Operation**: `softmax`
**Result**: ALL 5 TDD STAGES PASSED
**Description**: Computes `exp(x_i) / sum(exp(x_j))` along a given dimension (dim=-1 or dim=-2), with optional numerical stability (max subtraction).

**Function signature**:
```python
softmax(input_tensor, dim=-1, *, numeric_stable=True) -> ttnn.Tensor
```

**Supported modes**: dim=-1 (width), dim=-2 (height), numeric_stable=True/False

---

## Pipeline Execution

| Phase | Agent | Output | Commits |
|-------|-------|--------|---------|
| 0: Discovery | orchestrator | Reference selection: reduce_w, reduce_h | — |
| 1: Analysis | ttnn-operation-analyzer (x2) | reduce_w_analysis.md, reduce_h_analysis.md | `03a1900ab8` |
| 2: Design | ttnn-operation-architect | op_design.md, .tdd_state.json | `8e10682b66`, `58f0fa158d` |
| 3: Build | ttnn-generic-op-builder | Python infra + stub kernels + tests | `d1d63a4c64`, `8bce5c157d` |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | 5 kernel implementations | `8bce5c157d`..`e41ff670cb` |
| 5: Report | orchestrator | REPORT.md | (this file) |

---

## Agent Summaries

### Phase 1: Analyzers (ttnn-operation-analyzer)

Two analyzers ran in parallel:

1. **Reduce W Analysis** (`reduce_w_analysis.md`): Analyzed `reduce_op_multi_core_w_program_factory.cpp`. Key findings:
   - Width reduction processes tiles in row-major order
   - Uses `reduce<>` template with configurable policies (WaitAndPopPerTile, NoWaitNoPop, etc.)
   - Scaler CB (index 8) holds packed bf16 reduce scaler
   - Work distributed per output row across cores

2. **Reduce H Analysis** (`reduce_h_analysis.md`): Analyzed `reduce_op_multi_core_h_program_factory.cpp`. Key findings:
   - Height reduction uses REDUCE_COL dimension (hardware terminology)
   - Chunked column processing limited by DEST_AUTO_LIMIT (4-16 tiles)
   - Reader uses column-interleaved tile ordering: N C W_skip H W_chunk
   - `reduce<>` library supports `post_reduce_op` lambda for in-place transformations

### Phase 2: Architect (ttnn-operation-architect)

Produced `op_design.md` with:
- **CB Layout**: CB0 (input), CB8 (scaler), CB16 (output), CB24-27 (intermediates for max, exp, sum, recip)
- **Work distribution**: Tiles distributed across available cores per batch/channel
- **Kernel strategy**: Separate reader/compute/writer kernels for dim=-1 and dim=-2 paths
- **TDD stages**: 5 progressive stages from passthrough to full stable softmax on both dimensions
- Registered all stages via `tdd_orchestrator.py`

### Phase 3: Builder (ttnn-generic-op-builder)

Created:
- `softmax.py` — Entry point with validation (dtype, layout, rank, dim)
- `softmax_program_descriptor.py` — CB config, kernel selection based on dim, runtime args
- 5 stub kernel files (reader_w, reader_h, compute_w, compute_h, writer)
- `__init__.py` — Re-exports
- Integration test + 5 TDD stage test files

### Phase 4: TDD Kernel Writer (ttnn-kernel-writer-tdd)

Implemented all 5 stages sequentially:

| Stage | Status | Attempts | Failures |
|-------|--------|----------|----------|
| data_pipeline_w | PASSED | 0 | None |
| exp_w | PASSED | 0 | None |
| softmax_unstable_w | PASSED | 2 (1 free + 1 hard) | compilation_error (NoAccumulation scope), numerical_mismatch (max diff 0.29) |
| softmax_stable_w | PASSED | 0 | None |
| softmax_stable_h | PASSED | 0 | None |

**Notable**: Stage `softmax_unstable_w` had 2 failures:
1. **Compilation error**: `NoAccumulation` not in scope — needed `compute_kernel_lib::NoAccumulation` qualification. Free retry.
2. **Numerical mismatch**: Max diff 0.29 — fixed by correcting reduce helper parameters. Hard attempt.

---

## TDD Pipeline Results

```
Stage 1: data_pipeline_w   ✅ PASSED (0 attempts, 0 failures)
Stage 2: exp_w              ✅ PASSED (0 attempts, 0 failures)
Stage 3: softmax_unstable_w ✅ PASSED (2 attempts, 2 failures then pass)
Stage 4: softmax_stable_w   ✅ PASSED (0 attempts, 0 failures)
Stage 5: softmax_stable_h   ✅ PASSED (0 attempts, 0 failures)

Overall: 5/5 stages passed
Total attempts used: 2 (out of 30 total budget)
Failure classifications: 1x compilation_error, 1x numerical_mismatch
```

---

## Files Produced

### Operation code (`ttnn/ttnn/operations/softmax/`)
```
├── __init__.py
├── softmax.py                          # Entry point with validation
├── softmax_program_descriptor.py       # CB config, work distribution
├── kernels/
│   ├── softmax_reader_w.cpp           # Reader for dim=-1
│   ├── softmax_reader_h.cpp           # Reader for dim=-2
│   ├── softmax_compute_w.cpp          # Compute for dim=-1
│   ├── softmax_compute_h.cpp          # Compute for dim=-2
│   └── softmax_writer.cpp             # Shared writer
├── op_design.md                        # Architecture design doc
├── .tdd_state.json                     # TDD pipeline state
└── agent_logs/                         # Breadcrumbs and logs
    ├── reduce_w_analysis.md
    ├── reduce_h_analysis.md
    ├── ttnn-operation-analyzer_breadcrumbs.jsonl
    ├── ttnn-operation-architect_breadcrumbs.jsonl
    ├── ttnn-operation-architect_execution_log.md
    ├── ttnn-generic-op-builder_breadcrumbs.jsonl
    ├── ttnn-generic-op-builder_execution_log.md
    └── ttnn-kernel-writer-tdd_breadcrumbs.jsonl
```

### Tests (`tests/ttnn/unit_tests/operations/softmax/`)
```
├── __init__.py
├── test_softmax.py                    # Integration test
├── test_stage_data_pipeline_w.py      # TDD stage 1
├── test_stage_exp_w.py                # TDD stage 2
├── test_stage_softmax_unstable_w.py   # TDD stage 3
├── test_stage_softmax_stable_w.py     # TDD stage 4
└── test_stage_softmax_stable_h.py     # TDD stage 5
```

---

## Git History

```
e41ff670cb [ttnn-kernel-writer-tdd] stage softmax_stable_h: passed
65a41f0c5d [ttnn-kernel-writer-tdd] stage softmax_stable_w: passed
9a34e7f705 [ttnn-kernel-writer-tdd] stage softmax_unstable_w: passed
1bd17b9d38 [ttnn-kernel-writer-tdd] stage exp_w: passed
b90607ed84 [ttnn-kernel-writer-tdd] stage data_pipeline_w: passed
8bce5c157d [ttnn-generic-op-builder] logs: softmax execution log and breadcrumbs
d1d63a4c64 [ttnn-generic-op-builder] stubs: softmax
58f0fa158d [ttnn-operation-architect] finalize: softmax execution log and breadcrumbs
8e10682b66 [ttnn-operation-architect] design: softmax
530595d38d [ttnn-operation-analyzer] update breadcrumbs for reduce_w analysis
03a1900ab8 [ttnn-operation-analyzer] analysis: reduce_w
```

---

## Decisions and Deviations

1. **Reference selection**: Used reduce_w and reduce_h factories as primary references (not the existing C++ softmax). The reduce factories provided clearer patterns for the generic_op infrastructure.

2. **Kernel split**: Separate compute kernels for dim=-1 (compute_w) and dim=-2 (compute_h) rather than a single kernel with runtime branching. This simplifies each kernel and follows the reduce_op pattern.

3. **Shared writer**: Single writer kernel serves both dim=-1 and dim=-2 paths since output writing logic is identical.

4. **TDD stage ordering**: Progressive build-up from passthrough → exp → unstable softmax → stable softmax → height dim. This allowed incremental validation.

5. **Numerical tolerance**: Used rtol=0.02, atol=0.1 for softmax stages (bfloat16 has limited precision, especially after multiple ops like exp + reduce + recip + mul).

---

## Infrastructure Issues

- **No device hangs encountered** during TDD testing
- **No build failures** (kernels compile at runtime)
- **Compilation error** in stage 3 was a scoping issue (`NoAccumulation` vs `compute_kernel_lib::NoAccumulation`), resolved in 1 retry
- **Numerical mismatch** in stage 3 required adjusting reduce helper parameters, resolved in 1 additional attempt

---

## Suggestions for Improving the Agent Pipeline

1. **Analyzer focus directives worked well**: Role-based scoping kept analysis focused and reduced noise for the architect.
2. **TDD staging was effective**: Progressive stages caught issues early (the compilation error in stage 3 would have been harder to debug in a more complex stage).
3. **The `NoAccumulation` scoping issue** suggests the kernel writer could benefit from a reference of common namespace-qualified identifiers.
4. **Parallel analyzer execution** saved significant time — both completed in ~14 minutes.
