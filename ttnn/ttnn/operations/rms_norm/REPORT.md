# RMS Norm Operation - Pipeline Report

## Summary

| Field | Value |
|-------|-------|
| **Operation** | `rms_norm` |
| **Math** | `RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + eps) * gamma` |
| **Result** | **ALL 4 TDD STAGES PASSED** |
| **Op Path** | `ttnn/ttnn/operations/rms_norm/` |
| **Test Path** | `tests/ttnn/unit_tests/operations/rms_norm/` |
| **Import** | `from ttnn.operations.rms_norm import rms_norm` |
| **Layouts** | ROW_MAJOR_LAYOUT, TILE_LAYOUT (both tested per stage) |
| **Dtypes** | bfloat16 (float32 supported via fp32_dest_acc_en) |

---

## Pipeline Execution

| Phase | Agent | Status | Key Output |
|-------|-------|--------|------------|
| 0: Discovery | Coordinator | COMPLETE | 3 references: tilize, untilize, batch_norm |
| 1: Analysis | ttnn-operation-analyzer (x3) | COMPLETE | tilize_analysis.md, untilize_analysis.md, batch_norm_analysis.md |
| 2: Design | ttnn-operation-architect | COMPLETE | op_design.md (352 lines), 4 TDD stages registered |
| 3: Build | ttnn-generic-op-builder | COMPLETE | Python infra + stub kernels + test files |
| 4: TDD Kernels | ttnn-kernel-writer-tdd (x3 sessions) | COMPLETE | All 4 stages passed |
| 5: Report | Coordinator | COMPLETE | This file |
| 6: Self-Reflection | ttnn-self-reflection | PENDING | |

---

## Agent Summaries

### Phase 1: Analyzers (3 parallel agents)

- **tilize-analyzer** (input_stage): Analyzed `tilize_single_core_program_factory.cpp`. Documented how RM sticks are read from DRAM, CB sizing for stick-to-tile batching (32 sticks per tile), and single-core work distribution. Output: `agent_logs/tilize_analysis.md` (20KB).

- **untilize-analyzer** (output_stage): Analyzed `untilize_single_core_program_factory.cpp`. Documented untilize helper usage, writer kernel pattern for RM stick output, and output CB sizing. Output: `agent_logs/untilize_analysis.md` (23KB).

- **batchnorm-analyzer** (compute_core): Analyzed `batch_norm_program_factory.cpp`. Documented compute kernel structure, CB layout for intermediates, SFPU operation sequences, and scalar/constant CB setup. Output: `agent_logs/batch_norm_analysis.md` (33KB).

### Phase 2: Architect

Produced `op_design.md` covering:
- Two-pass data flow: Pass 1 (square + reduce), Pass 2 (rsqrt + normalize + gamma)
- 12 circular buffers (cb0-cb26) with role-specific sizing
- Single-core work distribution (all tile-rows on core 0,0)
- 4 TDD stages with incremental complexity

### Phase 3: Builder

Created complete Python infrastructure:
- `rms_norm.py` (117 lines): Entry point with validation
- `rms_norm_program_descriptor.py` (409 lines): CB config, kernel setup, runtime args
- 3 stub kernel files + 5 test files (1 integration + 4 stage tests)
- All tests parametrized over both layouts

### Phase 4: TDD Kernel Writer

Required 3 agent sessions due to context window limits:
1. Session 1: Completed stage 1 (data_pipeline)
2. Session 2: Completed stage 2 (square_reduce)
3. Session 3: Completed stages 3 (rms_normalize) and 4 (gamma_scale)

---

## TDD Pipeline Results

| Stage | Name | Result | Attempts | Upstream Fixes |
|-------|------|--------|----------|----------------|
| 1 | data_pipeline | PASS | 1 | None (clean pass) |
| 2 | square_reduce | PASS | 1 | Output shape changed to reduced `[..., 32]` for intermediate stage |
| 3 | rms_normalize | PASS | 1 | Reverted output shape to full input; fixed rsqrt include; used `prepare_reduce_scaler` for eps fill |
| 4 | gamma_scale | PASS | 2 | Fixed gamma TA placeholder `[0]` → `[0, 0]`; fixed cb_gamma/cb_norm from 2 to Wt pages; fixed gamma stick replication |

**No device hangs encountered.** All stages passed on both ROW_MAJOR_LAYOUT and TILE_LAYOUT.

---

## Files Produced

### Operation (`ttnn/ttnn/operations/rms_norm/`)
```
__init__.py                              # Re-export rms_norm
rms_norm.py                              # Entry point (117 lines)
rms_norm_program_descriptor.py           # Program descriptor (409 lines)
kernels/rms_norm_reader.cpp              # Reader kernel (164 lines)
kernels/rms_norm_compute.cpp             # Compute kernel (143 lines)
kernels/rms_norm_writer.cpp              # Writer kernel (69 lines)
op_design.md                             # Design document (352 lines)
.tdd_state.json                          # TDD pipeline state
```

### Tests (`tests/ttnn/unit_tests/operations/rms_norm/`)
```
test_rms_norm.py                         # Integration test
test_stage_data_pipeline.py              # TDD stage 1
test_stage_square_reduce.py              # TDD stage 2
test_stage_rms_normalize.py              # TDD stage 3
test_stage_gamma_scale.py                # TDD stage 4
```

### Logs (`ttnn/ttnn/operations/rms_norm/agent_logs/`)
```
pipeline_breadcrumbs.md                  # Pipeline coordinator log
tilize_analysis.md                       # Tilize reference analysis
untilize_analysis.md                     # Untilize reference analysis
batch_norm_analysis.md                   # Batch norm reference analysis
ttnn-operation-analyzer_breadcrumbs.jsonl
ttnn-operation-architect_breadcrumbs.jsonl
ttnn-generic-op-builder_breadcrumbs.jsonl
ttnn-generic-op-builder_execution_log.md
ttnn-kernel-writer-tdd_breadcrumbs.jsonl
```

---

## Git History

```
b99fe3b154 [ttnn-kernel-writer-tdd] stage gamma_scale: passed
e528fa9bb0 [ttnn-kernel-writer-tdd] stage rms_normalize: passed
ceef161c1b [ttnn-kernel-writer-tdd] stage square_reduce: passed
b2f32ac777 [ttnn-kernel-writer-tdd] stage data_pipeline: passed
04309616fc [ttnn-generic-op-builder] logs: rms_norm execution log and breadcrumbs
3a0be7020e [ttnn-generic-op-builder] stubs: rms_norm
2eae63ffac [ttnn-operation-architect] finalize: commit completion breadcrumbs
66fa0f3324 [ttnn-operation-architect] design: rms_norm
800aa86427 [ttnn-operation-analyzer] update breadcrumbs for untilize analysis
b3d712500a [ttnn-operation-analyzer] analysis: untilize (output_stage reference)
```

---

## Decisions and Deviations

### Assumptions Made
1. **Single-core implementation**: Used 1x1 core grid for simplicity. Multi-core can be added later.
2. **bfloat16 testing only**: Stage tests use bfloat16. Float32 support is configured (fp32_dest_acc_en) but not separately tested in TDD stages.
3. **Standard tile size (32x32)**: All tiles use the standard 32x32 format.

### Deviations from Design
1. **Epsilon fill method**: Used `prepare_reduce_scaler` instead of custom broadcast fill for epsilon CB. Functionally equivalent since epsilon is used with SCALAR broadcast, and `prepare_reduce_scaler` fills row 0 of each face which is exactly what SCALAR broadcast reads.
2. **Stage 2 output shape**: Temporarily changed output to reduced shape `[..., 32]` for the square_reduce stage, then reverted to full shape for stage 3.
3. **CB page counts**: cb_gamma (c4) and cb_norm (c26) were increased from 2 pages to Wt pages. The design specified 2 pages (streaming), but tilize/mul helpers need all Wt tiles available simultaneously to avoid deadlock when Wt > 2.

### Upstream Fixes by TDD Writer
- Fixed gamma TensorAccessorArgs placeholder from `[0]` to `[0, 0]` (interleaved TA uses 2 CT args)
- Fixed gamma stick replication (read page 0 x32 instead of pages 0..31, since gamma is single-stick)
- Fixed rsqrt include path

---

## Infrastructure Issues

- **Pre-commit hook failure**: The "Convert .ipynb to .py" hook failed during commits by analyzer agents, but the analysis files were created successfully. This did not block the pipeline.
- **Context window limits**: The TDD kernel writer required 3 separate agent sessions to complete all 4 stages, as each session hit context limits. The agent resumed correctly each time.
- **No device hangs**: All test runs completed without device hangs or timeouts.
- **No build issues**: Kernels compiled successfully at runtime (no C++ build step needed).

---

## Suggestions for Pipeline Improvement

1. **Context efficiency**: The TDD kernel writer consumed 3 agent sessions. Consider compressing analysis content passed to later stages, or implementing incremental context management.
2. **CB sizing validation**: The design specified 2 pages for cb_gamma and cb_norm, but the TDD writer discovered they need Wt pages. The architect should account for tilize helper requirements that need full tile-row buffers.
3. **Gamma TA placeholder**: The builder generated `[0]` as placeholder for absent gamma TensorAccessorArgs, but interleaved layout requires 2 CT args. The template should document the correct placeholder size per layout type.
4. **Stage 2 output shape handling**: The square_reduce stage required a temporary output shape change that was reverted in stage 3. Consider whether intermediate stages should use a separate output tensor or skip shape changes.
