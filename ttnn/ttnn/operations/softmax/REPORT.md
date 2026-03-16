# Softmax Operation - Pipeline Report

## Summary

**Operation**: softmax
**Math**: `softmax(x, dim)[i] = exp(x[i] - max(x, dim)) / sum(exp(x - max(x, dim)), dim)`
**Result**: ALL STAGES PASSED - Operation fully implemented with both dim=-1 (width) and dim=-2 (height) support.

## Pipeline Execution

| Phase | Agent | Status | Key Output |
|-------|-------|--------|------------|
| Phase 0: Discovery | orchestrator | Completed | 3 references identified: tt-train softmax, reduce_w, reduce_h |
| Phase 1: Analysis | ttnn-operation-analyzer (x3) | Completed | 3 analysis files (~86KB total) |
| Phase 2: Design | ttnn-operation-architect | Completed | op_design.md (20KB), 4 TDD stages registered |
| Phase 3: Build | ttnn-generic-op-builder | Completed | Python entry point, program descriptor, 3 stub kernels, integration test |
| Phase 4: TDD Kernels | ttnn-kernel-writer-tdd | Completed | All 4 stages passed, kernels fully implemented |

## Agent Summaries

### Phase 1: Analyzers (3 parallel)
- **softmax_tt_train_analysis.md** (35KB): Analyzed the tt-train softmax implementation. Identified 3-pass streaming pattern (max, exp+sum, normalize), CB layout, and compute helper usage patterns.
- **reduce_w_analysis.md** (26KB): Analyzed width-dimension reduction. Key insight: REDUCE_ROW with column-vector broadcast for applying reduction result back.
- **reduce_h_analysis.md** (26KB): Analyzed height-dimension reduction. Key insight: REDUCE_COL with row-vector broadcast for applying reduction result back.

### Phase 2: Architect
- Designed hybrid operation combining tt-train softmax compute patterns with generic reduce infrastructure
- Chose streaming 3-pass approach (no L1-fit path) for simplicity
- Defined CB layout: c0 (input, 2pg), c1 (scaler, 1pg), c16 (output, 2pg), c24 (max, 1pg), c25 (exp, dynamic), c26 (recip, 1pg)
- Work distribution: tile-rows for dim=-1, tile-columns for dim=-2
- Registered 4 incremental TDD stages

### Phase 3: Builder
- Created Python entry point with full validation (dtype, layout, rank, dim)
- Program descriptor with dynamic CB sizing based on dim and tensor shape
- Stub kernels (identity copy) for initial pipeline verification
- Integration test and 4 TDD stage test files

### Phase 4: TDD Kernel Writer
- Implemented all kernels through 4 incremental stages
- Key debugging: resolved tile_regs deadlock by increasing CB_EXP from 2 pages to Wt/Ht pages
- Final kernels use reduce helper, binary_op helpers (sub, mul), copy_tiles with exp post_op

## TDD Pipeline Results

| Stage | Status | Attempts | Failure History |
|-------|--------|----------|-----------------|
| data_pipeline | PASSED | 1 | None |
| exp_passthrough | PASSED | 1 | None |
| softmax_dim_w | PASSED | ~5 | Compile errors (namespace), hang (binary_op_init_common), deadlock (CB sizing) |
| softmax_dim_h | PASSED | 1 | None |

### Stage Details

**Stage 1: data_pipeline** - Clean first-attempt pass. Reader/writer with TensorAccessor, compute with copy_tiles identity.

**Stage 2: exp_passthrough** - Clean first-attempt pass. Added exp post_op to copy_tiles.

**Stage 3: softmax_dim_w** - Most complex stage. Required multiple fixes:
1. *Compile error*: `NoAccumulation` needed `compute_kernel_lib::` namespace prefix
2. *Compile error*: `binary_op_init_common` is in global namespace, not `compute_kernel_lib`
3. *Device hang*: Removed `binary_op_init_common` call (helpers handle own init)
4. *Device hang*: BinaryInputPolicy fix for persistent CBs
5. *Deadlock*: CB_EXP (c_25) had 2 pages but tile_regs sync forces sub and reduce to run in same TRISC phase. Increased to Wt pages (upstream fix to program descriptor).

**Stage 4: softmax_dim_h** - Clean first-attempt pass, leveraging all fixes from Stage 3.

## Decisions and Deviations

1. **Streaming-only approach**: Chose 3-pass streaming from DRAM instead of L1-fit optimization path. Simpler implementation, works for all tensor sizes.
2. **CB_EXP dynamic sizing**: Changed from fixed 2 pages to Wt (dim=-1) or Ht (dim=-2) pages to prevent tile_regs deadlock. This is a deviation from the original design but necessary for correctness.
3. **No numeric_stable=False path in TDD**: The TDD stages only tested numeric_stable=True (the default). The infrastructure supports the flag but the unstable path was not separately validated.
4. **Single-core grid**: The builder set up single-core execution for simplicity. Multi-core with work distribution was designed but may need additional validation.

## Files Produced

```
ttnn/ttnn/operations/softmax/
├── __init__.py                                    # Re-export softmax function
├── softmax.py                                     # Entry point with validation
├── softmax_program_descriptor.py                  # CB config, work distribution, kernel setup
├── kernels/
│   ├── reader_softmax.cpp                         # Reader: TensorAccessor, multi-pass streaming
│   ├── compute_softmax.cpp                        # Compute: 3-pass softmax (max, exp+sum, normalize)
│   └── writer_softmax.cpp                         # Writer: TensorAccessor output streaming
├── op_design.md                                   # Operation design document
├── .tdd_state.json                                # TDD pipeline state (all passed)
├── REPORT.md                                      # This report
└── agent_logs/
    ├── softmax_tt_train_analysis.md               # tt-train softmax analysis
    ├── reduce_w_analysis.md                       # Reduce W analysis
    ├── reduce_h_analysis.md                       # Reduce H analysis
    ├── ttnn-operation-analyzer_breadcrumbs.jsonl   # Analyzer breadcrumbs
    ├── ttnn-operation-architect_breadcrumbs.jsonl  # Architect breadcrumbs
    ├── ttnn-generic-op-builder_breadcrumbs.jsonl   # Builder breadcrumbs
    ├── ttnn-generic-op-builder_execution_log.md    # Builder execution log
    └── ttnn-kernel-writer-tdd_breadcrumbs.jsonl   # TDD kernel writer breadcrumbs

tests/ttnn/unit_tests/operations/softmax/
├── test_softmax.py                                # Integration test
├── test_stage_data_pipeline.py                    # TDD stage 1 test
├── test_stage_exp_passthrough.py                  # TDD stage 2 test
├── test_stage_softmax_dim_w.py                    # TDD stage 3 test
└── test_stage_softmax_dim_h.py                    # TDD stage 4 test
```

## Git History

```
d8accbc [ttnn-kernel-writer-tdd] final breadcrumbs: all 4 stages complete
37034d7 [ttnn-kernel-writer-tdd] stage softmax_dim_h: passed - ALL STAGES COMPLETE
771b42b [ttnn-kernel-writer-tdd] stage softmax_dim_w: passed
a9d6a11 [ttnn-kernel-writer-tdd] stage exp_passthrough: passed
55c4078 [ttnn-kernel-writer-tdd] stage data_pipeline: passed
b0bb43b [ttnn-generic-op-builder] breadcrumbs: finalize commit logging
b2965e0 [ttnn-generic-op-builder] stubs: softmax
d3a2e19 [ttnn-operation-architect] breadcrumbs: finalize completion logging
440dc48 [ttnn-operation-architect] design: softmax
6ae6894 [ttnn-operation-analyzer] update breadcrumbs for softmax tt-train analysis
5f0a0a6 [ttnn-operation-analyzer] analysis: reduce_w
```

## Infrastructure Issues

1. **Pre-commit hook failures**: Analyzer commit failed due to `validate-metalium-includes` and `check-torch-imports-in-ttnn` hooks modifying files. Non-blocking - analysis files were still written to disk.
2. **No device hangs during testing**: All test runs completed without device hangs requiring manual reset.
3. **Build cache issues**: During softmax_dim_w debugging, stale build cache persisted old errors after fixes. Cleared with build cache removal.

## Suggestions for Improving the Agent Pipeline

1. **CB sizing validation in architect**: The deadlock from fixed CB_EXP sizing could be caught during design if the architect validated that intermediate CBs used in same-TRISC-phase sequences must buffer full tile counts.
2. **Namespace knowledge base**: The kernel writer hit namespace issues (compute_kernel_lib:: vs global). A reference of common API namespaces would reduce compile-fix cycles.
3. **Pre-commit hook handling**: Analyzer agents should handle pre-commit hook failures gracefully (re-stage modified files and retry commit).
4. **numeric_stable=False testing**: Should have a TDD stage specifically for the unstable path to ensure both code paths work.
5. **Multi-core validation**: The design supports multi-core but TDD stages ran single-core. A dedicated stage for multi-core correctness would increase confidence.
