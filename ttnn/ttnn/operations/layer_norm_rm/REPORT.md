# layer_norm_rm - Pipeline Report

## Summary

**Operation**: `layer_norm_rm`
**Description**: Layer normalization on row-major interleaved tensors with optional gamma/beta affine parameters.
**Overall Result**: SUCCESS - All 4 TDD stages passed, operation fully functional.
**Date**: 2026-03-10

### Math Definition
```
mean = row_sum(x) / W
centered = x - mean
var = row_sum(centered²) / W
output = centered * rsqrt(var + epsilon)
output = output * gamma + beta  (optional affine)
```

### Function Signature
```python
layer_norm_rm(input_tensor, gamma=None, beta=None, *, epsilon=1e-5)
```

---

## Pipeline Execution

| Phase | Agent | Duration (approx) | Output |
|-------|-------|--------------------|--------|
| Phase 0: Discovery | Main orchestrator | ~2 min | 3 reference operations identified |
| Phase 1: Analysis | 3x ttnn-operation-analyzer (parallel) | ~14 min | tilize, untilize, softmax analyses |
| Phase 2: Design | ttnn-operation-architect | ~9 min | op_design.md + .tdd_state.json |
| Phase 3: Build | ttnn-generic-op-builder | ~10 min | Python infra + stub kernels + tests |
| Phase 4: TDD Kernels | ttnn-kernel-writer-tdd | ~26 min | All 4 stages implemented and passing |
| Phase 5: Reporting | Main orchestrator | ~1 min | REPORT.md |

**Total pipeline duration**: ~62 minutes

---

## Per-Agent Summaries

### Analyzer Agents (Phase 1)

Three analyzers ran in parallel with role-based focus directives:

1. **Tilize Analyzer** (input_stage)
   - Analyzed `tilize_multi_core_interleaved_program_factory.cpp`
   - Key findings: RM stick reading pattern, input CB sizing, stick-to-tile batching, core assignment strategy
   - Output: `agent_logs/tilize_analysis.md` (20KB)

2. **Untilize Analyzer** (output_stage)
   - Analyzed `untilize_multi_core_program_factory.cpp`
   - Key findings: `pack_untilize_block` helper usage, output CB sizing, stick extraction pattern, writer kernel DRAM write pattern
   - Output: `agent_logs/untilize_analysis.md` (24KB)

3. **Softmax Analyzer** (compute_core)
   - Analyzed `tt-train/sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp`
   - Key findings: Multi-pass row reduction structure, CB layout for intermediates, matmul-based reduction for precision, DST register constraints with fp32_dest_acc_en, masking strategy
   - Output: `agent_logs/softmax_analysis.md` (43KB)

### Architect Agent (Phase 2)

- Designed hybrid operation: tilize input + softmax-like compute + untilize output
- CB layout: 11+ circular buffers covering input, output, intermediates, and scalars
- Multi-pass compute: tilize → mean reduction → subtract → square → variance reduction → rsqrt → normalize → optional affine → untilize
- Registered 4 TDD stages with incremental complexity
- Output: `op_design.md` (19KB)

### Builder Agent (Phase 3)

- Created Python entry point with validation (dtype, layout, gamma/beta shape checks)
- Created program descriptor with CB configuration and work distribution
- Generated 3 stub kernel files
- Created integration test + verified 4 TDD stage test files
- Output: Python infrastructure + stubs + tests

### TDD Kernel Writer (Phase 4)

- Implemented all 4 stages sequentially with test verification
- Produced 3 complete kernel files: reader (4.7KB), compute (6.9KB), writer (1.7KB)
- Made 1 upstream fix to program descriptor during affine_transform stage
- Output: Working kernels passing all tests

---

## TDD Pipeline Results

| Stage | Status | Attempts | Retries | Failure Classification |
|-------|--------|----------|---------|----------------------|
| 1. data_pipeline | PASSED | 1 | 0 | — |
| 2. reduce_mean_sub | PASSED | 1 | 0 | — |
| 3. variance_normalize | PASSED | 1 | 0 | — |
| 4. affine_transform | PASSED | 2 | 0 | 1x runtime_error (TT_FATAL in tilize) |

### Stage Details

**Stage 1 - data_pipeline**: Identity passthrough (RM → tilize → untilize → RM). Verified basic data movement pipeline works. First-pass success.

**Stage 2 - reduce_mean_sub**: Added row-wise mean computation (SUM reduce with 1/W scaler) and mean subtraction (SUB with column broadcast). Output: `x - mean(x)`. First-pass success.

**Stage 3 - variance_normalize**: Added squaring, variance reduction, epsilon addition, rsqrt, and normalization multiply. Full layer norm without affine. First-pass success.

**Stage 4 - affine_transform**: Added gamma/beta reader paths and affine application (multiply gamma, add beta). Had one failure on attempt 1 due to a runtime error in tilize — the kernel writer diagnosed this as an issue with gamma/beta tensor handling in the program descriptor and fixed it. Passed on attempt 2.

### Failure Detail (Stage 4, Attempt 1)
- **Classification**: runtime_error
- **Error**: `TT_FATAL @ tilize_device_operation.cpp:3`
- **Root Cause**: Program descriptor issue with gamma/beta tensor address passing
- **Resolution**: Fixed upstream program descriptor to correctly handle optional gamma/beta tensors
- **Cost**: HARD (1 retry budget consumed)

---

## Files Produced

### Operation Directory: `ttnn/ttnn/operations/layer_norm_rm/`
```
├── __init__.py                              # Re-exports layer_norm_rm()
├── layer_norm_rm.py                         # Entry point with validation (5.2KB)
├── layer_norm_rm_program_descriptor.py      # CB config, work distribution (12.2KB)
├── op_design.md                             # Operation design document (19.4KB)
├── .tdd_state.json                          # TDD pipeline state (5.6KB)
├── .breadcrumbs.md                          # TDD kernel writer breadcrumbs (5.1KB)
├── REPORT.md                                # This report
├── kernels/
│   ├── layer_norm_rm_reader.cpp             # Reader kernel (4.7KB)
│   ├── layer_norm_rm_compute.cpp            # Compute kernel (6.9KB)
│   └── layer_norm_rm_writer.cpp             # Writer kernel (1.7KB)
└── agent_logs/
    ├── tilize_analysis.md                   # Tilize reference analysis (20KB)
    ├── untilize_analysis.md                 # Untilize reference analysis (24KB)
    ├── softmax_analysis.md                  # Softmax reference analysis (43KB)
    ├── ttnn-operation-analyzer_breadcrumbs.jsonl
    ├── ttnn-operation-architect_breadcrumbs.jsonl
    ├── ttnn-generic-op-builder_breadcrumbs.jsonl
    ├── ttnn-generic-op-builder_execution_log.md
    └── ttnn-kernel-writer-tdd_breadcrumbs.jsonl
```

### Test Directory: `tests/ttnn/unit_tests/operations/layer_norm_rm/`
```
├── __init__.py
├── layer_norm_rm.py                         # Test helper module
├── test_layer_norm_rm.py                    # Integration test (3.9KB)
├── test_stage_data_pipeline.py              # TDD Stage 1 test (1.7KB)
├── test_stage_reduce_mean_sub.py            # TDD Stage 2 test (1.7KB)
├── test_stage_variance_normalize.py         # TDD Stage 3 test (1.8KB)
└── test_stage_affine_transform.py           # TDD Stage 4 test (2.3KB)
```

---

## Git History

| Commit | Message |
|--------|---------|
| `9128348cab` | [ttnn-operation-analyzer] analysis: tilize (multi-core interleaved) |
| `045e313a5f` | [ttnn-operation-analyzer] analysis: untilize (multi-core interleaved) |
| `47913da0f6` | [ttnn-operation-analyzer] breadcrumbs: tilize analysis completion event |
| `7706ef5c67` | [ttnn-operation-analyzer] analysis: softmax (compute core focus) |
| `17554c0316` | [ttnn-operation-architect] design: layer_norm_rm |
| `94d59a85a1` | [ttnn-operation-architect] breadcrumbs: completion events |
| `06d535778c` | [ttnn-generic-op-builder] stubs: layer_norm_rm |
| `f3887acd22` | [ttnn-generic-op-builder] breadcrumbs: completion event |
| `335953a11f` | [ttnn-kernel-writer-tdd] stage data_pipeline: passed |
| `5724c2b22e` | [ttnn-kernel-writer-tdd] stage reduce_mean_sub: passed |
| `5bbd5a8a89` | [ttnn-kernel-writer-tdd] stage variance_normalize: passed |
| `4e453b5285` | [ttnn-kernel-writer-tdd] stage affine_transform: passed |

---

## Key Decisions and Deviations

### Decisions Made
1. **Hybrid mode**: Combined tilize (input), softmax (compute pattern), untilize (output) as references
2. **Softmax from tt-train**: Used `tt-train/sources/ttml/metal/ops/softmax/` since the main TTNN softmax was removed in recent commits
3. **4 TDD stages**: Incremental build-up from identity passthrough → mean/sub → full normalize → affine transform
4. **Row-major throughout**: Input and output stay in RM layout; tilize/untilize happen inside kernels

### Deviations from Spec
- None significant. All spec requirements implemented as specified.

### Pain Points
1. **Stage 4 runtime error**: The affine transform stage had one failure due to program descriptor issues with optional gamma/beta tensor handling. The TDD kernel writer autonomously diagnosed and fixed the upstream issue.

---

## Infrastructure Issues

- **Device access**: No device access errors encountered
- **Device hangs**: No hangs detected during any test run
- **Build**: No C++ build required (kernels compile at runtime via generic_op)
- **Python venv**: No venv issues
- **Test framework**: `scripts/tt-test.sh --dev` worked correctly for all TDD stages

---

## Recommendations for Pipeline Improvement

1. **Analyzer output size**: Even with role-based focus directives, analyses are large (20-43KB). Consider further scoping to reduce architect context consumption.
2. **Stage 4 upstream fix pattern**: The TDD kernel writer had to fix the program descriptor during stage 4. Consider having the builder create more complete program descriptors that account for optional inputs from the start.
3. **Parallel analyzer efficiency**: All 3 analyzers ran in parallel successfully (~14 min wall clock). This is a good pattern to maintain.
4. **First-pass success rate**: 3 of 4 stages passed on first attempt (75%). The single retry was for a legitimate upstream issue, not a kernel bug.
