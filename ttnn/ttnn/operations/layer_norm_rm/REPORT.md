# Pipeline Report: layer_norm_rm

## Summary

**Operation**: `layer_norm_rm` — Layer normalization on row-major interleaved tensors
**Result**: ALL STAGES PASSED (3/3)
**Total commits**: 9 (3 analysis + 1 design + 1 build + 1 build logs + 3 TDD stages)
**Branch**: `mstaletovic/10_02_LN_TDD_BREAD`

## Pipeline Execution

| Phase | Agent | Status | Key Output |
|-------|-------|--------|------------|
| 0 - Discovery | Orchestrator | PASS | 3 references: tilize (input), untilize (output), batch_norm (compute) |
| 1 - Analysis | ttnn-operation-analyzer ×3 | PASS | 3 analysis files (tilize, untilize, batch_norm) |
| 2 - Design | ttnn-operation-architect | PASS | op_design.md (388 lines), 3 TDD stages registered |
| 3 - Build | ttnn-generic-op-builder | PASS | Python infra + stubs, 6/6 integration tests pass |
| 4 - TDD Kernels | ttnn-kernel-writer-tdd | PASS | 3/3 stages pass, 3 hard attempts total |
| 5 - Report | Orchestrator | PASS | This file |
| 6 - Self-Reflection | ttnn-self-reflection | (pending) | |

## Per-Agent Summary

### Analyzers (Phase 1) — 3 parallel agents
- **Tilize (input_stage)**: Identified stick-to-tile batching (32 sticks per CB push), work distribution (1D block splitting by tile-rows), TensorAccessor setup pattern
- **Untilize (output_stage)**: Documented untilize helper signature, output CB sizing (Wt×2 tiles double-buffered), writer stick extraction pattern
- **Batch norm (compute_core)**: Found `binary_dest_reuse_tiles` optimization, three-tier CB lifetime pattern, dynamic CB routing for optional affine params, FPU vs SFPU paths

### Architect (Phase 2)
- Designed 10-phase compute pipeline: tilize → reduce mean → subtract → square → reduce var → add eps+rsqrt → mul inv_std → gamma mul → beta add → untilize
- Defined 13 circular buffers with role assignments
- Registered 3 TDD stages with progressive complexity

### Builder (Phase 3)
- Created `layer_norm_rm.py` entry point with validation (dtype, layout, gamma/beta width)
- Created `layer_norm_rm_program_descriptor.py` with 13 CBs, work distribution, per-core runtime args
- Stub kernels compile and run on device — 6/6 integration tests pass

### Kernel Writer (Phase 4)
- Implemented full 10-phase compute kernel using kernel_lib helpers
- Reader: RM stick reads via TensorAccessor, reduce scaler/epsilon prep, gamma/beta tile reads
- Writer: RM stick writes via TensorAccessor (minimal changes from stub)
- Found and fixed 2 design doc errors and 2 upstream Python issues

## TDD Pipeline Results

| Stage | Name | Result | Hard Attempts | Free Retries | Failure Classifications |
|-------|------|--------|--------------|--------------|------------------------|
| 0 | mean_subtract | PASS | 0/6 | 0 | — |
| 1 | full_normalize | PASS | 1/6 | 0 | numerical_mismatch (SCALAR→COL broadcast fix) |
| 2 | affine_transform | PASS | 2/6 | 0 | runtime_error (tilize volume), hang_cb_deadlock (Bulk→PerTile output) |

**Total hard attempts**: 3 out of 18 budget (16.7% utilization)

### Failure Details

1. **Stage 1, attempt 1** (`numerical_mismatch`): Design doc specified SCALAR broadcast for inv_std multiplication, but REDUCE_ROW output stores per-row values in column 0. SCALAR only broadcasts `[0][0]`. Fixed by changing to COL broadcast. Max diff: 1.6875.

2. **Stage 2, attempt 1** (`runtime_error`): `ttnn.tilize()` requires tensor volume ≥ TILE_HW (1024), but gamma/beta at shape (1,1,1,W) with small W fails. Fixed by padding H to 32 with `ttnn.pad` before tilize.

3. **Stage 2, attempt 2** (`hang_cb_deadlock`): Bulk output policy on gamma/beta phases deadlocks when Wt > 1 — Bulk tries to reserve all output pages while input pages still occupy the shared CB. Fixed by switching to PerTile output policy.

## Files Produced

### Operation (`ttnn/ttnn/operations/layer_norm_rm/`)
```
├── __init__.py                              # Re-export layer_norm_rm
├── layer_norm_rm.py                         # Entry point + validation
├── layer_norm_rm_program_descriptor.py      # ProgramDescriptor builder (13 CBs)
├── kernels/
│   ├── layer_norm_rm_reader.cpp             # RM stick reader + scaler/gamma/beta
│   ├── layer_norm_rm_compute.cpp            # 10-phase compute pipeline
│   └── layer_norm_rm_writer.cpp             # RM stick writer
├── op_design.md                             # Architecture + kernel design
├── .tdd_state.json                          # TDD pipeline state (all passed)
├── REPORT.md                                # This file
└── agent_logs/
    ├── tilize_analysis.md                   # Input stage analysis
    ├── untilize_analysis.md                 # Output stage analysis
    ├── batch_norm_analysis.md               # Compute core analysis
    ├── ttnn-operation-analyzer_breadcrumbs.jsonl
    ├── ttnn-operation-architect_breadcrumbs.jsonl
    ├── ttnn-generic-op-builder_breadcrumbs.jsonl
    ├── ttnn-generic-op-builder_execution_log.md
    └── ttnn-kernel-writer-tdd_breadcrumbs.jsonl
```

### Tests (`tests/ttnn/unit_tests/operations/layer_norm_rm/`)
```
├── __init__.py
├── layer_norm_rm.py                         # Re-export for stage test imports
├── test_layer_norm_rm.py                    # Integration test (6 cases)
├── test_stage_mean_subtract.py              # TDD stage 0
├── test_stage_full_normalize.py             # TDD stage 1
└── test_stage_affine_transform.py           # TDD stage 2
```

## Git History

```
45888bc31f [ttnn-kernel-writer-tdd] stage affine_transform: passed
6869bea1ea [ttnn-kernel-writer-tdd] stage full_normalize: passed
956e7ddafd [ttnn-kernel-writer-tdd] stage mean_subtract: passed
b63087465e [ttnn-generic-op-builder] logs: layer_norm_rm execution log
6a0cecfd4d [ttnn-generic-op-builder] stubs: layer_norm_rm
23a395d480 [ttnn-operation-architect] finalize: layer_norm_rm breadcrumbs
58a90bdfa5 [ttnn-operation-architect] design: layer_norm_rm
a1f688203e [ttnn-operation-analyzer] analysis: batch_norm (compute_core)
a1f1bf9c31 [ttnn-operation-analyzer] analysis: untilize (output_stage)
```

## Decisions and Deviations

### Design Deviations (kernel writer found and fixed)
1. **SCALAR → COL broadcast for inv_std**: The design doc's Phase 7 specified SCALAR broadcast, which is incorrect — REDUCE_ROW output has per-row values in column 0, requiring COL broadcast to replicate each row's value across columns.
2. **Bulk → PerTile output for gamma/beta phases**: In-place CB reuse with Bulk output deadlocks for Wt > 1 because Bulk reserves all output pages while input pages still occupy the CB.

### Upstream Fixes (kernel writer modified Python)
1. **Gamma/beta padding before tilize**: Small gamma/beta tensors (volume < 1024) need `ttnn.pad` to meet `ttnn.tilize()` volume requirement.
2. **TensorAccessorArgs for gamma/beta**: Program descriptor needed gamma/beta TensorAccessorArgs in reader compile-time args with dynamic start indices.

## Infrastructure Issues

- **No device hangs during execution** — all hangs were caught by tt-test.sh timeout/triage
- **No build failures** — kernels compile at runtime without issues
- **No venv problems** — Python environment stable throughout
- **Device reset needed once** after stage 2 hang (tt-smi -r)

## Recommendations for Pipeline Improvement

1. **Design doc broadcast validation**: The architect should validate that REDUCE_ROW output requires COL (not SCALAR) broadcast for applying reduction results. This is a common and counterintuitive mistake.
2. **Bulk output policy warning**: The architect should flag Bulk output policy as dangerous when input and output share the same CB or when Wt > 1. PerTile is always safe.
3. **Small tensor tilize handling**: The builder should automatically handle padding for tensors smaller than TILE_HW before `ttnn.tilize()`.
4. **TensorAccessorArgs completeness**: The builder should ensure all tensor inputs that the reader kernel accesses have corresponding TensorAccessorArgs in compile-time args.
