# Row Centralize Operation - Build Report

## Summary

Successfully built a row-wise standardization TTNN operation (`row_centralize`) using the Generic Op workflow with TDD kernel pipeline. The operation takes a row-major interleaved bfloat16 tensor and computes `y = (x - mean(x)) * rsqrt(var(x) + epsilon)` along the last dimension.

**Result**: All 3 TDD stages passed on first attempt, zero retries needed.

---

## Pipeline Execution

| Phase | Agent | Duration | Output |
|-------|-------|----------|--------|
| Phase 0 | Orchestrator | ~2 min | Reference selection (tilize, untilize, batch_norm) |
| Phase 1 | 3x ttnn-operation-analyzer (parallel) | ~7 min | 3 analysis files |
| Phase 2 | ttnn-operation-planner | ~5.5 min | `row_centralize_spec.md` |
| Phase 3a | ttnn-generic-op-builder | ~5.5 min | Python infra + stub kernels |
| Phase 3b | ttnn-kernel-designer (parallel) | ~3.7 min | `kernel_design.md` |
| Phase 4 | 3x ttnn-kernel-writer (sequential TDD) | ~14 min | 3 kernel implementations |
| **Total** | | **~30 min** | |

---

## Agent Summaries

### ttnn-operation-analyzer (Phase 1)

Three analyzers ran in parallel on reference operations:

1. **Tilize (input_stage)**: Analyzed `tilize_multi_core_interleaved_program_factory.cpp`. Key findings: RM stick reading pattern, `split_blocks_for_tilize` work distribution, tilize block API usage.

2. **Untilize (output_stage)**: Analyzed `untilize_multi_core_program_factory.cpp`. Key findings: Stick extraction pattern from untilized tile-rows (32 sticks per tile-row), TensorAccessor-based DRAM write addressing.

3. **Batch Norm (compute_core)**: Analyzed `batch_norm_program_factory.cpp`. Key findings: `binary_dest_reuse_tiles` pattern for chaining ops, `rsqrt_tile` usage, epsilon CB pattern, `FILL_TILE_WITH_FIRST_ELEMENT` broadcast.

### ttnn-operation-planner (Phase 2)

Produced comprehensive spec with:
- 12 circular buffers (c_0 through c_25)
- 9 compute phases per tile-row
- Single-core execution strategy
- Detailed compile-time and runtime argument layouts
- 7 design decisions documented with rationale

**Key decisions**:
- Single-core for simplicity (multi-core as future optimization)
- Block-at-a-time processing (one tile-row per iteration)
- Two-pass CB c_3 pattern (centered tiles used for both squaring and final multiply)
- All compute in one kernel (tilize + math + untilize)

### ttnn-generic-op-builder (Phase 3a)

Created Python infrastructure:
- `row_centralize.py`: Entry point with validation and output allocation
- `row_centralize_program_descriptor.py`: CB config, kernel setup, bf16 packing
- `__init__.py`: Module exports
- `test_row_centralize.py`: PyTorch reference comparison test
- 3 stub kernel files (reader, compute, writer)

### ttnn-kernel-designer (Phase 3b)

Produced kernel design document mapping all 9 compute phases to helper functions or raw calls:
- Phases 1-6, 8-9: **USE HELPER** (tilize, reduce, sub, square, add, mul, untilize)
- Phase 7 (rsqrt): **NO HELPER** - raw implementation required (copy_tile + rsqrt_tile + pack_tile)

**Key design contribution**: Identified `copy_tile_to_dst_init_short(cbid)` requirement before rsqrt, and confirmed two-pass CB patterns (c_1 and c_3) with correct input policies.

### ttnn-kernel-writer (Phase 4)

Three stage-gated invocations:

1. **tilize_untilize**: Implemented all 3 kernel files from scratch (reader with scaler/epsilon generation, compute with tilize+untilize, writer with stick extraction)
2. **centralize**: Added Phases 2+3 to compute kernel, changed untilize input from c_1 to c_3
3. **full_standardize**: Added Phases 4-8 to compute kernel, changed untilize input from c_3 to c_6

**Deviation from design**: `copy_tile_to_dst_init_short()` requires `cbid` as first argument (design omitted it). Fixed to `copy_tile_to_dst_init_short(cb_var_plus_eps)`.

---

## TDD Pipeline Results

| Stage | Name | Description | Attempts | Result | Commit |
|-------|------|-------------|----------|--------|--------|
| 0 | tilize_untilize | Reader + tilize + untilize + writer identity roundtrip | 1/3 | PASS | `475754f4` |
| 1 | centralize | + reduce mean + subtract (x - mean) | 1/3 | PASS | `5cb1b41e` |
| 2 | full_standardize | + square + variance + epsilon + rsqrt + multiply | 1/3 | PASS | `17b5a9be` |

**Test shapes verified**:
- Stage 0: `(1,1,32,64)`, `(1,1,64,128)` - 2/2 passed
- Stage 1: `(1,1,32,64)`, `(1,1,64,128)` - 2/2 passed
- Stage 2: `(1,1,32,64)`, `(1,1,64,128)`, `(1,1,32,32)` - 3/3 passed

**Failure classifications encountered**: None. All stages passed on first attempt.

**Tolerances used**:
- Stage 0 (identity): rtol=0.01, atol=0.01
- Stage 1 (centralize): rtol=0.02, atol=0.1
- Stage 2 (full standardize): rtol=0.05, atol=0.2

---

## Breadcrumb/Log Files

| Agent | Breadcrumbs | Execution Log |
|-------|-------------|---------------|
| ttnn-operation-analyzer | `agent_logs/ttnn-operation-analyzer_breadcrumbs.jsonl` | (embedded in analysis files) |
| ttnn-operation-planner | `agent_logs/ttnn-operation-planner_breadcrumbs.jsonl` | - |
| ttnn-generic-op-builder | `agent_logs/ttnn-generic-op-builder_breadcrumbs.jsonl` | - |
| ttnn-kernel-designer | `agent_logs/ttnn-kernel-designer_breadcrumbs.jsonl` | `agent_logs/ttnn-kernel-designer_execution_log.md` |
| ttnn-kernel-writer | `agent_logs/ttnn-kernel-writer_breadcrumbs.jsonl` | `agent_logs/ttnn-kernel-writer_execution_log.md` |

All files located under `ttnn/ttnn/operations/row_centralize/agent_logs/`.

---

## Files Produced

```
ttnn/ttnn/operations/row_centralize/
├── __init__.py                              # Module exports
├── row_centralize.py                        # Entry point + validation
├── row_centralize_program_descriptor.py     # CB config, kernel setup, bf16 packing
├── row_centralize_spec.md                   # Functional specification
├── kernel_design.md                         # Kernel implementation design
├── test_row_centralize.py                   # E2E PyTorch comparison test
├── test_stage_tilize_untilize.py            # TDD stage 0 test
├── test_stage_centralize.py                 # TDD stage 1 test
├── test_stage_full_standardize.py           # TDD stage 2 test
├── .tdd_state.json                          # TDD pipeline state
├── kernels/
│   ├── row_centralize_reader.cpp            # Reader: scaler/eps gen + RM stick reads
│   ├── row_centralize_compute.cpp           # 9-phase compute pipeline
│   └── row_centralize_writer.cpp            # Writer: RM stick writes
└── agent_logs/
    ├── tilize_analysis.md
    ├── untilize_analysis.md
    ├── batch_norm_analysis.md
    ├── ttnn-operation-analyzer_breadcrumbs.jsonl
    ├── ttnn-operation-planner_breadcrumbs.jsonl
    ├── ttnn-generic-op-builder_breadcrumbs.jsonl
    ├── ttnn-kernel-designer_breadcrumbs.jsonl
    ├── ttnn-kernel-designer_execution_log.md
    ├── ttnn-kernel-writer_breadcrumbs.jsonl
    └── ttnn-kernel-writer_execution_log.md
```

---

## Git History

```
ad6a15a [kw-tdd] row_centralize: all 3 TDD stages passed
17b5a9b [ttnn-kernel-writer] full_standardize: add Phases 4-8
5cb1b41 [ttnn-kernel-writer] centralize stage: add reduce_row mean + sub_col
475754f [ttnn-kernel-writer] tilize_untilize: implement reader/compute/writer
ea027ff [ttnn-kernel-writer] tilize_untilize: implement reader/compute/writer
0070838 [ttnn-generic-op-builder] Add row_centralize generic_op infrastructure
9045c34 [ttnn-kernel-designer] logs: row_centralize execution log
c7cd793 [ttnn-kernel-designer] design: row_centralize
0b77588 [ttnn-operation-planner] spec: row_centralize
9a62ca3 [ttnn-operation-analyzer] Tilize analysis for row_centralize
```

---

## Decisions and Deviations

### Assumptions Made (Fully Automated Mode)
1. **Operation name**: `row_centralize` (matches the task title)
2. **Reference selection**: Tilize (input), untilize (output), batch_norm (compute) - no layernorm reference used since batch_norm had the relevant normalization pattern
3. **Single-core only**: Simplest correct implementation; multi-core is a future optimization
4. **Dims must be multiples of 32**: No padding support in v1
5. **Stage grouping**: Combined data_pipeline + bookends into stage 1 (tilize_untilize) since RM I/O requires tilize/untilize anyway

### Deviations from Spec
1. **`copy_tile_to_dst_init_short` argument**: Design document omitted the `cbid` argument. Kernel writer correctly added `cb_var_plus_eps` as the argument.

### Pain Points
- None significant. The helper library APIs worked exactly as documented. The TDD pipeline caught no regressions - all stages passed on first attempt.
- The only minor friction was ensuring the untilize input CB changed correctly across stages (c_1 -> c_3 -> c_6 as each stage added more compute phases).
