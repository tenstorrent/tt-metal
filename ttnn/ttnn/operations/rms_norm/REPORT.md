# RMS Norm Operation — Pipeline Report

## Summary

**Operation**: `rms_norm`
**Math**: `RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma`
**Result**: ALL 4 TDD STAGES PASSED
**Total pipeline commits**: 10 (from analyzer through final TDD stage)

The operation supports both ROW_MAJOR_LAYOUT and TILE_LAYOUT inputs with in-kernel tilize/untilize conversion. Optional gamma scaling with RM gamma tilized in-kernel. Supported dtypes: bfloat16 and float32.

---

## Pipeline Execution

| Phase | Agent | Output | Status |
|-------|-------|--------|--------|
| 0 - Discovery | Orchestrator | 3 references identified | Complete |
| 1 - Analysis | ttnn-operation-analyzer (x3) | tilize, untilize, moreh_norm_w analyses | Complete |
| 2 - Design | ttnn-operation-architect | op_design.md, .tdd_state.json (4 stages) | Complete |
| 3 - Build | ttnn-generic-op-builder | Python infra + stub kernels + tests | Complete |
| 4 - TDD Kernels | ttnn-kernel-writer-tdd | All 4 stages passed | Complete |
| 5 - Report | Orchestrator | REPORT.md | Complete |

---

## Agent Summaries

### Phase 1: Analyzers (3 parallel)

**Tilize (input_stage)**:
- Analyzed `tilize_multi_core_interleaved_program_factory.cpp`
- Key findings: 32-stick batching pattern, TensorAccessor for DRAM reads, single-buffered CB with ntiles_per_block pages, `split_blocks_for_tilize()` work distribution
- Output: `agent_logs/tilize_analysis.md`

**Untilize (output_stage)**:
- Analyzed `untilize_multi_core_program_factory.cpp`
- Key findings: `untilize<block_width_tiles, input_cb, output_cb>(num_blocks)` is self-contained, handles all CB sync. Double-buffered output CB. Writer reads sticks and writes via TensorAccessor.
- Output: `agent_logs/untilize_analysis.md`

**Moreh Norm W (compute_core)**:
- Analyzed `moreh_norm_program_factory_w_other.cpp`
- Key findings: W-dimension reduction pattern with reduce_init/reduce_tile helpers, scaler CB setup, multi-tile row processing with intermediate CBs
- Output: `agent_logs/moreh_norm_w_analysis.md`

### Phase 2: Architect

- Designed 8-phase compute pipeline: tilize → square → reduce → add_eps+rsqrt → 2nd_tilize → mul_col → mul_row(gamma) → untilize
- CB layout: cb_x(0), cb_xsq(24), cb_reduce(25), cb_eps(9), cb_normed(26), cb_gamma_rm(1), cb_gamma(27), cb_scaler(8), cb_out(16)
- Registered 4 TDD stages with incremental complexity
- Output: `op_design.md`, `.tdd_state.json`

### Phase 3: Builder

- Created `rms_norm.py` (entry point with validation), `rms_norm_program_descriptor.py` (CB config, work distribution), `__init__.py`
- Created stub kernels: `rms_norm_reader.cpp`, `rms_norm_compute.cpp`, `rms_norm_writer.cpp`
- Generated integration test and 4 TDD stage test files
- Output: `agent_logs/ttnn-generic-op-builder_execution_log.md`

### Phase 4: TDD Kernel Writer

- Implemented all kernels incrementally across 4 stages
- Fixed multiple upstream issues in program descriptor and entry point
- Output: `agent_logs/ttnn-kernel-writer-tdd_breadcrumbs.jsonl`

---

## TDD Pipeline Results

| Stage | Name | Result | Hard Attempts | Free Retries | Failure Classifications |
|-------|------|--------|---------------|--------------|------------------------|
| 1 | data_pipeline | PASS | 0 | 0 | None |
| 2 | square_reduce_mean | PASS | 4 | 0 | TypeError (1), hang_unknown (1), hang_cb_deadlock (2) |
| 3 | rms_norm_no_gamma | PASS | 0 | 1 | compilation_error (1 - free retry) |
| 4 | rms_norm_with_gamma | PASS | 0 | 0 | None |

**Total attempts**: 4 hard + 1 free = 5 failures before all stages passed

### Stage 2 Failures (square_reduce_mean):
1. **TypeError** (attempt 1): Runtime args indexing issue in program descriptor
2. **Hang unknown** (attempt 2): CB deadlock from incorrect page counts
3. **CB deadlock** (attempt 3-4): cb_xsq needed Wt pages (not 1) because square writes all tiles before reduce starts consuming on same compute thread

### Stage 3 Failure (rms_norm_no_gamma):
1. **Compilation error** (free retry): `sfpu_init.h` include doesn't exist; `rsqrt.h` path corrected

---

## Design Deviations

1. **cb_normed routing for RM no-gamma**: Design specified `final_tile_cb = cb_x` with `untilize(cb_x, cb_out)`. Changed to `final_tile_cb = cb_normed` and `untilize(cb_normed, cb_out)` to avoid in-place deadlock where mul_col reads from cb_x AND writes to cb_x when cb_x is full.

2. **cb_normed page count**: Design specified 1 page for gamma (streaming). Changed to Wt pages because sequential compute phases on the same thread cannot stream 1-at-a-time without deadlock.

3. **compute_kernel_hw_startup for gamma path**: When HAS_GAMMA, the first compute operation is tilize(cb_gamma_rm → cb_gamma), so hw_startup uses those CBs instead of the input tilize/square CBs.

---

## Files Produced

### Operation (`ttnn/ttnn/operations/rms_norm/`)
```
├── __init__.py                          # Re-export rms_norm function
├── rms_norm.py                          # Entry point with validation
├── rms_norm_program_descriptor.py       # CB config, work distribution, kernel setup
├── kernels/
│   ├── rms_norm_reader.cpp              # RM stick reading, tilize staging, scaler/eps/gamma generation
│   ├── rms_norm_compute.cpp             # 8-phase: tilize→square→reduce→rsqrt→tilize→mul_col→mul_row→untilize
│   └── rms_norm_writer.cpp              # RM stick writing via TensorAccessor
├── op_design.md                         # Architecture + kernel design document
├── .tdd_state.json                      # TDD pipeline state (all 4 passed)
└── agent_logs/                          # Breadcrumbs and analysis files
```

### Tests (`tests/ttnn/unit_tests/operations/rms_norm/`)
```
├── __init__.py
├── test_rms_norm.py                     # Integration test
├── test_stage_data_pipeline.py          # Stage 1: identity passthrough
├── test_stage_square_reduce_mean.py     # Stage 2: square + reduce
├── test_stage_rms_norm_no_gamma.py      # Stage 3: full RMS norm (no gamma)
└── test_stage_rms_norm_with_gamma.py    # Stage 4: full RMS norm (with gamma)
```

---

## Git History

```
94b2d4c7af [ttnn-kernel-writer-tdd] stage rms_norm_with_gamma: passed (ALL STAGES COMPLETE)
099e46c256 [ttnn-kernel-writer-tdd] stage rms_norm_no_gamma: passed
867b9378e0 [ttnn-kernel-writer-tdd] stage square_reduce_mean: passed
ab26b592a2 [ttnn-kernel-writer-tdd] stage data_pipeline: passed
bc1663400a [ttnn-generic-op-builder] logs: rms_norm execution log and breadcrumbs
23721a008c [ttnn-generic-op-builder] stubs: rms_norm
5e1169e23e [ttnn-operation-architect] finalize: rms_norm breadcrumbs
4beb9cdd34 [ttnn-operation-architect] design: rms_norm
29f08886e8 [ttnn-operation-analyzer] update breadcrumbs for moreh_norm_w analysis
2c7bbb6014 [ttnn-operation-analyzer] analysis: tilize breadcrumb update
ae5aa505bd [ttnn-operation-analyzer] analysis: moreh_norm_w (add analysis file)
713d30134d [ttnn-operation-analyzer] analysis: moreh_norm_w
```

---

## Decisions Made

1. **Reference selection**: Chose tilize (input), untilize (output), moreh_norm_w (compute) as hybrid references
2. **Hybrid mode**: 3 references with different roles — tilize for RM→tile, untilize for tile→RM, moreh_norm_w for W-reduction compute pattern
3. **4 TDD stages**: Incremental complexity from identity passthrough → square+reduce → full norm → norm+gamma
4. **Layout parametrization**: All TDD stages test both ROW_MAJOR_LAYOUT and TILE_LAYOUT
5. **Epsilon delivery**: Passed as float32 bit-cast uint32 runtime arg, converted to bfloat16 in reader kernel

---

## Infrastructure Issues

- **Stage 2 hangs (2 occurrences)**: CB deadlocks due to incorrect page counts for sequential same-thread operations. Resolved by increasing cb_xsq to Wt pages.
- **Include path errors**: `sfpu_init.h` doesn't exist in the kernel API headers; `rsqrt.h` needed correct path
- No device access errors, no build failures, no venv problems

---

## Suggestions for Improving the Agent Pipeline

1. **CB page count reasoning**: The design doc should explicitly call out when sequential operations on the same compute thread require full-row buffering (not streaming). This caused 3 of 5 failures.
2. **Include path validation**: The architect should verify kernel API include paths exist before specifying them in the design doc.
3. **In-place CB deadlock detection**: The design doc's routing of `final_tile_cb = cb_x` for read+write on the same CB is a known deadlock pattern. The architect should flag this.
4. **TensorAccessor for multi-input ops**: When gamma needs DRAM access, the program descriptor must provide TensorAccessor CT args. The design should explicitly note this.
