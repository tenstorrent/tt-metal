# layer_norm_rm — Pipeline Report

## Summary

**Operation**: `layer_norm_rm`
**Purpose**: Row-major layer normalization with optional affine transform (gamma/beta)
**Result**: **ALL STAGES PASSED** — 4/4 TDD stages, 0 hard retries total
**Mode**: Fully Automated (no human intervention)
**Pipeline Mode**: Hybrid (tilize input + batch_norm compute + untilize output)

The operation takes a bfloat16 row-major interleaved tensor, normalizes each row (mean subtraction → variance → inv_sqrt → standardize), and optionally applies gamma (scale) and beta (shift) affine transforms. Input/output are row-major; compute is done in tile format via tilize/untilize in the compute kernel.

---

## Pipeline Execution

| Phase | Agent | Description | Commits | Key Output |
|-------|-------|-------------|---------|------------|
| 0 | Orchestrator | Discovery — identified 3 reference operations | — | Reference table |
| 1 | ttnn-operation-analyzer (×3) | Parallel analysis of tilize, untilize, batch_norm | `85e7b63`, `21b3c5a`, `a371bdb` | 3 analysis .md files |
| 2 | ttnn-operation-architect | Design document + TDD stage registration | `eb66a41`, `9130e4c` | op_design.md, .tdd_state.json |
| 3 | ttnn-generic-op-builder | Python infra + stub kernels + tests | `8b1db44`, `590d26d` | 3 .py files, 3 stub .cpp files, 5 test files |
| 4 | ttnn-kernel-writer-tdd | All 4 TDD stages implemented | `4e814aa`→`0ac56c7` | 3 implemented kernels |
| 5 | Orchestrator | Report generation | (this commit) | REPORT.md |

---

## Agent Summaries

### Phase 1: Analyzers (3 parallel)

**Tilize (input_stage)** — Analyzed `tilize_multi_core_interleaved_program_factory.cpp`:
- Reader reads RM sticks from DRAM, packs 32 sticks into tile-sized CB pages
- Work distribution: rows distributed across cores
- Key insight: input CB must use tile_page_size (not stick_size) for tilize helper

**Untilize (output_stage)** — Analyzed `untilize_multi_core_program_factory.cpp`:
- Writer extracts RM sticks from tiles and writes to DRAM
- Untilize helper converts tile format back to row-major
- Key insight: output CB page_size matches stick_size for DRAM writes

**Batch Norm (compute_core)** — Analyzed `batch_norm_program_factory.cpp`:
- Normalization compute pattern with gamma/beta handling
- Reduce operations for mean/variance computation
- Scalar CB setup for epsilon and reduce scalers
- Key insight: multi-pass CB reuse patterns for intermediate results

### Phase 2: Architect

Produced comprehensive op_design.md covering:
- **CB Layout**: 11 circular buffers (input RM, tilized tiles, intermediates for mean/variance/centered, scalar CBs for epsilon and reduce scaler, gamma/beta, output)
- **Work Distribution**: Rows distributed across cores; each core processes nblocks of 32-row tile-rows
- **Compute Phases**: 7 phases (tilize → reduce_mean → subtract_mean → square_centered → reduce_variance → inv_sqrt_normalize → untilize), plus optional affine phase
- **TDD Stages**: 4 incremental stages registered (data_pipeline → reduce_mean → variance_normalize → affine_transform)

### Phase 3: Builder

Created:
- `layer_norm_rm.py` — Entry point with validation (dtype, layout, gamma/beta shape checks)
- `layer_norm_rm_program_descriptor.py` — Program descriptor with CB config, kernel setup, runtime args
- `__init__.py` — Re-export
- 3 stub kernels (reader/compute/writer)
- Integration test + 4 TDD stage tests

### Phase 4: TDD Kernel Writer

Single agent session implementing all 4 stages sequentially:

| Stage | Name | Result | Attempts | Upstream Fixes |
|-------|------|--------|----------|----------------|
| 0 | data_pipeline | PASS | 0 hard | cb_in_rm page_size → tile_size; nblocks moved to RT arg; epsilon fill moved to reader |
| 1 | reduce_mean | PASS | 0 hard | None |
| 2 | variance_normalize | PASS | 0 hard | cb_var_input total_size expanded from 1→Wt tiles |
| 3 | affine_transform | PASS | 0 hard | None |

---

## Upstream Fixes During TDD

The kernel writer identified and fixed 4 issues in the program descriptor:

1. **cb_in_rm page_size**: Changed from `stick_size` to `tile_page_size`. The `tilize_init`/`tilize_block` helpers read face/tile dimensions from the CB's metadata — stick-sized pages give wrong tile dimensions.

2. **nblocks_per_core**: Moved from compile-time define to runtime arg for compute kernel. Cliff cores (last core handling remainder rows) get different nblocks than other cores, so this must be a runtime value.

3. **Epsilon fill location**: Moved from compute kernel to reader kernel. Compute kernels lack `get_write_ptr()` access — only data movement kernels can write to L1/CB memory directly.

4. **cb_var_input sizing**: Expanded from 1 tile to Wt tiles. The square operation produces all Wt tiles before the reduce pass consumes them, so the intermediate CB must hold a full tile-row.

---

## Files Produced

### Operation Code (`ttnn/ttnn/operations/layer_norm_rm/`)
```
__init__.py                             # Re-export
layer_norm_rm.py                        # Entry point with validation
layer_norm_rm_program_descriptor.py     # CB config, work distribution, kernel setup
kernels/layer_norm_rm_reader.cpp        # Reader: RM sticks + epsilon + gamma/beta
kernels/layer_norm_rm_compute.cpp       # Compute: tilize → LN math → untilize
kernels/layer_norm_rm_writer.cpp        # Writer: RM sticks to DRAM
op_design.md                            # Architecture + kernel design
.tdd_state.json                         # TDD pipeline state (all passed)
```

### Tests (`tests/ttnn/unit_tests/operations/layer_norm_rm/`)
```
test_layer_norm_rm.py                   # Integration test
test_stage_data_pipeline.py             # TDD stage 0
test_stage_reduce_mean.py               # TDD stage 1
test_stage_variance_normalize.py        # TDD stage 2
test_stage_affine_transform.py          # TDD stage 3
```

### Agent Logs (`ttnn/ttnn/operations/layer_norm_rm/agent_logs/`)
```
phase0_discovery.md                     # Discovery decisions
tilize_analysis.md                      # Tilize reference analysis
untilize_analysis.md                    # Untilize reference analysis
batch_norm_analysis.md                  # Batch norm reference analysis
ttnn-operation-analyzer_breadcrumbs.jsonl
ttnn-operation-architect_breadcrumbs.jsonl
ttnn-operation-architect_execution_log.md
ttnn-generic-op-builder_breadcrumbs.jsonl
ttnn-kernel-writer-tdd_breadcrumbs.jsonl
```

---

## Git History

```
35899cd [ttnn-kernel-writer-tdd] final breadcrumbs for layer_norm_rm pipeline completion
0ac56c7 [ttnn-kernel-writer-tdd] stage affine_transform: passed — all 4 stages complete
c140446 [ttnn-kernel-writer-tdd] stage variance_normalize: passed
79a0b07 [ttnn-kernel-writer-tdd] stage reduce_mean: passed
4e814aa [ttnn-kernel-writer-tdd] stage data_pipeline: passed
590d26d [ttnn-generic-op-builder] logs: finalize breadcrumbs for layer_norm_rm
8b1db44 [ttnn-generic-op-builder] stubs: layer_norm_rm
9130e4c [ttnn-operation-architect] logs: final breadcrumb for layer_norm_rm
eb66a41 [ttnn-operation-architect] design: layer_norm_rm
a371bdb [ttnn-operation-analyzer] analysis: batch_norm (compute_core reference)
21b3c5a [ttnn-operation-analyzer] analysis: untilize (output_stage reference)
85e7b63 [ttnn-operation-analyzer] analysis: tilize (input_stage reference)
```

---

## Decisions Made

1. **Reference selection**: Chose batch_norm as compute reference over softmax — batch_norm has closer gamma/beta affine handling and similar reduce-based normalization pattern.
2. **Single-core first**: Design targets multi-core interleaved with row-based work distribution (matching tilize reference pattern).
3. **In-compute tilize/untilize**: Rather than separate tilize/untilize programs, the compute kernel handles tilize→math→untilize internally. Reader/writer deal only with RM sticks.
4. **Epsilon via reader**: Epsilon scalar is written to its CB by the reader kernel (not compute), since compute kernels lack direct L1 write access.
5. **Tolerances**: Relaxed tolerances for later stages (rtol=0.05, atol=0.2) due to bfloat16 precision loss through multiple reduction and sqrt operations.

## Deviations from Spec

None significant. All upstream fixes were to the program descriptor configuration, not architectural deviations from op_design.md.

## Pain Points

- **Analyzer output size**: Even with role-based focus, analyses are 20-33KB each. Could be trimmed further for pipeline use.
- **cb_in_rm page_size**: This is a known recurring issue (documented in MEMORY.md). The architect's design correctly specified tile_page_size, but the builder stub used stick_size. The kernel writer caught and fixed it.

## Infrastructure Issues

- No device hangs encountered
- No build failures
- No venv problems
- All tests ran cleanly via `scripts/tt-test.sh --dev`

---

## Suggestions for Pipeline Improvement

1. **Builder should respect design's CB page_size**: The generic-op-builder should read the architect's CB specifications more carefully, especially page_size for tilize input CBs.
2. **Epsilon handling convention**: Establish a standard pattern for scalar constants (reader fills CB vs. compile-time define) to avoid the "compute can't write to L1" discovery each time.
3. **CB sizing validation**: The builder could validate that intermediate CB sizes match the data flow (e.g., if square produces Wt tiles, the CB must hold Wt tiles).
4. **Analyzer trimming**: Consider structured output format (JSON) instead of free-form markdown to reduce context consumption in downstream agents.
