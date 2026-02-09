# Planner Execution Log: row_standardize

## 1. Metadata

| Field | Value |
|-------|-------|
| **Operation** | row_standardize |
| **Agent** | ttnn-operation-planner |
| **Mode** | Hybrid (3 references) |
| **Status** | SUCCESS |
| **Output** | `row_standardize_spec.md` |
| **Predecessor** | ttnn-operation-analyzer |
| **Successor** | generic_op_builder + kernel-designer (parallel) |

## 2. Input Interpretation

| Field | Value | Source | Confidence |
|-------|-------|--------|------------|
| Operation name | row_standardize | User request | HIGH |
| Mode | Hybrid | 3 reference analyses provided | HIGH |
| Input stage ref | tilize_analysis.md | User request | HIGH |
| Compute core ref | softmax_analysis.md | User request | HIGH |
| Output stage ref | untilize_analysis.md | User request | HIGH |
| Mathematical formula | (x - mean) * rsqrt(var + eps) | User request | HIGH |
| Input layout | ROW_MAJOR | User request | HIGH |
| Output layout | ROW_MAJOR | User request | HIGH |
| Dtypes | bfloat16, float32 | User request | HIGH |
| Epsilon default | 1e-5 | User request | HIGH |
| Infrastructure | generic_op (Python-based) | User request | HIGH |

## 2a. Reference Analysis Extraction

| Reference | Role | Key Information Extracted |
|-----------|------|---------------------------|
| tilize_analysis.md | input_stage | RM stick reading via TensorAccessor (32 sticks per block), tilize_block in compute kernel, CB c_0/c_16 single-buffered at Wt pages, split_blocks_for_tilize distribution, FP32 dest acc support |
| softmax_analysis.md | compute_core | 6-stage row-wise compute pipeline (MAX, SUB_COL, EXP, SUM_reduce, RECIP, MUL_COL), WaitUpfrontNoPop for tile reuse, CBs c_24-c_28 for intermediates, reduce_helpers_compute.hpp library, generate_bcast_scaler for scaler tiles, FP32 dest acc via defines |
| untilize_analysis.md | output_stage | Compute untilize via untilize_block/pack_untilize_block, writer writes sticks row-by-row (32 per block), TensorAccessor for stick-level paging, split_blocks_for_tilize distribution |

### Component Mapping (Hybrid Mode)

| Component | Source Reference | Modifications Needed |
|-----------|-----------------|---------------------|
| Reader kernel | tilize | Adapted: add scaler/epsilon tile generation at startup |
| Compute (tilize phase) | tilize | None - use compute_kernel_lib::tilize<>() directly |
| Compute (standardize phases) | softmax | Adapted: SUM instead of MAX, square instead of exp, rsqrt instead of recip, add epsilon step |
| Compute (untilize phase) | untilize | None - use compute_kernel_lib::untilize<>() directly |
| Writer kernel | untilize | Adapted: simplified for interleaved-only, non-sharded output |

### Interface Compatibility (Hybrid Mode)

| Interface | From | To | Compatible? | Notes |
|-----------|------|-----|-------------|-------|
| Reader -> Compute (tilize) | reader.cb_rm_in (RM sticks) | tilize_block input | YES | Standard tilize pattern, 32 sticks -> Wt tiles |
| Compute (tilize) -> Compute (standardize) | cb_tilized (tiles) | reduce/sub/mul input | YES | All ops work on standard tiles |
| Compute (standardize) -> Compute (untilize) | cb_tilized_out (tiles) | untilize_block input | YES | Standard tile format |
| Compute (untilize) -> Writer | cb_rm_out (RM sticks) | writer output | YES | 32 sticks per block, standard untilize writer pattern |

### DeepWiki Queries

| Query | Findings | How Used |
|-------|----------|----------|
| generic_op infrastructure | ProgramDescriptor with KernelDescriptor, CBDescriptor. Output tensor last in io_tensors. | Confirmed design approach for generic_op builder |
| rsqrt_tile, add_tiles_bcast_scalar | Both available. add_tiles_bcast_scalar for scalar add, rsqrt_tile for reciprocal sqrt. Also found fused add_rsqrt_tile. | Used for epsilon addition + rsqrt design (Phase 6-7) |
| generate_reduce_scaler format | Expects packed bf16 (bf16 << 16 \| bf16). Float value converted to bfloat16 and packed. | Used for 1/W scaler design in reader kernel |

### Design Decisions

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| Kernel architecture | Separate tilize/compute/untilize ops vs single fused kernel | Single compute kernel with tilize+standardize+untilize | Avoids 2 extra DRAM round-trips for intermediate tile data |
| Core count | Single-core vs multi-core | Single-core prototype | Simplifies implementation; each tile-row is independent, multi-core extension straightforward |
| Processing granularity | Per-tile vs per-tile-row | Per-tile-row (Wt tiles) | Row reductions require full W dimension to be resident |
| Reduce scaler | 1.0 + post-divide vs 1/W | 1/W as reduce scaler | Computes mean directly in reduce, avoids extra multiply |
| Epsilon handling | Compile-time vs runtime | Runtime arg, scalar broadcast tile | Epsilon varies per call, compile-time would defeat caching |
| Intermediate data format | Always bf16 vs fp32 for intermediates | fp32 when input is fp32, else match input | Standard precision pattern from softmax reference |
| CB count | Minimize (reuse CBs) vs clarity | 11 distinct CBs | Clarity for implementation; compute_kernel_lib helpers manage CB sync |

## 3. Execution Timeline

| Step | Action | Duration | Notes |
|------|--------|----------|-------|
| 1 | Read tilize_analysis.md | - | Full read, extracted input stage patterns |
| 2 | Read softmax_analysis.md | - | Full read, extracted compute pipeline patterns |
| 3 | Read untilize_analysis.md | - | Full read, extracted output stage patterns |
| 4 | Read table-templates.md | - | Template formats for spec tables |
| 5 | Read kernel helper libraries | - | reduce_helpers, binary_op_helpers, scalar_helpers, tilize_helpers, untilize_helpers |
| 6 | DeepWiki queries (3x) | - | generic_op, rsqrt/add_bcast, reduce_scaler format |
| 7 | Write row_standardize_spec.md | - | Full spec with all sections |
| 8 | Write execution log | - | This file |
| 9 | Git commit | - | Spec + logs |

## 4. Recovery Summary

No errors or recovery needed. Specification was produced in a single pass.

## 5. Deviations

None. All instructions followed as specified.

## 6. Artifacts

| File | Type | Description |
|------|------|-------------|
| `row_standardize_spec.md` | Specification | Full functional specification for row_standardize operation |
| `planner_breadcrumbs.jsonl` | Breadcrumbs | Event log with 14 entries |
| `planner_execution_log.md` | Execution log | This file |

## 7. Handoff Notes

### For generic_op_builder
- This operation uses generic_op infrastructure (Python-based, no C++ scaffolding)
- 11 circular buffers defined with specific IDs (c_0, c_1, c_2, c_3, c_4, c_16, c_24-c_28)
- CB page sizes MUST be dtype-aware (2048 for bf16, 4096 for f32)
- Single-core grid (1x1)
- Output tensor = same shape/dtype/layout as input, allocated on device

### For kernel-designer
- Single compute kernel performs tilize + 6-phase standardize + untilize
- Uses kernel library helpers: tilize<>, untilize<>, reduce<>, sub<COL>, mul<COL>, square<>
- Reader generates 2 persistent scalar tiles (scaler, epsilon)
- Writer uses untilize-style stick writing pattern
- FP32_DEST_ACC_EN define needed for float32 inputs

### For kernel-writer
- Reader pattern: tilize reference reader + scalar tile generation
- Compute pattern: tilize + softmax-adapted pipeline + untilize (all in one kernel)
- Writer pattern: untilize reference writer (simplified for interleaved only)
- All runtime args documented with exact indices in spec

## 8. Instruction Recommendations

- The spec assumes W fits within L1 memory budget. For very large W (>1568 for bf16, >768 for f32), a "large-W" variant would be needed. This could be added as a future enhancement.
- The single-core design is adequate for prototype but production use would benefit from multi-core distribution using the split_blocks_for_tilize pattern from the tilize/untilize references.
