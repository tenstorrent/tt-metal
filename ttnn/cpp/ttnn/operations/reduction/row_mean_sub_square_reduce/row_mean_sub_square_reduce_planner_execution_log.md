# Execution Log: Row Mean Sub Square Reduce Planning

## Session Info
- **Started**: Session start
- **Planning Mode**: Hybrid
- **New Operation**: row_mean_sub_square_reduce (alias: variance_w)
- **Reference Analyses**:
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_analysis.md` (role: input_stage)
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_multi_core_w_analysis.md` (role: compute_core)
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_analysis.md` (role: output_stage)

## Execution Timeline

### Step 1: Read Reference Analysis Documents
**Action**: Read all three reference analysis files to extract component information
**Tool**: Read tool for each file
**Results**:

#### Tilize Analysis (input_stage)
- Work unit: Block (row of tiles), ntiles_per_block = padded_W / TILE_WIDTH
- CB configuration: c_0 for row-major input, c_16 for tiled output
- Reader pattern: TensorAccessor for stick reading, 32 sticks per block
- Compute: tilize_block helper from tilize_helpers.hpp
- Key insight: Single-buffered CBs for simplicity

#### Reduce W Analysis (compute_core)
- Work unit: Row of tiles (Wt tiles -> 1 tile)
- Scaler generation: generate_reduce_scaler with packed bfloat16 (1/W for mean)
- CB configuration: c_0 input (double-buffered), c_2 scaler, c_3 output
- Compute: REDUCE_ROW_SUM_VIA_MM optimization for matmul-based reduction
- Key insight: Scaler persists for entire program, generated once

#### Untilize Analysis (output_stage)
- Work unit: Block (row of tiles -> TILE_HEIGHT rows of elements)
- CB configuration: c_0 for tiled input, c_16 for row-major output
- Writer pattern: Row-by-row stick writing with TensorAccessor
- Compute: untilize_block or pack_untilize helper
- Key insight: Output page = single row of elements

**Decision**: All three references use compatible data formats at interfaces (tiled between compute stages).

### Step 2: Read Table Templates and Logging Reference
**Action**: Read table-templates.md and agent-execution-logging.md for output format
**Tool**: Read tool
**Result**: Obtained standard table formats for CB configuration, kernel specs, and argument tables. Confirmed execution logging is enabled.

### Step 3: Query DeepWiki for Variance Computation Patterns
**Action**: Query DeepWiki for reduce_tile, variance computation, and helper functions
**Tool**: mcp__deepwiki__ask_question

#### Query 1: Variance computation with reduce_tile
**Question**: "How does reduce_tile work for variance computation along W dimension?"
**Key Findings**:
- reduce_init with REDUCE_OP::SUM, REDUCE_DIM::REDUCE_ROW for W reduction
- Scaler CB contains 1/W for averaging
- Variance pattern: compute mean -> subtract -> square -> reduce again
- Reference: layernorm and groupnorm kernels use similar patterns

#### Query 2: generate_reduce_scaler format
**Question**: "How does generate_reduce_scaler work?"
**Key Findings**:
- Takes CB ID and packed_scalar (uint32 with two bfloat16 values)
- Fills 512 uint32 entries in tile (1024 bfloat16 elements)
- Same bfloat16 value duplicated in both halves of uint32
- Uses pack_two_bfloat16_into_uint32 for packing

#### Query 3: sub_tiles for subtraction
**Question**: "What is sub_tiles function and how to subtract tiles?"
**Key Findings**:
- sub_tiles_bcast_scalar for scalar broadcast subtraction
- Requires sub_tiles_bcast_scalar_init_short initialization
- Scalar in tile position [0,0], result is C[h,w] = A[h,w] - B[0,0]
- Perfect for subtracting mean from input tiles

#### Query 4: mul_tiles/square_tile for squaring
**Question**: "How does mul_tiles/square_tile work?"
**Key Findings**:
- square_tile is dedicated SFPU operation for element-wise squaring
- Works on tile already in DST register
- Requires square_tile_init() before use
- More efficient than mul_tiles(x, x) for squaring

#### Query 5: tilize/untilize helpers
**Question**: "How do tilize and untilize compute helpers work?"
**Key Findings**:
- tilize_init(icb, block, ocb), tilize_block(icb, block, ocb), tilize_uninit
- untilize_init(icb), untilize_block(icb, block, ocb), untilize_uninit
- pack_untilize variant for hardware-accelerated path
- Includes headers: compute_kernel_api/tilize.h, untilize.h

### Step 4: Design Component Mapping
**Action**: Map components from each reference to the new operation
**Decision**:

| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader kernel | tilize | input_stage | Add scaler generation |
| CB_rm_in (c_0) | tilize | input_stage | None |
| Tilize phase | tilize | input_stage | Feed to reduce instead of writer |
| CB_scaler (c_2) | reduce_w | compute_core | None |
| Reduce (mean) | reduce_w | compute_core | First reduction step |
| Sub/Square | NEW | compute_core | Custom compute logic |
| Reduce (variance) | reduce_w | compute_core | Second reduction step |
| Untilize phase | untilize | output_stage | Input is 1 tile (variance) |
| CB_rm_out (c_16) | untilize | output_stage | None |
| Writer kernel | untilize | output_stage | Output sticks are TILE_WIDTH |

### Step 5: Define CB Configuration
**Action**: Determine CB IDs, sizes, and purposes for variance pipeline
**Decision**: 7 CBs needed

| CB ID | Name | Purpose | Sizing |
|-------|------|---------|--------|
| c_0 | CB_rm_in | Row-major input | Wt tiles |
| c_1 | CB_tilized | Tilized input | Wt tiles |
| c_2 | CB_scaler | Scaler (1/W) | 1 tile |
| c_3 | CB_mean | Mean for broadcast | 1 tile |
| c_4 | CB_intermediate | Squared differences | Wt tiles |
| c_5 | CB_out_tiled | Variance tile | 1 tile |
| c_16 | CB_rm_out | Row-major output | 2 tiles (double-buffer) |

**Rationale**:
- Need full row (Wt tiles) for tilized input and intermediate to compute row-wise operations
- Mean and variance are single tiles (reduced)
- Double-buffer output for compute/write overlap
- Scaler persists for program lifetime

### Step 6: Design Work Distribution
**Action**: Determine parallelization strategy
**Decision**: Parallelize over Ht (tile rows)

**Rationale**:
- Each tile row is independent (variance along W, not across rows)
- Work unit = one tile row = Wt input tiles -> 1 output tile
- Load balancing: two-group split (same as reduce_w)
- Grid: min(NC * Ht, max_cores)

### Step 7: Design Data Flow
**Action**: Determine kernel responsibilities and CB flow
**Decision**:

1. **Reader** (BRISC):
   - Generate scaler once at start
   - Read Wt * TILE_HEIGHT sticks per tile-row
   - Push to CB_rm_in

2. **Compute** (TRISC):
   - Tilize from CB_rm_in to CB_tilized
   - Reduce CB_tilized to CB_mean (mean)
   - For each tile: sub_bcast_scalar, square, push to CB_intermediate
   - Reduce CB_intermediate to CB_out_tiled (variance)
   - Untilize from CB_out_tiled to CB_rm_out

3. **Writer** (NCRISC):
   - Read from CB_rm_out
   - Write TILE_HEIGHT sticks per output tile

### Step 8: Write Functional Specification
**Action**: Create comprehensive spec document
**Tool**: Write tool
**Result**: Created `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/row_mean_sub_square_reduce/row_mean_sub_square_reduce_spec.md`

## Reference Analysis Extraction

| Reference | Role | Section | Key Information Extracted |
|-----------|------|---------|---------------------------|
| tilize | input_stage | Work Unit | Block = ntiles_per_block tiles (one tile row) |
| tilize | input_stage | Data Flow | Reader reads sticks, compute tilizes |
| tilize | input_stage | CB Config | c_0 for input (ntiles_per_block), c_16 for output |
| tilize | input_stage | Reader Pattern | TensorAccessor, 32 sticks per block |
| reduce_w | compute_core | Work Unit | Row of Wt tiles -> 1 output tile |
| reduce_w | compute_core | Scaler | generate_reduce_scaler with packed bfloat16 |
| reduce_w | compute_core | CB Config | c_0 input, c_2 scaler, c_3 output |
| reduce_w | compute_core | Compute | REDUCE_ROW_SUM_VIA_MM or reduce_tile |
| untilize | output_stage | Work Unit | Block -> TILE_HEIGHT output rows |
| untilize | output_stage | CB Config | c_0 tiled input, c_16 row-major output |
| untilize | output_stage | Writer Pattern | Row-by-row writes with TensorAccessor |

## Component Mapping (Hybrid Mode)

| Component | Source Reference | Extraction Notes |
|-----------|-----------------|------------------|
| Reader kernel | tilize | Use TensorAccessor pattern for stick reading |
| Scaler generation | reduce_w | Add generate_reduce_scaler call to reader |
| CB_rm_in | tilize (c_0) | Same pattern, sized for Wt tiles |
| Tilize compute | tilize | Use tilize_block helper |
| Reduce (mean) | reduce_w | First REDUCE_ROW with scaler 1/W |
| Sub/Square | NEW | sub_tiles_bcast_scalar + square_tile |
| CB_mean | NEW | Hold mean for broadcast subtraction |
| CB_intermediate | NEW | Hold squared differences |
| Reduce (variance) | reduce_w | Second REDUCE_ROW with same scaler |
| Untilize compute | untilize | Use untilize_block helper |
| CB_rm_out | untilize (c_16) | Same pattern, double-buffered |
| Writer kernel | untilize | Row-by-row stick writing |

## Interface Analysis (Hybrid Mode)

| Interface | Status | Notes |
|-----------|--------|-------|
| Reader->Compute(tilize) | Compatible | Row-major sticks in CB |
| Tilize->Reduce(mean) | Compatible | Tiled format |
| Mean->Sub | Compatible | Single tile, scalar broadcast |
| Sub->Square | Compatible | Same tile in DST |
| Square->Reduce(var) | Compatible | Tiled format |
| Var->Untilize | Compatible | Single tile |
| Untilize->Writer | Compatible | Row-major sticks |

## Files Read

| File | Purpose | Key Findings |
|------|---------|--------------|
| tilize_multi_core_interleaved_analysis.md | Input stage reference | TensorAccessor for sticks, tilize_block helper |
| reduce_multi_core_w_analysis.md | Compute core reference | generate_reduce_scaler, REDUCE_ROW pattern |
| untilize_multi_core_analysis.md | Output stage reference | Row-by-row writer, untilize_block helper |
| table-templates.md | Output formatting | Standard table formats for spec |
| agent-execution-logging.md | Logging requirements | Breadcrumb format, git commit requirements |

## DeepWiki Queries

| Query | Response Summary | How Used |
|-------|------------------|----------|
| reduce_tile for variance | reduce_init/reduce_tile with REDUCE_ROW, scaler 1/W | Defined compute phases 2 and 4 |
| generate_reduce_scaler | Packed bfloat16 in uint32, fills 512 entries | Specified reader scaler generation |
| sub_tiles | sub_tiles_bcast_scalar for scalar subtraction | Specified compute phase 3 (subtract mean) |
| mul_tiles/square_tile | square_tile SFPU primitive | Specified compute phase 3 (square) |
| tilize/untilize helpers | tilize_init/block/uninit, untilize_init/block/uninit | Specified compute phases 1 and 5 |

## Design Decisions Made

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| Single vs multi-pass | Single-pass, two-pass | Single-pass | Avoids re-reading input from DRAM |
| Work distribution | Tile, row, batch | Row (Ht) | Natural unit for W reduction |
| CB sizing | Single-buffer, double-buffer | Mixed | Wt for input/intermediate, double for output |
| Compute architecture | Separate kernels, fused | Fused | Maximizes data locality |
| Scaler strategy | Separate, shared | Shared 1/W | Same scaler for both mean computations |
| Mean broadcast | Manual copy, bcast_scalar | bcast_scalar | Hardware-efficient scalar broadcast |

## Errors/Issues Encountered

| Issue | Context | Resolution |
|-------|---------|------------|
| None | - | - |

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/row_mean_sub_square_reduce/` | Created (directory) | New operation directory |
| `row_mean_sub_square_reduce_spec.md` | Created | Functional specification |
| `row_mean_sub_square_reduce_planner_execution_log.md` | Created | This execution log |

## Final Status
- **Completed**: Yes
- **Output File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/row_mean_sub_square_reduce/row_mean_sub_square_reduce_spec.md`
- **Open Questions**:
  1. Double-buffer tilized input? (currently: no, re-tilize to save L1)
  2. FP32 accumulation for reduce? (recommendation: enable for precision)
  3. Output padding handling? (recommendation: leave undefined, document)
