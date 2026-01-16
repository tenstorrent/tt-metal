# Execution Log: Layernorm (Interleaved Multi-Core) Analysis

## Session Info
- **Started**: Session start
- **Operation**: layernorm
- **Program Factory Path**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core.cpp`
- **Output Path**: `ttnn/cpp/ttnn/operations/normalization/layernorm_fused_rm/agent_logs/layernorm_analysis.md`

## Execution Timeline

### Step 1: Initialize Logging
**Action**: Attempted to initialize breadcrumbs using shell script
**Command/Tool**: Bash - init_breadcrumbs.sh
**Result**: Tool permission auto-denied (prompts unavailable)
**Decision**: Proceeded with manual tracking of execution steps

### Step 2: Read Program Factory and Discover Kernel Files
**Action**: Read the main program factory file and glob for related kernel files
**Command/Tool**: Read, Glob
**Result**:
- Successfully read `layernorm_op_multi_core.cpp` (591 lines)
- Found kernel files in `device/kernels/` directory:
  - Compute: `layernorm.cpp`, `layernorm_welford.cpp`, `layernorm_large_tensor.cpp`, etc.
  - Dataflow: `reader_unary_interleaved_ln.cpp`, `writer_unary_interleaved_start_id_blocked.cpp`, etc.
**Decision**: Focus on standard (non-Welford, non-large-tensor) interleaved kernels for primary analysis

### Step 3: Read Table Templates
**Action**: Read table templates for standardized output format
**Command/Tool**: Read
**Result**: Retrieved table templates for Tensor Format, CB Configuration, Core Distribution, Arguments, and Kernel Specification
**Decision**: Use templates for consistent documentation format

### Step 4: Read Reader Kernel
**Action**: Read reader_unary_interleaved_ln.cpp
**Command/Tool**: Read
**Result**:
```cpp
// Key findings:
// - Uses TensorAccessor for reading
// - Generates scaler tile using generate_reduce_scaler
// - Generates epsilon scalar using generate_bcast_col_scalar
// - Reads gamma/beta only on first row (ncht == 0)
// - Uses read_block_to_cb helper from layernorm_dataflow_utils.h
```
**Decision**: Need to read utility headers to understand helper functions

### Step 5: Read Compute Kernel
**Action**: Read layernorm.cpp compute kernel
**Command/Tool**: Read
**Result**:
```cpp
// Key findings:
// - row_wise_mean for E[x] computation
// - sub_tiles_bcast_cols for (x - E[x])
// - mul_tiles for (x - E[x])^2
// - row_wise_mean for Var[x]
// - rsqrt_tile for 1/sqrt(Var+eps)
// - mul_tiles_bcast_cols for normalization
// - mul_tiles_bcast_rows for gamma
// - add_tiles_bcast_rows for beta
```
**Decision**: Need to query DeepWiki about broadcast operations

### Step 6: Read Writer Kernel
**Action**: Read writer_unary_interleaved_start_id_blocked.cpp
**Command/Tool**: Read
**Result**:
```cpp
// Simple writer:
// - Uses cb_wait_front, noc_async_write_tile, noc_async_write_barrier, cb_pop_front
// - Processes tiles in blocks
```
**Decision**: Writer is straightforward, focus analysis on compute kernel complexity

### Step 7: Read Utility Headers
**Action**: Read layernorm_dataflow_utils.h, numeric.h, blocked_range.h
**Command/Tool**: Read
**Result**:
- `layernorm_dataflow_utils.h`: `read_block_to_cb` helper function
- `numeric.h`: `row_wise_mean`, `row_wise_accumulate_with_epilogue` implementations
- `blocked_range.h`: `BlockedRange` class for structured iteration
**Decision**: Key insight - row_wise_mean applies 1/N scaling in epilogue

### Step 8: Query DeepWiki - Broadcast Operations
**Action**: Ask about bcast_cols and bcast_rows
**Command/Tool**: mcp__deepwiki__ask_question
**Result**:
```
Key findings:
- bcast_cols: B is single-column, C[h,w] = A[h,w] op B[w]
- bcast_rows: B is single-row, C[h,w] = A[h,w] op B[h]
- Uses llk_unpack_AB and llk_math_eltwise_binary
- Init functions configure hardware (add_bcast_rows_init_short, etc.)
```
**Decision**: This explains gamma/beta application - gamma uses row broadcast (applies per column), not column broadcast

### Step 9: Query DeepWiki - Layernorm Algorithm
**Action**: Ask about row-wise mean and variance computation
**Command/Tool**: mcp__deepwiki__ask_question
**Result**:
```
Key findings:
- Standard: mean via sum/N, variance via E[(x-mean)^2]
- Welford: Iterative update for numerical stability
- numeric::row_wise_mean function handles accumulation
```
**Decision**: Confirmed understanding of algorithm structure

### Step 10: Query DeepWiki - rsqrt_tile
**Action**: Ask about rsqrt_tile function
**Command/Tool**: mcp__deepwiki__ask_question
**Result**:
```
Key findings:
- Element-wise 1/sqrt(x) on destination register
- Must call rsqrt_tile_init first
- LEGACY_RSQRT template controls approximation
```
**Decision**: Documented in normalization section of analysis

### Step 11: Query DeepWiki - Circular Buffers
**Action**: Ask about CB capacity, block size, double buffering
**Command/Tool**: mcp__deepwiki__ask_question
**Result**:
```
Key findings:
- Capacity = total_size / page_size
- Double buffering needs capacity >= 2 * block_size
- Producer/consumer pattern with reserve/push/wait/pop
```
**Decision**: Applied to CB configuration table in analysis

### Step 12: Read Type Definitions
**Action**: Read layernorm_types.hpp, layernorm_common.hpp, layernorm_common.cpp
**Command/Tool**: Read
**Result**:
- `LayerNormType`: LAYERNORM or RMSNORM
- `LayerNormDefaultProgramConfig`: legacy_reduction, legacy_rsqrt, use_welford flags
- `create_reciprocal_tensor_if_needed`: Creates LUT for Welford algorithm
**Decision**: Document program config options and their effects

### Step 13: Read Scaler Generation Utilities
**Action**: Read generate_reduce_scaler.hpp, generate_bcast_scalar.hpp
**Command/Tool**: Read
**Result**:
```cpp
// generate_reduce_scaler:
// - Fills tile with zeros
// - Sets first 8 elements in each of 4 faces to scaler value

// generate_bcast_col_scalar:
// - Creates W-broadcast tile (column-wise scalar)
```
**Decision**: Document scaler tile structure for understanding reduce operations

### Step 14: Read Partial Tile Handling
**Action**: Read custom_tiles.h for partial reduce scaler
**Command/Tool**: Read
**Result**:
```cpp
// generate_partial_reduce_scaler:
// - Creates tile with zeros in unused columns
// - Only sets values for columns < num_cols
```
**Decision**: Important for understanding W not divisible by 32 handling

### Step 15: Read Policies
**Action**: Read policies.h for input handling policies
**Command/Tool**: Read
**Result**:
```cpp
// Policies control CB behavior:
// - PartialBlockWithPopPolicy/WithoutPopPolicy
// - FullBlockWithPopPolicy/WithoutPopPolicy
```
**Decision**: Document policy effect on CB synchronization

### Step 16: Create Output Directory
**Action**: Create agent_logs directory
**Command/Tool**: Bash mkdir
**Result**: Directory created successfully
**Decision**: Proceed with writing analysis document

### Step 17: Write Analysis Document
**Action**: Write comprehensive analysis to layernorm_analysis.md
**Command/Tool**: Write
**Result**: Successfully created 15KB+ analysis document
**Decision**: Document covers all required sections

## Files Read

| File | Purpose | Key Findings |
|------|---------|--------------|
| `layernorm_op_multi_core.cpp` | Main program factory | CB configuration, kernel selection, core distribution |
| `reader_unary_interleaved_ln.cpp` | Reader kernel | Tile reading, scaler generation, gamma/beta read-once pattern |
| `layernorm.cpp` | Compute kernel | Mean, variance, rsqrt, gamma/beta broadcast |
| `writer_unary_interleaved_start_id_blocked.cpp` | Writer kernel | Simple blocked tile writing |
| `layernorm_dataflow_utils.h` | Reader utilities | read_block_to_cb helper |
| `numeric.h` | Compute utilities | row_wise_mean, accumulate_with_epilogue |
| `blocked_range.h` | Generic utilities | BlockedRange iteration class |
| `layernorm_types.hpp` | Type definitions | LayerNormType, program configs |
| `layernorm_common.hpp/cpp` | Common functions | create_reciprocal_tensor_if_needed |
| `generate_reduce_scaler.hpp` | Scaler generation | Reduce scaler tile format |
| `generate_bcast_scalar.hpp` | Broadcast scalars | Column/row broadcast tile generation |
| `custom_tiles.h` | Partial tiles | Partial reduce scaler for non-32 width |
| `policies.h` | CB policies | Input handling policy definitions |
| `table-templates.md` | Output format | Standard table templates |

## DeepWiki Queries

| Query | Response Summary | How Used |
|-------|------------------|----------|
| Broadcast operations (bcast_cols, bcast_rows) | bcast_cols broadcasts single column, bcast_rows broadcasts single row | Explained gamma (row bcast) and mean (col bcast) application |
| Layernorm algorithm in TTNN | Standard mean+variance, Welford variant available | Confirmed algorithm structure in analysis |
| rsqrt_tile function | 1/sqrt(x) on dest register, needs init | Documented in normalization step |
| Circular buffer capacity/buffering | Capacity = total/page, double buffer needs 2x block | Applied to CB configuration table |
| reduce_tile function | Query failed (error) | Proceeded with source code analysis instead |

## Errors Encountered

| Error | Context | Resolution |
|-------|---------|------------|
| Bash permission denied | init_breadcrumbs.sh script | Manual tracking in this log |
| DeepWiki query failed | reduce_tile question | Analyzed source code directly instead |
| File not found | deprecated path for scaler headers | Used Glob to find correct paths |

## Key Decisions

| Decision Point | Options Considered | Choice Made | Rationale |
|----------------|-------------------|-------------|-----------|
| Which kernel variant to analyze | Standard, Welford, Large-tensor | Standard (non-Welford, non-large) | Most common case, clearest for understanding |
| How to handle missing reduce_tile info | DeepWiki retry, source analysis | Source analysis | Sufficient info from numeric.h |
| CB table detail level | High detail vs summary | Comprehensive with all 14+ CBs | Critical for understanding data flow |
| Gamma/beta broadcast direction | Column vs row | Row broadcast (per-column application) | Verified via DeepWiki and code analysis |

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `layernorm_fused_rm/agent_logs/` | Created | Output directory |
| `layernorm_analysis.md` | Created | Full analysis document (~15KB) |
| `analyzer_layernorm_execution_log.md` | Created | This execution log |

## Final Status
- **Completed**: Yes
- **Output File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm_fused_rm/agent_logs/layernorm_analysis.md`
- **Issues**: None - all sections completed

## Deviations from Standard Workflow

1. **Breadcrumb initialization failed**: Bash permissions were auto-denied. Tracked progress manually via TodoWrite and this execution log.

2. **DeepWiki query failure**: One query about reduce_tile failed. Worked around by analyzing the source code in `numeric.h` which provided complete understanding.

## Pain Points

1. **Large number of CBs**: The layernorm operation uses 14+ circular buffers with complex interdependencies. Tracing data flow required careful attention to conditional compilation paths.

2. **Conditional compilation**: The kernel has multiple variants controlled by defines (FUSE_PRE_ADD, FUSE_GAMMA, FUSE_BETA, RMSNORM). This makes the code harder to follow but allows efficient specialization.

3. **Broadcast semantics confusion**: The terms "row broadcast" and "column broadcast" can be confusing. Clarified via DeepWiki that row broadcast means the single-row tensor is applied to all rows (used for gamma/beta which are per-column scalars).
