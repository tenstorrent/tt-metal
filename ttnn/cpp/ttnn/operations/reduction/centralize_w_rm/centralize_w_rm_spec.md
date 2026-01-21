# centralize_w_rm Functional Specification

## Overview
- **Operation Name**: centralize_w_rm
- **Category**: reduction
- **Planning Mode**: Hybrid
- **Reference Operation(s)**: tilize, reduce_w, binary_op (bcast_sub), untilize
- **Reference Analyses**:
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/tilize_analysis.md` (role: input_stage)
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/reduce_w_analysis.md` (role: compute_reduce)
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp` (role: compute_bcast_sub)
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/untilize_analysis.md` (role: output_stage)

## Mathematical Definition

### Formula
```
mean[..., 0] = (1/W) * sum(input[..., j] for j in range(W))
output[..., j] = input[..., j] - mean[..., 0]  for all j in range(W)
```

Where:
- `W` is the size of the last (width) dimension
- The mean is computed per row, then broadcast-subtracted from every element in that row
- The output has the SAME shape as input (not reduced)

### Semantic Description
This operation centralizes data by subtracting the row-wise mean from each element. For each row (along the last dimension), it:
1. Computes the arithmetic mean across the width dimension
2. Broadcasts this mean value to all positions in the row
3. Subtracts the broadcast mean from the original values

The result has zero mean along each row. This is a common preprocessing step for normalization operations.

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | - | - | Input tensor in row-major layout |
| memory_config | MemoryConfig | No | - | input.memory_config() | Output memory configuration |

### Input Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Rank | Must be >= 2D | "must be at least 2D" |
| Layout | ROW_MAJOR | "must be in ROW_MAJOR layout" |
| Memory Layout | INTERLEAVED | "must be interleaved" |
| Device | Must be on device | "must be on device" |
| Dtype | BFLOAT16 or FLOAT32 | "unsupported dtype" |
| Width | Must be divisible by 32 (padded) | "width must be padded to tile boundary" |
| Height | Must be divisible by 32 (padded) | "height must be padded to tile boundary" |

### Output Tensor Specification

| Property | Value |
|----------|-------|
| **Logical Shape** | Same as input (no dimension reduction) |
| **Padded Shape** | Same as input padded shape |
| **Dtype** | Same as input |
| **Layout** | ROW_MAJOR |
| **Memory Layout** | INTERLEAVED |
| **Buffer Type** | DRAM |

## Component Sources (Hybrid Mode)

This operation is composed from multiple references:

### Input Stage (from tilize_analysis.md)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Reader kernel | tilize.reader | Adapt to read input sticks, push to CB_0 |
| CB_in configuration | tilize.CB_0 | Single-buffered, row-major sticks |
| Compute (tilize phase) | tilize.compute | Use tilize_helpers.hpp, output to CB_1 |

### Compute Stage - Reduce (from reduce_w_analysis.md)
| Component | Source | Modifications |
|-----------|--------|---------------|
| CB_tiled (input) | CB_1 | Intermediate tiled data from tilize |
| CB_scaler | reduce_w.CB_2 | Contains 1/W for mean computation |
| CB_mean (output) | CB_3 | Reduced tiles (Ht x 1 tiles) containing row means |
| Math operations | reduce_w.compute | Use reduce_helpers.hpp with REDUCE_ROW and SUM type |

### Compute Stage - Broadcast Subtract (from binary_op_helpers.hpp)
| Component | Source | Modifications |
|-----------|--------|---------------|
| CB_original | CB_1 | Reuse tiled input (must keep for subtraction) |
| CB_mean | CB_3 | Mean values to broadcast |
| CB_centralized | CB_4 | Result of broadcast subtraction |
| Math operations | binary_op_helpers.hpp | Use `sub<BroadcastDim::COL>()` |

### Output Stage (from untilize_analysis.md)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Compute (untilize phase) | untilize.compute | Use untilize_helpers.hpp, input from CB_4 |
| CB_out configuration | CB_16 | Row-major output sticks (full width) |
| Writer kernel | untilize.writer | Write row-major sticks to DRAM |

### Interface Compatibility
| Interface | Component A | Component B | Format A | Format B | Compatible? |
|-----------|------------|-------------|----------|----------|-------------|
| Reader -> Tilize | CB_0 (row-major sticks) | tilize compute input | ROW_MAJOR | ROW_MAJOR | Yes |
| Tilize -> Reduce | CB_1 (tiled) | reduce compute input | TILE_LAYOUT | TILE_LAYOUT | Yes |
| Tilize -> BcastSub | CB_1 (tiled) | bcast_sub input A | TILE_LAYOUT | TILE_LAYOUT | Yes |
| Reduce -> BcastSub | CB_3 (tiled, Ht x 1) | bcast_sub input B | TILE_LAYOUT | TILE_LAYOUT | Yes |
| BcastSub -> Untilize | CB_4 (tiled, Ht x Wt) | untilize compute input | TILE_LAYOUT | TILE_LAYOUT | Yes |
| Untilize -> Writer | CB_16 (row-major sticks) | writer output | ROW_MAJOR | ROW_MAJOR | Yes |

### CB ID Resolution
| Logical CB | Source Ref | Original ID | Final ID | Notes |
|------------|-----------|-------------|----------|-------|
| CB_in (row-major) | tilize | c_0 | c_0 | Input row-major sticks |
| CB_tiled | tilize (output) | c_16 | c_1 | Tiled data (kept for subtraction) |
| CB_scaler | reduce_w | c_2 | c_2 | Scaler tile (1/W) |
| CB_mean | reduce_w (output) | c_3 | c_3 | Reduced tiled data (Ht x 1) |
| CB_centralized | bcast_sub (output) | new | c_4 | Centralized tiled data (Ht x Wt) |
| CB_out (row-major) | untilize | c_16 | c_16 | Output row-major sticks |

## Design Decisions

### Decision 1: Single Unified Compute Kernel
- **Choice**: Combine tilize, reduce, broadcast-subtract, and untilize into a single compute kernel
- **Rationale**: All four phases use kernel helper libraries which handle hardware initialization. Combining them avoids multiple kernel launches and simplifies CB coordination. The phases execute sequentially within each block.
- **Alternatives Considered**: Four separate compute kernels (rejected due to complexity and synchronization overhead)
- **Tradeoffs**: Single kernel is simpler but requires careful CB management

### Decision 2: Keep Original Tiled Data for Subtraction
- **Choice**: Use CB_1 to store tiled input data, which persists across reduce phase for use in broadcast subtract
- **Rationale**: After tilizing, we need both the reduced mean AND the original tiled data. CB_1 capacity must hold Wt tiles (full tile-row width) to enable subtraction from all original tiles.
- **Alternatives Considered**: Re-read from DRAM or re-tilize (rejected due to inefficiency)
- **Tradeoffs**: Larger CB footprint but avoids redundant data movement

### Decision 3: CB Configuration with 6 Circular Buffers
- **Choice**: Use 6 CBs: c_0 (row-major input), c_1 (tiled original), c_2 (scaler), c_3 (mean), c_4 (centralized), c_16 (row-major output)
- **Rationale**: Each phase transition requires a CB for data handoff. The extra CB_4 is needed because we cannot overwrite CB_1 (original data needed for subtraction).
- **Alternatives Considered**: Reusing CB_1 for output (rejected because original data needed for subtraction)
- **Tradeoffs**: Uses more CB slots but provides clear data flow

### Decision 4: Single-Core Implementation
- **Choice**: Process entire tensor on a single core
- **Rationale**: Matches reduce_mean_w_rm reference for simplicity. Multi-core can be added later by distributing tile-rows across cores.
- **Alternatives Considered**: Multi-core with work splitting (deferred for future enhancement)
- **Tradeoffs**: Limited parallelism but simpler implementation and debugging

### Decision 5: Mean via SUM with 1/W Scaler
- **Choice**: Compute mean using reduce SUM operation with scaler = 1/W
- **Rationale**: Standard approach used by reduce_w for MEAN. The hardware reduce operation multiplies each element by the scaler during reduction.
- **Alternatives Considered**: Separate SUM then divide (rejected due to extra operation)
- **Tradeoffs**: Requires precomputing 1/W and filling scaler tile

### Decision 6: Block-Based Processing with Full Row Retention
- **Choice**: Process data in blocks of one tile-row at a time, keeping full row in CB_1 for subtraction
- **Rationale**: For centralization, we need ALL Wt tiles of a row to subtract the mean from each. After reducing Wt tiles to 1 mean tile, we broadcast-subtract from all Wt original tiles.
- **Alternatives Considered**: Tile-by-tile processing (not possible - need full row for subtraction)
- **Tradeoffs**: CB_1 and CB_4 must hold Wt tiles (full tile-row width)

### Decision 7: BroadcastDim::COL for Mean Subtraction
- **Choice**: Use `sub<BroadcastDim::COL>()` from binary_op_helpers.hpp
- **Rationale**: REDUCE_ROW produces a column-shaped output (Ht x 1 tiles). To broadcast this across the width dimension, we use COL broadcast which replicates the column values across all width positions.
- **Alternatives Considered**: Manual broadcast loop (rejected - helper library is cleaner)
- **Tradeoffs**: Relies on binary_op_helpers.hpp which is well-tested

## Work Distribution

### Work Unit Definition
| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (row of tiles) |
| **Unit size** | `Wt` input tiles -> `Wt` output tiles per tile-row |
| **Total units** | `Ht` tile-rows (height in tiles) |
| **Loop structure** | For each tile-row: tilize block, reduce to mean, bcast-sub from original, untilize result |

### Parallelization Strategy
- **Grid**: 1x1 (single core for initial implementation)
- **Work per core**: All tile-rows
- **Load balancing**: N/A for single core

## Data Flow

### High-Level Flow
```
DRAM (row-major) -> Reader -> CB_0 -> Tilize -> CB_1 -----> BcastSub -> CB_4 -> Untilize -> CB_16 -> Writer -> DRAM
                                                  |              ^
                                                  v              |
                                              Reduce -----> CB_3 (mean)
                                                  ^
                                                  |
                                             CB_2 (scaler)
```

### Detailed Data Flow

1. **Reader Kernel** (RISCV_0 / BRISC):
   - Reads row-major sticks from DRAM using TensorAccessor
   - For each block (tile-row), reads 32 consecutive rows (TILE_HEIGHT sticks)
   - Each stick has width = `padded_width` elements
   - Pushes to CB_0 in format ready for tilization
   - Additionally generates scaler tile in CB_2 (value = 1/W for mean)

2. **Compute Kernel** (RISCV_2,3,4 / Unpack, Math, Pack):
   - **Phase 1 - Tilize**:
     - Waits for row-major data in CB_0
     - Calls `compute_kernel_lib::tilize()` from tilize_helpers.hpp
     - Produces tiled data to CB_1 (Wt tiles per tile-row)
   - **Phase 2 - Reduce**:
     - Waits for tiled data in CB_1 and scaler in CB_2
     - Calls `compute_kernel_lib::reduce<SUM, REDUCE_ROW>()` from reduce_helpers.hpp
     - Produces reduced mean tile (1 tile) to CB_3
     - NOTE: Does NOT pop CB_1 - original data needed for subtraction
   - **Phase 3 - Broadcast Subtract**:
     - Uses CB_1 (original tiled, Wt tiles) and CB_3 (mean, 1 tile)
     - Calls `compute_kernel_lib::sub<BroadcastDim::COL>()` from binary_op_helpers.hpp
     - Produces centralized tiles (Wt tiles) to CB_4
     - Pops CB_1 and CB_3 after subtraction
   - **Phase 4 - Untilize**:
     - Waits for centralized tiles in CB_4
     - Calls `compute_kernel_lib::untilize<Wt>()` from untilize_helpers.hpp
     - Produces row-major output to CB_16

3. **Writer Kernel** (RISCV_1 / NCRISC):
   - Waits for row-major sticks in CB_16
   - Writes to output DRAM buffer using TensorAccessor
   - Output sticks have same width as input (full width preserved)

### Kernel Data Movement

| Kernel | Core | NOC | Actual Function |
|--------|------|-----|-----------------|
| reader | RISCV_0 | NOC0 | Read row-major sticks from DRAM, generate scaler tile |
| compute | RISCV_2,3,4 | N/A | Tilize -> Reduce -> BcastSub -> Untilize |
| writer | RISCV_1 | NOC1 | Write row-major sticks to DRAM |

### Circular Buffer Requirements

| CB ID | Name | Purpose | Producer | Consumer | Sizing Strategy | Lifetime |
|-------|------|---------|----------|----------|-----------------|----------|
| c_0 | CB_in | Row-major input sticks | Reader | Compute (tilize) | `Wt` tiles (one tile-row) | Block |
| c_1 | CB_tiled | Original tiled data | Compute (tilize) | Compute (bcast_sub) | `Wt` tiles (one tile-row) | Block |
| c_2 | CB_scaler | Scaler tile (1/W) | Reader | Compute (reduce) | 1 tile | Program |
| c_3 | CB_mean | Reduced mean tiles | Compute (reduce) | Compute (bcast_sub) | 1 tile | Block |
| c_4 | CB_centralized | Centralized tiled data | Compute (bcast_sub) | Compute (untilize) | `Wt` tiles (one tile-row) | Block |
| c_16 | CB_out | Row-major output sticks | Compute (untilize) | Writer | `Wt` tiles (one tile-row) | Block |

**CB Sizing Notes**:
- CB_0, CB_1, CB_4, CB_16: Capacity equals `Wt` tiles to hold one complete tile-row
- CB_2: Capacity is 1 tile, persists for entire program (scaler value 1/W)
- CB_3: Capacity is 1 tile (single mean value per tile-row)
- Total L1 footprint per core: ~4*Wt + 2 tiles

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access

**Read Pattern**:
- Reads row-major sticks from input DRAM buffer
- For each block (tile-row): reads 32 sticks sequentially
- Each stick has width = `padded_width` bytes
- Uses TensorAccessor for address generation
- Pattern: Sequential sticks, batched by 32 for tile height

**Scaler Generation**:
- Generates one scaler tile containing value 1/W (reciprocal of width)
- Uses `generate_reduce_scaler()` pattern from reduce_w reference
- Value is packed as 2x bfloat16 in uint32

### RISCV_1 ("writer" / NCRISC) Access

**Write Pattern**:
- Writes row-major sticks to output DRAM buffer
- Output stick width = `padded_width` elements (same as input)
- For each tile-row: writes 32 sticks
- Uses TensorAccessor for address generation
- Pattern: Sequential sticks with barrier per stick

### Compute Access

**CB_0 (input)**:
- Wait: `Wt` tiles (one tile-row worth of row-major sticks)
- Pop: `Wt` tiles after tilize_block completes

**CB_1 (tiled original)**:
- Reserve: `Wt` tiles
- Push: `Wt` tiles after tilize_block
- Wait: `Wt` tiles before reduce (for reduce input)
- NOTE: Do NOT pop after reduce - needed for bcast_sub
- Pop: `Wt` tiles after bcast_sub completes

**CB_2 (scaler)**:
- Wait: 1 tile (at reduce init)
- No pop (persistent throughout program)

**CB_3 (mean)**:
- Reserve: 1 tile
- Push: 1 tile after reduce
- Wait: 1 tile before bcast_sub
- Pop: 1 tile after bcast_sub

**CB_4 (centralized)**:
- Reserve: `Wt` tiles
- Push: `Wt` tiles after bcast_sub
- Wait: `Wt` tiles before untilize
- Pop: `Wt` tiles after untilize

**CB_16 (output)**:
- Reserve: `Wt` tiles
- Push: `Wt` tiles after untilize_block

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one input row in bytes |
| 1 | packed_scaler_value | uint32_t | Scaler (1/W) packed as 2x bfloat16 |
| 2+ | TensorAccessorArgs | struct | Source buffer accessor parameters |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_blocks | uint32_t | Number of tile-rows to process (Ht) |
| 1 | Wt | uint32_t | Width in tiles (tiles per row) |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | Output CB ID (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output row in bytes |
| 2+ | TensorAccessorArgs | struct | Destination buffer accessor parameters |

## Runtime Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_sticks | uint32_t | Total rows to read (`Ht * 32`) |
| 2 | start_stick_id | uint32_t | Starting row index (0 for single core) |

### Compute Kernel

(No runtime arguments - all parameters are compile-time)

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_sticks | uint32_t | Total output rows to write (`Ht * 32`) |
| 2 | start_stick_id | uint32_t | Starting output row index (0 for single core) |

## Edge Cases

| Condition | Expected Behavior |
|-----------|------------------|
| Single tile (32x32 input) | Centralizes to single output tile, each row mean-subtracted |
| Width = 32 (single tile column) | Trivial reduction, each row becomes its mean-subtracted value |
| Large width (many tiles) | Block-based processing handles any tile count |
| Height = 32 (single tile row) | Single block processed |
| Non-tile-aligned logical dimensions | Input must be padded to tile boundaries |
| Empty tensor | Should be rejected with error |
| Constant row (all same values) | Output is all zeros for that row |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-operation-scaffolder** | API Specification, Input Tensor Requirements, Output Tensor Specification |
| **ttnn-factory-builder** | Circular Buffer Requirements, Work Distribution, Data Flow, Component Sources |
| **ttnn-kernel-dataflow** | Kernel Data Movement, Memory Access Patterns, Component Sources |
| **ttnn-kernel-compute** | Compute Access, Mathematical Definition, Component Sources |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

What behavior should be verified (agents decide how/when to test):

### Validation Behavior
- Wrong tensor rank (< 2D) -> error containing "must be at least 2D"
- Wrong layout (TILE_LAYOUT) -> error containing "must be in ROW_MAJOR layout"
- Wrong memory layout (SHARDED) -> error containing "must be interleaved"
- Tensor not on device -> error containing "must be on device"
- Unsupported dtype (e.g., INT32) -> error containing "unsupported dtype"
- Non-tile-aligned width -> error containing "width must be padded to tile boundary"
- Non-tile-aligned height -> error containing "height must be padded to tile boundary"

### Shape Behavior
- Output logical shape = input logical shape (no reduction)
- Output padded shape = input padded shape

### Functional Behavior
- Single tile (32x32): output matches `input - torch.mean(input, dim=-1, keepdim=True)`
- Multi-tile (64x128): output matches PyTorch reference
- Row means of output should be approximately zero
- Numerical accuracy: relative tolerance 1e-3 for bfloat16, 1e-5 for float32

## Open Questions

None - all design decisions have been made with reasonable assumptions for single-core row-major implementation.

## References
- Reference analyses:
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/tilize_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/reduce_w_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/untilize_analysis.md`
- Kernel helper libraries:
  - `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp`
  - `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`
  - `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
- Low-level APIs:
  - `tt_metal/include/compute_kernel_api/bcast.h` (sub_tiles_bcast_cols)
- Documentation consulted:
  - METALIUM_GUIDE.md (kernel architecture, CB coordination)
  - tech_reports/tensor_layouts/tensor_layouts.md (row-major vs tiled layouts)
  - DeepWiki: broadcast operations in TTNN compute kernels
