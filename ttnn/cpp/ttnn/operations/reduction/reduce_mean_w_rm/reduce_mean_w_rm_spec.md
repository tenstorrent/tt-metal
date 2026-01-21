# reduce_mean_w_rm Functional Specification

## Overview
- **Operation Name**: reduce_mean_w_rm
- **Category**: reduction
- **Planning Mode**: Hybrid
- **Reference Operation(s)**: tilize, reduce_w, untilize
- **Reference Analyses**:
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/tilize_analysis.md` (role: input_stage)
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/reduce_w_analysis.md` (role: compute_core)
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/untilize_analysis.md` (role: output_stage)

## Mathematical Definition

### Formula
```
output[..., 0] = mean(input[..., :]) = (1/W) * sum(input[..., j] for j in range(W))
```

Where:
- `W` is the size of the last (width) dimension
- The output has logical width 1, padded to 32 for tile alignment

### Semantic Description
This operation computes the arithmetic mean across the width (last) dimension of a row-major tensor. The input tensor is first converted from row-major to tiled format, then the mean reduction is computed along the width dimension, and finally the result is converted back to row-major format. The output tensor has the same shape as input except the last dimension becomes 1 (logically), padded to 32 (physically).

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
| **Logical Shape** | `input.shape[:-1] + [1]` (last dimension becomes 1) |
| **Padded Shape** | `input.padded_shape[:-1] + [32]` (padded to tile alignment) |
| **Dtype** | Same as input |
| **Layout** | ROW_MAJOR |
| **Memory Layout** | INTERLEAVED |
| **Buffer Type** | DRAM |

## Component Sources (Hybrid Mode)

This operation is composed from multiple references:

### Input Stage (from tilize_analysis.md)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Reader kernel | tilize.reader | Adapt CB ID from c_0 to c_0 (unchanged) |
| CB_in configuration | tilize.CB_0 | Single-buffered, row-major sticks |
| Compute (tilize phase) | tilize.compute | Use tilize_helpers.hpp, output to CB_1 instead of CB_16 |

### Compute Stage (from reduce_w_analysis.md)
| Component | Source | Modifications |
|-----------|--------|---------------|
| CB_tiled (input) | New (CB_1) | Intermediate tiled data from tilize |
| CB_scaler | reduce_w.CB_2 | Contains 1/W for mean computation |
| CB_reduced (output) | New (CB_3) | Reduced tiles (width=1 tile) |
| Math operations | reduce_w.compute | Use reduce_helpers.hpp with REDUCE_ROW and SUM type |

### Output Stage (from untilize_analysis.md)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Compute (untilize phase) | untilize.compute | Use untilize_helpers.hpp, input from CB_3 |
| CB_out configuration | untilize.CB_16 | Row-major output sticks (width=32 padded) |
| Writer kernel | untilize.writer | Write row-major sticks to DRAM |

### Interface Compatibility
| Interface | Component A | Component B | Format A | Format B | Compatible? |
|-----------|------------|-------------|----------|----------|-------------|
| Reader -> Tilize | CB_0 (row-major sticks) | tilize compute input | ROW_MAJOR | ROW_MAJOR | Yes |
| Tilize -> Reduce | CB_1 (tiled) | reduce compute input | TILE_LAYOUT | TILE_LAYOUT | Yes |
| Reduce -> Untilize | CB_3 (tiled, width=1 tile) | untilize compute input | TILE_LAYOUT | TILE_LAYOUT | Yes |
| Untilize -> Writer | CB_16 (row-major sticks) | writer output | ROW_MAJOR | ROW_MAJOR | Yes |

### CB ID Resolution
| Logical CB | Source Ref | Original ID | Final ID | Notes |
|------------|-----------|-------------|----------|-------|
| CB_in (row-major) | tilize | c_0 | c_0 | Input row-major sticks |
| CB_tiled | tilize (output) / reduce (input) | c_16 / c_0 | c_1 | Intermediate tiled data |
| CB_scaler | reduce_w | c_2 | c_2 | Scaler tile (1/W) |
| CB_reduced | reduce_w (output) / untilize (input) | c_3 / c_0 | c_3 | Reduced tiled data |
| CB_out (row-major) | untilize | c_16 | c_16 | Output row-major sticks |

## Design Decisions

### Decision 1: Single Unified Compute Kernel
- **Choice**: Combine tilize, reduce, and untilize into a single compute kernel
- **Rationale**: All three phases use the kernel helper libraries (tilize_helpers.hpp, reduce_helpers.hpp, untilize_helpers.hpp) which handle hardware initialization and state management. Combining them avoids the overhead of multiple kernel launches and simplifies CB coordination.
- **Alternatives Considered**: Three separate compute kernels (rejected due to complexity and overhead)
- **Tradeoffs**: Single kernel is simpler but tightly couples the three phases

### Decision 2: CB Configuration with 5 Circular Buffers
- **Choice**: Use 5 CBs: c_0 (row-major input), c_1 (tiled), c_2 (scaler), c_3 (reduced tiled), c_16 (row-major output)
- **Rationale**: Each phase transition requires a CB for data handoff. The scaler CB is required by the reduce operation.
- **Alternatives Considered**: Reusing CBs across phases (rejected because phases run sequentially and need distinct buffers for clarity)
- **Tradeoffs**: Uses more CB slots but provides clear data flow

### Decision 3: Single-Core Implementation
- **Choice**: Process entire tensor on a single core
- **Rationale**: User requirement for simplicity. Multi-core can be added later by distributing tile-rows across cores.
- **Alternatives Considered**: Multi-core with work splitting (deferred for future enhancement)
- **Tradeoffs**: Limited parallelism but simpler implementation and debugging

### Decision 4: Mean via SUM with 1/W Scaler
- **Choice**: Compute mean using reduce SUM operation with scaler = 1/W
- **Rationale**: This is the standard approach used by reduce_w for MEAN. The hardware reduce operation multiplies each element by the scaler during reduction.
- **Alternatives Considered**: Separate SUM then divide (rejected due to extra operation)
- **Tradeoffs**: Requires precomputing 1/W and filling scaler tile

### Decision 5: Block-Based Processing
- **Choice**: Process data in blocks of `ntiles_per_block` tiles (width of tensor in tiles)
- **Rationale**: Matches tilize/untilize block granularity. Each block is one horizontal strip of tiles.
- **Alternatives Considered**: Tile-by-tile processing (rejected due to inefficiency)
- **Tradeoffs**: Requires CB capacity to hold entire block width

### Decision 6: Output Width Padding to 32
- **Choice**: Output logical width is 1, but padded width is 32 for tile alignment
- **Rationale**: All tensor dimensions must be tile-aligned for proper TILE_LAYOUT. The reduced result occupies only the first element of each row, but storage is padded.
- **Alternatives Considered**: Keep logical width 1 without padding (impossible with TILE_LAYOUT)
- **Tradeoffs**: Output tensor has padding that users must be aware of

## Work Distribution

### Work Unit Definition
| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (row of tiles) |
| **Unit size** | `Wt` input tiles -> 1 output tile per tile-row |
| **Total units** | `Ht` tile-rows (height in tiles) |
| **Loop structure** | For each tile-row: tilize block, reduce row, untilize result |

### Parallelization Strategy
- **Grid**: 1x1 (single core for initial implementation)
- **Work per core**: All tile-rows
- **Load balancing**: N/A for single core

## Data Flow

### High-Level Flow
```
DRAM (row-major) -> Reader -> CB_0 -> Tilize -> CB_1 -> Reduce -> CB_3 -> Untilize -> CB_16 -> Writer -> DRAM (row-major)
                                                   ^
                                              CB_2 (scaler)
```

### Detailed Data Flow

1. **Reader Kernel** (RISCV_0 / BRISC):
   - Reads row-major sticks from DRAM using TensorAccessor
   - For each block (tile-row), reads 32 consecutive rows (TILE_HEIGHT sticks)
   - Pushes to CB_0 in format ready for tilization
   - Additionally generates scaler tile in CB_2 (value = 1/W for mean)

2. **Compute Kernel** (RISCV_2,3,4 / Unpack, Math, Pack):
   - **Phase 1 - Tilize**:
     - Waits for row-major data in CB_0
     - Calls `compute_kernel_lib::tilize()` from tilize_helpers.hpp
     - Produces tiled data to CB_1
   - **Phase 2 - Reduce**:
     - Waits for tiled data in CB_1 and scaler in CB_2
     - Calls `compute_kernel_lib::reduce<SUM, REDUCE_ROW>()` from reduce_helpers.hpp
     - Produces reduced tiles (width=1) to CB_3
   - **Phase 3 - Untilize**:
     - Waits for reduced tiles in CB_3
     - Calls `compute_kernel_lib::untilize<1>()` from untilize_helpers.hpp
     - Produces row-major output to CB_16

3. **Writer Kernel** (RISCV_1 / NCRISC):
   - Waits for row-major sticks in CB_16
   - Writes to output DRAM buffer using TensorAccessor
   - Output sticks have width = 32 (padded from logical width 1)

### Kernel Data Movement

| Kernel | Core | NOC | Actual Function |
|--------|------|-----|-----------------|
| reader | RISCV_0 | NOC0 | Read row-major sticks from DRAM, generate scaler tile |
| compute | RISCV_2,3,4 | N/A | Tilize -> Reduce -> Untilize |
| writer | RISCV_1 | NOC1 | Write row-major sticks to DRAM |

### Circular Buffer Requirements

| CB ID | Name | Purpose | Producer | Consumer | Sizing Strategy | Lifetime |
|-------|------|---------|----------|----------|-----------------|----------|
| c_0 | CB_in | Row-major input sticks | Reader | Compute (tilize) | `Wt` tiles (one tile-row) | Block |
| c_1 | CB_tiled | Intermediate tiled data | Compute (tilize) | Compute (reduce) | `Wt` tiles (one tile-row) | Block |
| c_2 | CB_scaler | Scaler tile (1/W) | Reader | Compute (reduce) | 1 tile | Program |
| c_3 | CB_reduced | Reduced tiled data | Compute (reduce) | Compute (untilize) | 1 tile (output width) | Block |
| c_16 | CB_out | Row-major output sticks | Compute (untilize) | Writer | 1 tile (output width) | Block |

**CB Sizing Notes**:
- CB_0 and CB_1: Capacity equals `Wt` tiles to hold one complete tile-row (single-buffered)
- CB_2: Capacity is 1 tile, persists for entire program (scaler value 1/W)
- CB_3 and CB_16: Capacity is 1 tile since output width is 1 tile after reduction

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
- Output stick width = 32 elements (padded from logical 1)
- For each tile-row: writes 32 sticks
- Uses TensorAccessor for address generation
- Pattern: Sequential sticks with barrier per stick

### Compute Access

**CB_0 (input)**:
- Wait: `Wt` tiles (one tile-row worth of row-major sticks)
- Pop: `Wt` tiles after tilize_block completes

**CB_1 (tiled)**:
- Reserve: `Wt` tiles
- Push: `Wt` tiles after tilize_block
- Wait: `Wt` tiles before reduce
- Pop: `Wt` tiles after reduce

**CB_2 (scaler)**:
- Wait: 1 tile (at reduce init)
- No pop (persistent throughout program)

**CB_3 (reduced)**:
- Reserve: 1 tile
- Push: 1 tile after reduce
- Wait: 1 tile before untilize
- Pop: 1 tile after untilize

**CB_16 (output)**:
- Reserve: 1 tile
- Push: 1 tile after untilize_block

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
| 1 | output_stick_size | uint32_t | Size of one output row in bytes (32 * element_size) |
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
| Single tile (32x32 input) | Reduces to single output value, padded to 32 |
| Width = 32 (single tile column) | Trivial reduction, each row becomes its mean |
| Large width (many tiles) | Block-based processing handles any tile count |
| Height = 32 (single tile row) | Single block processed |
| Non-tile-aligned logical dimensions | Input must be padded to tile boundaries |
| Empty tensor | Should be rejected with error |

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
- Output logical shape = `input.shape[:-1] + [1]`
- Output padded shape = `input.padded_shape[:-1] + [32]`

### Functional Behavior
- Single tile (32x32): output matches `torch.mean(input, dim=-1, keepdim=True)`
- Multi-tile (64x128): output matches PyTorch reference
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
  - `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
- Documentation consulted:
  - METALIUM_GUIDE.md (kernel architecture, CB coordination)
  - tech_reports/tensor_layouts/tensor_layouts.md (row-major vs tiled layouts)
