# reduce_avg_w_rm Functional Specification

## Overview
- **Operation Name**: reduce_avg_w_rm
- **Category**: reduction (fused with data_movement)
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize, reduce_w, untilize
- **Reference Analyses**:
  - `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_analysis.md` (role: input_stage)
  - `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_analysis.md` (role: compute_core)
  - `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_analysis.md` (role: output_stage)

## Mathematical Definition

### Formula
```
output[n, c, h, 0] = (1/W) * sum(input[n, c, h, w] for w in 0..W-1)
```

Where:
- `input` has shape `[N, C, H, W]`
- `output` has shape `[N, C, H, 1]` (logical), but physical width is tile-aligned (32)
- The scaler `1/W` converts sum to average

### Semantic Description
This operation computes the average of all elements along the width (last) dimension of a row-major input tensor, producing a row-major output tensor. The operation internally:
1. Tilizes row-major input data into 32x32 tiles
2. Performs width reduction (sum with 1/W scaler) on tiled data
3. Untilizes the reduced tiled output back to row-major format

This fused approach avoids intermediate memory round-trips between tilize, reduce, and untilize operations.

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | - | - | Input tensor in ROW_MAJOR layout |
| memory_config | MemoryConfig | No | - | input memory config | Output memory configuration |
| compute_kernel_config | ComputeKernelConfig | No | - | nullopt | Optional compute kernel configuration |

### Input Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Rank | Must be 4D | "Input tensor must be 4D [N, C, H, W]" |
| Layout | ROW_MAJOR | "Input tensor must be in ROW_MAJOR layout" |
| Memory layout | INTERLEAVED | "Input tensor must be interleaved" |
| Dtype | BFLOAT16 or FLOAT32 | "Unsupported dtype; must be bfloat16 or float32" |
| Device | Must be on device | "Input tensor must be on device" |
| Width | Must be > 0 | "Width dimension must be positive" |
| Height | Must be tile-aligned (multiple of 32) | "Height must be a multiple of 32 for tilization" |
| Width | Must be tile-aligned (multiple of 32) | "Width must be a multiple of 32 for tilization" |

### Output Tensor Specification
- **Shape**: `[N, C, H, 32]` (physical width padded to tile width)
- **Logical shape**: `[N, C, H, 1]` (only first element per row is valid)
- **Dtype**: Same as input
- **Layout**: ROW_MAJOR
- **Memory layout**: INTERLEAVED (or as specified by memory_config)

**Note**: The physical output width is 32 (one tile width) because untilize produces tile-aligned row-major data. The logical reduction produces width=1, but this is stored in the first column of a width-32 row-major output.

## Component Sources (Hybrid Mode)

This operation is composed from three reference operations:

### Input Stage (from tilize_multi_core_interleaved)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Reader kernel | tilize.reader | Modified to generate scaler tile (from reduce_w) |
| CB_rm_in configuration | tilize.CB_0 | New CB for raw row-major stick data |
| Compute (tilize phase) | tilize.compute | Use `compute_kernel_lib::tilize()` |

### Compute Stage (from reduce_w + new)
| Component | Source | Modifications |
|-----------|--------|---------------|
| CB_tilized | New | Intermediate CB between tilize and reduce |
| CB_scaler | reduce_w.CB_2 | Scaler tile for 1/W averaging |
| Math operations | reduce_w.compute | Use `compute_kernel_lib::reduce<SUM, REDUCE_ROW>()` |
| CB_reduced | New | Intermediate CB between reduce and untilize |

### Output Stage (from untilize_multi_core)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Compute (untilize phase) | untilize.compute | Use `compute_kernel_lib::untilize<>()` |
| CB_rm_out configuration | untilize.CB_16 | Output CB for row-major data |
| Writer kernel | untilize.writer | Adapted for row-major stick writes |

### Interface Compatibility

| Interface | Component A | Component B | Format A | Format B | Compatible? |
|-----------|------------|-------------|----------|----------|-------------|
| Reader->Compute | CB_rm_in | tilize | Row-major sticks | Row-major sticks | Yes |
| Tilize->Reduce | CB_tilized | reduce | Tiled (32x32) | Tiled (32x32) | Yes |
| Reduce->Untilize | CB_reduced | untilize | Tiled (32x32) | Tiled (32x32) | Yes |
| Untilize->Writer | CB_rm_out | writer | Row-major sticks | Row-major sticks | Yes |

### CB ID Resolution

| Logical CB | Source Ref | Original ID | Final ID | Notes |
|------------|-----------|-------------|----------|-------|
| CB_rm_in | tilize | c_0 | c_0 | Raw row-major input sticks |
| CB_tilized | New | - | c_1 | Tilized data for reduce input |
| CB_scaler | reduce_w | c_2 | c_2 | Scaler tile (1/W) |
| CB_reduced | New | - | c_3 | Reduced tiled output |
| CB_rm_out | untilize | c_16 | c_16 | Row-major output sticks |

## Design Decisions

### Decision 1: Single Compute Kernel for All Three Phases
- **Choice**: One compute kernel performs tilize -> reduce -> untilize sequentially
- **Rationale**: Eliminates intermediate DRAM round-trips; reduced data stays in L1
- **Alternatives Considered**: Three separate kernels (tilize, reduce, untilize) with DRAM storage
- **Tradeoffs**: More complex kernel logic, but significantly better performance due to data locality

### Decision 2: Work Unit is Tile Row (NC * Ht)
- **Choice**: Distribute work by tile rows, where each tile row contains Wt input tiles that reduce to 1 output tile
- **Rationale**: Matches reduce_w pattern; allows complete width reduction within a core
- **Alternatives Considered**: Tile-based distribution (would complicate reduction accumulation)
- **Tradeoffs**: Good parallelism for tall tensors; may underutilize cores for short tensors

### Decision 3: Scaler Generation in Reader Kernel
- **Choice**: Reader generates scaler tile (1/W) using `generate_reduce_scaler()`
- **Rationale**: Follows reduce_w pattern; scaler persists for entire program
- **Alternatives Considered**: Pre-allocated scaler buffer, compute-side scaler generation
- **Tradeoffs**: Small overhead for scaler generation, but simplifies host code

### Decision 4: Double-Buffered CBs for Pipeline Overlap
- **Choice**: CB_rm_in and CB_tilized are double-buffered; CB_scaler single-buffered (persistent)
- **Rationale**: Allows reader to prefetch next block while compute processes current
- **Alternatives Considered**: Single-buffering (simpler but no overlap)
- **Tradeoffs**: 2x CB memory usage, but better throughput

### Decision 5: Output Width is Tile-Aligned (32)
- **Choice**: Physical output has width=32 (one tile column), logical width=1
- **Rationale**: Untilize naturally produces tile-aligned row-major data
- **Alternatives Considered**: Strided write to produce width=1 output
- **Tradeoffs**: Slightly larger output buffer, but simpler implementation

### Decision 6: Use Kernel Helper Library Functions
- **Choice**: Use `compute_kernel_lib::tilize()`, `reduce()`, and `untilize()` helpers
- **Rationale**: Standardized, tested implementations; handles CB management and hardware init
- **Alternatives Considered**: Raw LLK calls
- **Tradeoffs**: Slight abstraction overhead, but much cleaner code and fewer bugs

## Work Distribution

### Work Unit Definition
- **Granularity**: Tile row
- **Unit size**: One tile row = Wt input tiles -> 1 output tile
- **Total units**: `NC * Ht` where `NC = N * C`, `Ht = H / 32`
- **Loop structure**: For each tile row: read 32 sticks, tilize to Wt tiles, reduce to 1 tile, untilize to 32 sticks

### Parallelization Strategy
- **Grid**: 1D (linear enumeration of available cores)
- **Work per core**: `num_tile_rows_per_core = ceil(NC * Ht / num_cores)`
- **Load balancing**: Two-group split (core_group_1 with +1 rows, core_group_2 with base rows)

**Distribution Algorithm**:
```
num_rows = NC * Ht
(num_cores, all_cores, core_group_1, core_group_2,
 rows_per_core_group_1, rows_per_core_group_2) = split_work_to_cores(grid_size, num_rows)
```

## Data Flow

### High-Level Flow
```
DRAM (ROW_MAJOR)     L1 Compute Kernel                              DRAM (ROW_MAJOR)
+---------------+    +------------------------------------------+    +---------------+
|               |    |                                          |    |               |
| stick_0       |--->| CB_rm_in --> [TILIZE] --> CB_tilized     |    | out_stick_0   |
| stick_1       |    |                              |           |    | out_stick_1   |
| ...           |    |            CB_scaler ------->|           |    | ...           |
| stick_31      |    |                              v           |    | out_stick_31  |
|               |    |             CB_reduced <-- [REDUCE_W]    |    |               |
|               |    |                  |                       |    |               |
|               |    |                  v                       |    |               |
|               |    |           [UNTILIZE] --> CB_rm_out ------+--->|               |
+---------------+    +------------------------------------------+    +---------------+
    Reader                     Compute Kernel                           Writer
```

### Detailed Per-Block Flow

**Reader Kernel** (per tile row):
1. Generate scaler tile once (1/W) into CB_scaler (first iteration only)
2. For each block (32 sticks = one tile height):
   - Calculate NoC addresses for 32 consecutive sticks
   - `cb_reserve_back(CB_rm_in, ntiles_per_block)` where ntiles_per_block = Wt
   - Async read 32 sticks into CB_rm_in
   - `noc_async_read_barrier()`
   - `cb_push_back(CB_rm_in, ntiles_per_block)`

**Compute Kernel** (per tile row):
1. **Tilize Phase**:
   - `cb_wait_front(CB_rm_in, Wt)` - wait for Wt tiles worth of stick data
   - `cb_reserve_back(CB_tilized, Wt)`
   - `tilize_block(CB_rm_in, Wt, CB_tilized)` - tilize sticks into Wt tiles
   - `cb_push_back(CB_tilized, Wt)`
   - `cb_pop_front(CB_rm_in, Wt)`

2. **Reduce Phase** (for each of the Wt tiles, accumulating to 1 output tile):
   - `tile_regs_acquire()`
   - For wt in 0..Wt:
     - `cb_wait_front(CB_tilized, 1)`
     - `reduce_tile<SUM, REDUCE_ROW>(CB_tilized, CB_scaler, 0, 0, 0)`
     - `cb_pop_front(CB_tilized, 1)`
   - `cb_reserve_back(CB_reduced, 1)`
   - `tile_regs_commit(); tile_regs_wait()`
   - `pack_tile(0, CB_reduced)`
   - `tile_regs_release()`
   - `cb_push_back(CB_reduced, 1)`

3. **Untilize Phase**:
   - `cb_wait_front(CB_reduced, 1)` - wait for 1 reduced tile
   - `cb_reserve_back(CB_rm_out, 1)`
   - `pack_untilize_block<1, 1>(CB_reduced, 1, CB_rm_out, 0)` - untilize 1 tile to 32 sticks
   - `cb_push_back(CB_rm_out, 1)`
   - `cb_pop_front(CB_reduced, 1)`

**Writer Kernel** (per tile row):
1. For each of 32 output sticks in the tile row:
   - `cb_wait_front(CB_rm_out, 1)`
   - Calculate output stick NoC address
   - `noc_async_write()` - write one stick (width=32 elements = 64 bytes for bf16)
   - `noc_async_write_barrier()`
   - `cb_pop_front(CB_rm_out, 1)`

### Kernel Data Movement

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (sticks) | CB_rm_in (c_0), CB_scaler (c_2) | Read 32 sticks per block, generate scaler once |
| compute | UNPACK/MATH/PACK | N/A | CB_rm_in, CB_scaler | CB_rm_out | tilize, reduce_w, untilize |
| writer | RISCV_1 | NOC1 | CB_rm_out (c_16) | DRAM (sticks) | Write 32 output sticks per block |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | CB_rm_in | Raw row-major input sticks | 2 * Wt tiles | Wt tiles | Double | Reader | Compute (tilize) | Block |
| c_1 | CB_tilized | Tilized intermediate data | 2 tiles | 1 tile | Double | Compute (tilize) | Compute (reduce) | Block |
| c_2 | CB_scaler | Scaler tile (1/W) | 1 tile | 1 tile | Single | Reader | Compute (reduce) | Program |
| c_3 | CB_reduced | Reduced tiled data | 2 tiles | 1 tile | Double | Compute (reduce) | Compute (untilize) | Block |
| c_16 | CB_rm_out | Row-major output sticks | 2 tiles | 1 tile | Double | Compute (untilize) | Writer | Block |

### CB Sizing Rationale

**CB_rm_in (c_0)**:
- Holds 32 sticks (one tile height) with width = W elements
- Stick data is logically "Wt tiles worth" since tilize_block processes block_w tiles
- Capacity: `2 * Wt * single_tile_size` for double-buffering
- Block size: `Wt * single_tile_size` (all sticks for one tile row)

**CB_tilized (c_1)**:
- Intermediate storage between tilize and reduce
- Streaming: reduce processes tiles one at a time
- Capacity: 2 tiles (double-buffered for tilize->reduce overlap)

**CB_scaler (c_2)**:
- Single scaler tile with value 1/W
- Generated once by reader, persists for entire program
- Capacity: 1 tile (single-buffered, never changes)

**CB_reduced (c_3)**:
- Holds 1 reduced tile per tile row
- Capacity: 2 tiles (double-buffered for reduce->untilize overlap)

**CB_rm_out (c_16)**:
- Holds untilized row-major output (32 sticks, each width=32 elements)
- Capacity: 2 tiles worth of sticks (double-buffered for untilize->writer overlap)

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access

**Read Pattern**: Sequential sticks from DRAM
```
For each tile_row assigned to this core:
    For j in 0..31 (tile_height):
        stick_id = tile_row * 32 + j
        Read stick[stick_id] from DRAM (W * element_size bytes)
```

**Access Details**:
- Pattern: Sequential within tile row, strided across tensor
- Granularity: One stick (row) per NoC read
- Interleaving: Sticks distributed round-robin across DRAM banks
- Uses `TensorAccessor` for address calculation

**Scaler Generation**:
- Generated once at kernel start
- Uses `generate_reduce_scaler(CB_scaler, packed_scaler_value)`
- Scaler value = 1.0f / W (computed on host, packed as 2x bfloat16)

### RISCV_1 ("writer" / NCRISC) Access

**Write Pattern**: Sequential sticks to DRAM
```
For each tile_row processed:
    For j in 0..31 (tile_height):
        output_stick_id = tile_row * 32 + j
        Write output_stick to DRAM (32 * element_size bytes = 64 bytes for bf16)
```

**Access Details**:
- Pattern: Sequential within tile row
- Granularity: One stick (32 elements) per NoC write
- Output sticks are narrower than input (32 vs W elements)
- Uses `TensorAccessor` for address calculation

### Compute Access

**CB Read/Write Patterns**:

1. **Tilize Phase**:
   - Read: `cb_wait_front(CB_rm_in, Wt)` - wait for all stick data
   - Write: `cb_reserve_back(CB_tilized, Wt)` - reserve space for tilized tiles
   - Process: `tilize_block` converts stick layout to tile layout
   - Release: `cb_pop_front(CB_rm_in, Wt)`, `cb_push_back(CB_tilized, Wt)`

2. **Reduce Phase**:
   - Read: `cb_wait_front(CB_tilized, 1)` per tile (streaming)
   - Read: `cb_wait_front(CB_scaler, 1)` once at start
   - Write: `cb_reserve_back(CB_reduced, 1)` per output tile
   - Process: Accumulate Wt tiles into 1 tile using `reduce_tile<SUM, REDUCE_ROW>`
   - Release: `cb_pop_front(CB_tilized, 1)` per tile, `cb_push_back(CB_reduced, 1)` per output

3. **Untilize Phase**:
   - Read: `cb_wait_front(CB_reduced, 1)` - wait for reduced tile
   - Write: `cb_reserve_back(CB_rm_out, 1)` - reserve space for sticks
   - Process: `pack_untilize_block` converts tile to stick layout
   - Release: `cb_pop_front(CB_reduced, 1)`, `cb_push_back(CB_rm_out, 1)`

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one input row in bytes (W * element_size) |
| 1 | packed_scaler_value | uint32_t | Two bfloat16 scaler values (1/W) packed into uint32 |
| 2+ | TensorAccessorArgs | varies | Bank distribution info for input tensor |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Width in tiles (tiles per row to reduce) |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_stick_size | uint32_t | Size of one output row in bytes (32 * element_size) |
| 1+ | TensorAccessorArgs | varies | Bank distribution info for output tensor |

## Runtime Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Base address of input buffer in DRAM |
| 1 | num_tile_rows | uint32_t | Number of tile rows to process (NC * Ht / num_cores) |
| 2 | start_tile_row | uint32_t | First tile row index this core processes |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tile_rows | uint32_t | Number of tile rows to process |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Base address of output buffer in DRAM |
| 1 | num_tile_rows | uint32_t | Number of tile rows to process |
| 2 | start_tile_row | uint32_t | First tile row index this core writes |

## Edge Cases

| Condition | Expected Behavior |
|-----------|-------------------|
| Single tile row (H=32, NC=1) | Process on single core, valid operation |
| Large tensor (many tile rows) | Distribute across all available cores |
| Width = 32 (Wt=1) | Valid, but trivial reduction (average of 32 columns) |
| Large width (Wt >> 1) | Valid, more reduction work per tile row |
| Non-tile-aligned height | Error: "Height must be a multiple of 32" |
| Non-tile-aligned width | Error: "Width must be a multiple of 32" |
| Empty tensor (any dim = 0) | Error: "Tensor dimensions must be positive" |
| Wrong layout (TILE) | Error: "Input must be ROW_MAJOR layout" |
| Wrong memory layout (SHARDED) | Error: "Input must be interleaved" |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-operation-scaffolder** | API Specification, Input Tensor Requirements, Output Tensor Specification, Edge Cases (for validation) |
| **ttnn-factory-builder** | Circular Buffer Configuration, Work Distribution, Data Flow, Component Sources, Compile-Time Arguments, Runtime Arguments |
| **ttnn-kernel-dataflow** | Kernel Data Movement, Memory Access Patterns (Reader/Writer sections), Component Sources (input/output stages) |
| **ttnn-kernel-compute** | Compute Access, Mathematical Definition, Component Sources (compute stage), CB ID Resolution |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

What behavior should be verified (agents decide how/when to test):

### Validation Behavior
- Wrong tensor rank (not 4D) -> error containing "must be 4D"
- Wrong layout (TILE_LAYOUT) -> error containing "must be in ROW_MAJOR layout"
- Wrong memory layout (SHARDED) -> error containing "must be interleaved"
- Unsupported dtype (INT32) -> error containing "unsupported dtype"
- Non-tile-aligned height -> error containing "must be a multiple of 32"
- Non-tile-aligned width -> error containing "must be a multiple of 32"

### Shape Behavior
- Input [N, C, H, W] -> Output [N, C, H, 32] (physical)
- Output logical shape interpretation: [N, C, H, 1]

### Functional Behavior
- Single tile row (H=32): output[:, :, :, 0] == mean(input, dim=-1)
- Multi-tile row: each output row is average of corresponding input row
- Numerical accuracy: within 1e-3 relative error vs PyTorch `torch.mean(input, dim=-1, keepdim=True)`
- Large width (W=1024): verify accumulation accuracy
- Multiple batches (N>1, C>1): verify batch independence

## Resolved Design Questions

1. **Output shape padding**: Return padded shape `[N, C, H, 32]`. The logical width is 1, but physical output is tile-aligned.

2. **FP32 accumulation**: Auto-enabled for FP32 inputs. When input dtype is FLOAT32, set `fp32_dest_acc_en = true` in compute config.

3. **Non-tile-aligned input support**: Tile alignment is required. Height and width must be multiples of 32. No automatic padding/unpadding.

## References
- Reference analyses:
  - `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_analysis.md`
  - `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_analysis.md`
  - `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_analysis.md`
- Kernel helper libraries:
  - `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp`
  - `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
- DeepWiki queries: (referenced in source analyses)
