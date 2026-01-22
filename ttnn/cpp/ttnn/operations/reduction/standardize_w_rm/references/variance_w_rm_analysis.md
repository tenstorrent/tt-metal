# variance_w_rm Implementation Analysis

## Overview

The `variance_w_rm` operation computes the population variance along the width (W) dimension of a row-major input tensor. This is a reduction operation that transforms an input of shape `[..., H, W]` to an output of shape `[..., H, 1]` (logical) padded to `[..., H, 32]` (physical).

**Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/variance_w_rm/device/variance_w_rm_program_factory.cpp`

### Mathematical Definition

```
mean[..., 0] = (1/W) * sum(input[..., j] for j in range(W))
centralized[..., j] = input[..., j] - mean[..., 0]  for all j in range(W)
squared[..., j] = centralized[..., j]^2  for all j in range(W)
variance[..., 0] = (1/W) * sum(squared[..., j] for j in range(W))
```

The operation uses **population variance** (divide by N, not N-1), which is standard for neural network operations like batch normalization.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile-row |
| **Unit size** | Wt tiles (one tile-row = 32 sticks x W elements = Wt tiles when tilized) |
| **Total units** | Ht tile-rows |
| **Loop structure** | Outer loop over Ht tile-rows, each iteration processes all 6 phases |

One work unit is a **tile-row**: 32 consecutive row-major sticks that, when tilized, produce Wt tiles. Each tile-row is processed through the complete 6-phase pipeline before moving to the next.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | `[..., H, W]` (at least 2D) |
| **Dimension convention** | Last dimension is W (reduced dimension), second-to-last is H |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | `[..., H, 1]` (last dimension reduced to 1) |
| **Padded shape** | `[..., H, 32]` (padded to tile boundary) |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or as specified in memory_config) |
| **Data type** | Same as input |

### Layout Transformations

1. **Input Read**: Row-major sticks read from DRAM (32 sticks per tile-row, each stick is W elements)
2. **Tilize**: Converts 32 sticks into Wt tiled tiles
3. **Compute**: Operates on tiles through 6 phases
4. **Untilize**: Converts 1 variance tile back to 32 row-major sticks (width=32)
5. **Output Write**: Row-major sticks written to DRAM (32 sticks, each stick is 32 elements)

## Data Flow Pattern

```
DRAM (RM sticks, shape [..., W])
    |
    v  [Reader: read 32 sticks per tile-row, generate scaler once at start]
CB_0 (cb_in_rm): RM sticks (2*Wt pages, double-buffered)
    |
    v  [Compute Phase 1: tilize]
CB_1 (cb_in_tiled): Wt tiled tiles [RETAINED for Phase 3 via PERSISTENT mode]
    |
    +---> [Compute Phase 2: reduce PERSISTENT] ---> CB_3 (cb_mean): 1 mean tile
    |                                                       |
    |<------------------------------------------------------+
    |     [Compute Phase 3: broadcast subtract (A=CB_1, B=CB_3)]
    v
CB_4 (cb_centralized): Wt centralized tiles
    |
    v  [Compute Phase 4: square (SQUARE binary op)]
CB_5 (cb_squared): Wt squared tiles
    |
    +---> [Compute Phase 5: reduce STREAMING] ---> CB_6 (cb_variance): 1 variance tile
    |
    v  [Compute Phase 6: untilize (1 tile -> 32 sticks)]
CB_16 (cb_out_rm): 32 RM output sticks (width=32, padded)
    |
    v  [Writer: write 32 sticks to reduced output]
DRAM (RM sticks, shape [..., 32] padded)
```

### Phase-by-Phase Data Flow

| Phase | Operation | Input CB(s) | Output CB | Description |
|-------|-----------|-------------|-----------|-------------|
| 1 | Tilize | CB_0 (Wt pages) | CB_1 (Wt tiles) | Convert 32 RM sticks to Wt tiles |
| 2 | Reduce (Mean) | CB_1 (Wt tiles), CB_2 (scaler) | CB_3 (1 tile) | REDUCE_ROW with PERSISTENT mode |
| 3 | Broadcast Sub | CB_1 (Wt tiles), CB_3 (1 tile) | CB_4 (Wt tiles) | Centralize: input - mean |
| 4 | Square | CB_4 (Wt tiles) | CB_5 (Wt tiles) | Element-wise (x-mean)^2 |
| 5 | Reduce (Variance) | CB_5 (Wt tiles), CB_2 (scaler) | CB_6 (1 tile) | REDUCE_ROW with STREAMING mode |
| 6 | Untilize | CB_6 (1 tile) | CB_16 (1 tile worth of sticks) | Convert 1 tile to 32 RM sticks |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in_rm | Input RM sticks | 2*Wt tiles | Wt tiles | Double | Reader | Compute (tilize) | Block |
| c_1 | cb_in_tiled | Tiled input (persists for bcast_sub) | Wt tiles | Wt tiles | Single | Compute (tilize) | Compute (reduce, sub) | Block |
| c_2 | cb_scaler | Scaler (1/W) for both reduces | 1 tile | 1 tile | Single | Reader | Compute (reduce x2) | Program |
| c_3 | cb_mean_tiled | Mean tile output from first reduce | 1 tile | 1 tile | Single | Compute (reduce1) | Compute (sub) | Block |
| c_4 | cb_centralized_tiled | Centralized tiles (input - mean) | Wt tiles | Wt tiles | Single | Compute (sub) | Compute (square) | Block |
| c_5 | cb_squared_tiled | Squared tiles ((x-mean)^2) | Wt tiles | Wt tiles | Single | Compute (square) | Compute (reduce2) | Block |
| c_6 | cb_variance_tiled | Variance tile from second reduce | 1 tile | 1 tile | Single | Compute (reduce2) | Compute (untilize) | Block |
| c_16 | cb_out_rm | Output RM sticks | 2 tiles | 1 tile | Double | Compute (untilize) | Writer | Block |

### CB Sizing Rationale

- **CB_0 (2*Wt)**: Double-buffered for reader/compute overlap. Reader fills one half while compute processes the other.
- **CB_1 (Wt)**: Single-buffered but must hold full tile-row for PERSISTENT reduce mode. Tiles must persist after reduce for the subsequent broadcast subtract phase.
- **CB_2 (1)**: Single persistent scaler tile containing 1/W, reused by both reduce operations throughout the program.
- **CB_3 (1)**: Single mean tile per tile-row, consumed immediately by broadcast subtract.
- **CB_4 (Wt)**: Full tile-row of centralized data, consumed by square operation.
- **CB_5 (Wt)**: Full tile-row of squared data, consumed by second reduce.
- **CB_6 (1)**: Single variance tile per tile-row, consumed by untilize.
- **CB_16 (2)**: Double-buffered, but only 1 tile output per tile-row (reduced width). The double buffering enables compute/writer overlap.

## Pipeline Pattern Summary

Based on the CB configurations:

| CB | Capacity | Block Size | Buffering Type | Overlap Potential |
|----|----------|------------|----------------|-------------------|
| CB_0 | 2*Wt | Wt | Double | Reader/Compute overlap |
| CB_1 | Wt | Wt | Single | No overlap (PERSISTENT retention) |
| CB_2 | 1 | 1 | Single | Program-lifetime (no overlap needed) |
| CB_3 | 1 | 1 | Single | No overlap (immediate consumption) |
| CB_4 | Wt | Wt | Single | No overlap |
| CB_5 | Wt | Wt | Single | No overlap |
| CB_6 | 1 | 1 | Single | No overlap |
| CB_16 | 2 | 1 | Double | Compute/Writer overlap |

The double-buffering on CB_0 and CB_16 enables pipelining between:
- Reader and Compute (via CB_0)
- Compute and Writer (via CB_16)

The intermediate CBs (CB_1 through CB_6) are single-buffered because they represent sequential dependencies within the compute kernel's 6-phase pipeline.

## Index Calculations

### Input Tensor Access (Reader Kernel)

The reader uses `TensorAccessor` pattern for mapping logical stick indices to physical memory:

```cpp
constexpr auto tensor_args = TensorAccessorArgs<4>();  // 4 compile-time args follow
const auto accessor = TensorAccessor(tensor_args, src_addr, input_stick_size);

// For each tile-row (ht in [0, Ht)):
//   For each stick in tile-row (s in [0, 32)):
//     stick_id = ht * 32 + s
uint64_t noc_addr = accessor.get_noc_addr(stick_id);
```

- **stick_id**: Linear stick index starting from 0
- **TensorAccessor**: Handles interleaved DRAM bank mapping internally

### Output Tensor Access (Writer Kernel)

```cpp
constexpr auto tensor_args = TensorAccessorArgs<2>();  // 2 compile-time args follow
const auto accessor = TensorAccessor(tensor_args, dst_addr, output_stick_size);

// Output sticks have same structure but narrower width (32 elements)
// stick_id still linear: ht * 32 + s
uint64_t noc_addr = accessor.get_noc_addr(stick_id);
```

### Tile Indexing in Compute

Within compute phases:
- **Tilize**: Processes Wt tiles per tile-row
- **Reduce PERSISTENT**: Indexed tile access (0 to Wt-1), tiles retained after operation
- **Broadcast Sub**: Input A uses indices 0 to Wt-1 (preloaded), Input B uses index 0 (single mean tile)
- **Square**: Uses same CB for both inputs (self-multiply pattern), indices 0 to Wt-1
- **Reduce STREAMING**: One-at-a-time tile processing, automatic popping
- **Untilize**: Single tile (index 0)

## Memory Access Patterns

### Read Pattern (Reader Kernel)

| Aspect | Description |
|--------|-------------|
| **Pattern** | Sequential stick access |
| **Memory** | DRAM (interleaved across banks) |
| **Granularity** | One stick at a time, 32 sticks per tile-row |
| **Stick size** | W * element_size (aligned to NoC requirements) |
| **Blocking** | Push Wt pages after reading all 32 sticks |

```cpp
for (uint32_t ht = 0; ht < Ht; ++ht) {
    cb_reserve_back(cb_in_rm, Wt);
    uint32_t l1_write_addr = get_write_ptr(cb_in_rm);
    for (uint32_t s = 0; s < 32; ++s) {
        uint64_t noc_addr = accessor.get_noc_addr(stick_id);
        noc_async_read(noc_addr, l1_write_addr, input_stick_size);
        l1_write_addr += input_stick_size;
        stick_id++;
    }
    noc_async_read_barrier();
    cb_push_back(cb_in_rm, Wt);
}
```

### Write Pattern (Writer Kernel)

| Aspect | Description |
|--------|-------------|
| **Pattern** | Sequential stick access |
| **Memory** | DRAM (interleaved across banks) |
| **Granularity** | One stick at a time, 32 sticks per tile-row |
| **Stick size** | 32 * element_size (reduced width, aligned) |
| **Blocking** | Wait for 1 tile, write 32 sticks, pop 1 tile |

```cpp
for (uint32_t ht = 0; ht < Ht; ++ht) {
    cb_wait_front(cb_out_rm, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_out_rm);
    for (uint32_t s = 0; s < 32; ++s) {
        uint64_t noc_addr = accessor.get_noc_addr(stick_id);
        noc_async_write(l1_read_addr, noc_addr, output_stick_size);
        l1_read_addr += output_stick_size;
        stick_id++;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out_rm, 1);
}
```

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (single core) |
| **Grid dimensions** | 1 x 1 |
| **Total cores** | 1 |
| **Work per core** | All Ht tile-rows |
| **Load balancing** | N/A (single core implementation) |

Current implementation uses a single core. Future multi-core extension would:
- Split Ht tile-rows across cores
- Use `tt::tt_metal::split_work_to_cores()` for distribution
- Each core processes a subset of tile-rows independently

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_stick_size_aligned | uint32_t | NoC-aligned stick size in bytes (W * element_size, rounded up) |
| 1 | packed_scaler_value | uint32_t | Two bfloat16 (1/W) packed into uint32 for reduce scaler |
| 2 | Ht | uint32_t | Height in tiles (number of tile-rows to process) |
| 3 | Wt | uint32_t | Width in tiles (tiles per row) |
| 4+ | TensorAccessorArgs | multiple | Input buffer address mode, page size, etc. |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Height in tiles (outer loop count) |
| 1 | Wt | uint32_t | Width in tiles (tiles per row, used by tilize, reduce, sub, square) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_stick_size_aligned | uint32_t | NoC-aligned output stick size (32 * element_size, rounded up) |
| 1 | Ht | uint32_t | Height in tiles (number of tile-rows) |
| 2+ | TensorAccessorArgs | multiple | Output buffer address mode, page size, etc. |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input DRAM buffer address |

#### Compute Kernel

None (all parameters are compile-time).

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output DRAM buffer address |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_variance_w_rm | RISCV_0 (BRISC) | NOC0 | DRAM (RM sticks) | CB_0, CB_2 | Read 32 sticks per tile-row, generate 1/W scaler once |

**File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/variance_w_rm/device/kernels/dataflow/reader_variance_w_rm.cpp`

**Key Logic**:
1. **Scaler Generation** (once at start): Uses `generate_reduce_scaler(cb_scaler, packed_scaler_value)` to fill a tile with the 1/W scaler value. This tile persists for the entire program and is used by both reduce operations.
2. **Stick Reading** (per tile-row): Reads 32 row-major sticks into CB_0 using TensorAccessor for DRAM address calculation. Each stick is W elements wide. After reading all 32 sticks, pushes Wt pages to signal data ready for tilize.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| variance_w_rm_compute | RISCV_2,3,4 | N/A | CB_0,1,2,3,4,5 | CB_1,3,4,5,6,16 | 6-phase pipeline: tilize, reduce, sub, square, reduce, untilize |

**File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/variance_w_rm/device/kernels/compute/variance_w_rm_compute.cpp`

**Key Logic**:

The compute kernel uses helper libraries from `ttnn/cpp/ttnn/kernel_lib/` for all operations:

**Phase 1 - Tilize**:
```cpp
compute_kernel_lib::tilize(cb_in_rm, Wt, cb_in_tiled, 1);
```
- Converts 32 RM sticks from CB_0 to Wt tiles in CB_1
- Helper handles all CB wait/reserve/push/pop

**Phase 2 - Reduce (Mean) with PERSISTENT mode**:
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputMode::PERSISTENT>(
    cb_in_tiled, cb_scaler, cb_mean_tiled,
    compute_kernel_lib::TileShape::row(Wt));
```
- REDUCE_ROW with SUM and 1/W scaler computes mean
- **PERSISTENT mode**: Waits for all Wt tiles but does NOT pop them
- This keeps tiles in CB_1 available for subsequent broadcast subtract

**Phase 3 - Broadcast Subtract (Centralize)**:
```cpp
compute_kernel_lib::sub<compute_kernel_lib::BroadcastDim::COL,
    PreloadedPopAtEnd, WaitUpfrontPopAtEnd>(
    cb_in_tiled, cb_mean_tiled, cb_centralized_tiled,
    compute_kernel_lib::BinaryTileShape::row(Wt));
```
- BroadcastDim::COL: Mean tile (1 tile) broadcasts across columns to match Wt tiles
- PreloadedPopAtEnd for Input A (CB_1): Tiles already present from PERSISTENT reduce, no wait needed
- WaitUpfrontPopAtEnd for Input B (CB_3): Wait for mean tile upfront
- Computes centralized = input - mean

**Phase 4 - Square**:
```cpp
compute_kernel_lib::binary_op<
    compute_kernel_lib::BinaryOpType::SQUARE,
    compute_kernel_lib::BroadcastDim::NONE,
    WaitUpfrontPopAtEnd>(
    cb_centralized_tiled, cb_centralized_tiled, cb_squared_tiled,
    compute_kernel_lib::BinaryTileShape::row(Wt));
```
- SQUARE binary op type: Handles self-multiply pattern (A*A) internally
- Both icb_a and icb_b point to same CB (CB_4), helper handles this

**Phase 5 - Reduce (Variance) with STREAMING mode**:
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputMode::STREAMING>(
    cb_squared_tiled, cb_scaler, cb_variance_tiled,
    compute_kernel_lib::TileShape::row(Wt));
```
- Same scaler (1/W) reused for mean of squared values = variance
- STREAMING mode: Processes and pops tiles one at a time (no persistence needed)

**Phase 6 - Untilize**:
```cpp
compute_kernel_lib::untilize<1, cb_variance_tiled, cb_out_rm>(1);
```
- Converts 1 variance tile to 32 RM sticks (width=32 elements each)
- Helper auto-selects pack_untilize or standard untilize based on width

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_variance_w_rm | RISCV_1 (NCRISC) | NOC1 | CB_16 | DRAM (reduced RM sticks) | Write 32 sticks (width=32) per tile-row |

**File**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/variance_w_rm/device/kernels/dataflow/writer_variance_w_rm.cpp`

**Key Logic**:
- Waits for 1 tile in CB_16 (represents 32 sticks of width 32)
- Writes 32 output sticks sequentially using TensorAccessor
- Output sticks are narrower than input (32 elements vs W elements)
- Pops 1 tile after writing all 32 sticks

## Implementation Notes

### Scaler Reuse Pattern
The 1/W scaler in CB_2 is generated once at program start and reused by both reduce operations. This is efficient because:
- Both operations compute a mean (mean of values, mean of squared deviations)
- The scaler tile persists for the program lifetime (CB_2 is never popped)

### PERSISTENT vs STREAMING Reduce Modes
- **PERSISTENT (Phase 2)**: Used because tiles need to persist in CB_1 for the subsequent broadcast subtract. The reduce helper waits for all tiles but does NOT pop them.
- **STREAMING (Phase 5)**: Used because squared tiles in CB_5 are not needed after variance reduction. Tiles are popped as processed, one at a time.

### BroadcastDim::COL with REDUCE_ROW
REDUCE_ROW reduces along the width dimension, producing a column-shaped output (1 tile per tile-row). When applying this result back via broadcast subtract, BroadcastDim::COL is used because:
- The mean tile has valid data only in column 0
- COL broadcast replicates this column across all columns of the input tiles

### CB Policy Types for Binary Operations
The compute kernel uses policy-based CB management:
- `PreloadedPopAtEnd`: Tiles already in CB (from PERSISTENT mode), no wait needed, pop all at end
- `WaitUpfrontPopAtEnd`: Wait for all tiles upfront, pop all at end

### Double Buffering Strategy
Only CB_0 (input) and CB_16 (output) use double buffering:
- Enables reader to fill next tile-row while compute processes current
- Enables compute to fill next output while writer processes current
- Intermediate CBs are single-buffered due to sequential phase dependencies

### Output Width Reduction
The key transformation is input width W to output width 32:
- Input: W elements per stick (could be large)
- Output: 32 elements per stick (single tile width, padded)
- This is fundamental to reduction operations - the reduced dimension becomes a single tile

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the TensorAccessor pattern in tt-metal and how does it work with TensorAccessorArgs for reading tensor data in kernels?"
   **Reason**: Needed to understand how the reader/writer kernels map logical stick indices to physical DRAM addresses
   **Key Findings**: TensorAccessor abstracts physical memory layout and bank distribution. TensorAccessorArgs passes tensor metadata from host to device, supporting both compile-time and runtime argument configurations.

2. **Query**: "What is the REDUCE_ROW operation in compute kernels and why does it output a column-shaped result (Ht tiles) when reducing along the row dimension?"
   **Reason**: Needed to understand the REDUCE_ROW semantics and why BroadcastDim::COL is paired with it
   **Key Findings**: REDUCE_ROW collapses rows of input tiles, producing a 1xW result per tile which is represented as column-shaped output. "REDUCE_ROW" means "reduce along row direction" = sum across width = column output.

3. **Query**: "What is the PERSISTENT mode in reduce operations and how does it differ from STREAMING mode in tt-metal compute kernels?"
   **Reason**: Needed to understand why Phase 2 uses PERSISTENT and Phase 5 uses STREAMING
   **Key Findings**: PERSISTENT mode retains input tiles after reduction for subsequent operations (like broadcast subtract). STREAMING mode processes and pops tiles one at a time. The variance_w_rm operation leverages PERSISTENT to keep tilized input for centralization.

4. **Query**: "What is the BroadcastDim::COL broadcast operation and when is it used together with REDUCE_ROW in compute kernels?"
   **Reason**: Needed to understand the relationship between REDUCE_ROW output shape and broadcast dimension
   **Key Findings**: BroadcastDim::COL broadcasts a single column across all columns during binary operations. It pairs naturally with REDUCE_ROW because the reduction produces column-shaped output that needs to be applied back to the original data.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp`
   **Reason**: Understanding reduce helper implementation and PERSISTENT vs STREAMING modes
   **Key Information**: ReduceInputMode enum defines STREAMING (one-at-a-time with automatic pop), STREAMING_BATCHED (bulk), PRELOADED (caller manages), and PERSISTENT (wait upfront, no pop). The helper manages all CB operations and DST register handling internally.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`
   **Reason**: Understanding broadcast subtract implementation and BroadcastDim
   **Key Information**: BroadcastDim specifies B operand shape and how it broadcasts. COL means B has shape [Ht, 1] and replicates right. After REDUCE_ROW (which produces column output), use BroadcastDim::COL to apply the result back.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
   **Reason**: Understanding tilize helper and how it converts RM sticks to tiles
   **Key Information**: Single unified function handles all tilize variants. Takes block_w (tiles per row) and num_blocks. Handles all CB wait/reserve/push/pop operations internally.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
   **Reason**: Understanding untilize helper and dispatch logic
   **Key Information**: Automatically dispatches between pack_untilize (hardware-accelerated) and standard untilize based on width and data format. For small widths (<=DEST limit), uses pack_untilize.

5. **Source**: `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp`
   **Reason**: Understanding how the 1/W scaler tile is generated
   **Key Information**: Fills a tile with zeros, then writes the packed scaler value to specific positions. The scaler is a bfloat16 value (1/W) packed twice into a uint32.

6. **Source**: `variance_w_rm_spec.md`
   **Reason**: Understanding the operation specification and design decisions
   **Key Information**: Detailed specification including 6-phase pipeline design, CB sizing rationale, and comparison with centralize_w_rm reference operation.

## Relevance for standardize_w_rm

The standardize_w_rm operation will extend variance_w_rm with:
1. **Epsilon parameter**: Added to variance for numerical stability
2. **rsqrt computation**: rsqrt(variance + epsilon) to get inverse standard deviation
3. **Final multiplication**: centralized * rsqrt_result to produce normalized output
4. **Output shape preservation**: Output same shape as input (not reduced)

Key reusable components:
- Phases 1-4 (tilize, mean reduce, centralize, square) are identical
- CB_0 through CB_4 configuration can be reused
- Reader kernel largely unchanged
- Scaler generation pattern for 1/W

Key modifications needed:
- Add CB for rsqrt result (1 tile)
- Add epsilon to variance before rsqrt
- Add rsqrt operation (or compute via recip(sqrt()))
- Add broadcast multiply (centralized * rsqrt)
- Modify output CB sizing (Wt tiles per tile-row, not 1)
- Modify writer to output full-width sticks

The PERSISTENT mode pattern from Phase 2 may need to be extended to Phase 3 as well, since centralized tiles (CB_4) must persist for the final broadcast multiply after rsqrt is computed.
