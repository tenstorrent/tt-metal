# Centralize W RM Implementation Analysis

## Overview

The `centralize_w_rm` operation performs row-major width-wise centralization (subtracting the mean from each element along the W dimension). Given an input tensor, it computes `output[i] = input[i] - mean(input_row)` for each row.

**Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/centralize_w_rm/device/centralize_w_rm_program_factory.cpp`

This operation follows a four-phase pipeline per tile-row:
1. **Tilize**: Convert row-major sticks to tiled format
2. **Reduce (Mean)**: Compute row-wise mean using SUM with 1/W scaler
3. **Broadcast Subtract**: Subtract mean from each element (original - mean)
4. **Untilize**: Convert back to row-major format

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile-row |
| **Unit size** | Wt tiles (one tile-row = 32 sticks x W elements = Wt tiles) |
| **Total units** | Ht tile-rows |
| **Loop structure** | Outer loop over Ht tile-rows, each iteration processes all 4 phases |

One work unit is a **tile-row**: 32 consecutive row-major sticks that, when tilized, produce Wt tiles. Each tile-row is processed through the complete 4-phase pipeline before moving to the next.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, H, W] (collapsed to [H, W] for processing) |
| **Dimension convention** | Row-major with W as contiguous dimension |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 (configurable) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, H, W] (same as input, no dimension reduction) |
| **Dimension convention** | Row-major with W as contiguous dimension |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 (same as input) |

### Layout Transformations

1. **Tilize** (Phase 1): 32 RM sticks -> Wt tiles (ROW_MAJOR -> TILE_LAYOUT)
2. **Untilize** (Phase 4): Wt tiles -> 32 RM sticks (TILE_LAYOUT -> ROW_MAJOR)

## Data Flow Pattern

```
DRAM (RM sticks)
    |
    v  [Reader: read 32 sticks per tile-row]
CB_0 (cb_in_rm): RM sticks
    |
    v  [Compute: tilize]
CB_1 (cb_in_tiled): Wt tiled tiles [RETAINED for Phase 3]
    |
    +---> [Compute: reduce with PERSISTENT mode] ---> CB_3 (cb_mean_tiled): 1 mean tile
    |                                                     |
    |<----------------------------------------------------+
    |     [Compute: broadcast subtract (A=CB_1, B=CB_3)]
    v
CB_4 (cb_centralized_tiled): Wt centralized tiles
    |
    v  [Compute: untilize]
CB_16 (cb_out_rm): 32 RM output sticks
    |
    v  [Writer: write 32 sticks]
DRAM (RM sticks)
```

**Critical Pattern - CB_1 Retention**:
The key insight of this operation is that CB_1 (tiled input) must be **retained across phases 2 and 3**. The reduce operation uses `ReduceInputMode::PERSISTENT` which:
- Waits for all Wt tiles upfront
- Processes all tiles via indexed access
- Does **NOT** pop the tiles after processing

This allows the subsequent `sub` operation to read the same tiles from CB_1 for the broadcast subtract, without the reader needing to re-read the data.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in_rm | Input RM sticks | 2*Wt tiles | Wt tiles | Double | Reader | Compute (tilize) | Block |
| c_1 | cb_in_tiled | Tiled input | Wt tiles | Wt tiles | Single | Compute (tilize) | Compute (reduce, sub) | Block |
| c_2 | cb_scaler | Scaler (1/W) | 1 tile | 1 tile | Single | Reader | Compute (reduce) | Program |
| c_3 | cb_mean_tiled | Reduced mean | 1 tile | 1 tile | Single | Compute (reduce) | Compute (sub) | Block |
| c_4 | cb_centralized_tiled | Centralized tiles | Wt tiles | Wt tiles | Single | Compute (sub) | Compute (untilize) | Block |
| c_16 | cb_out_rm | Output RM sticks | 2*Wt tiles | Wt tiles | Double | Compute (untilize) | Writer | Block |

### CB Sizing Rationale

- **CB_0 (2*Wt)**: Double-buffered for reader/compute overlap
- **CB_1 (Wt)**: Single-buffered but must hold full tile-row for PERSISTENT mode
- **CB_2 (1)**: Single scaler tile, reused for entire program
- **CB_3 (1)**: Single mean tile output per tile-row
- **CB_4 (Wt)**: Full tile-row of centralized data
- **CB_16 (2*Wt)**: Double-buffered for compute/writer overlap

## Pipeline Pattern Summary

| CB | Capacity/Block | Buffering Type | Overlap Potential |
|----|----------------|----------------|-------------------|
| CB_0 | 2*Wt / Wt | Double | Reader/Compute overlap |
| CB_1 | Wt / Wt | Single | None (retained for reuse) |
| CB_2 | 1 / 1 | Single | None (persistent scaler) |
| CB_3 | 1 / 1 | Single | None (consumed immediately) |
| CB_4 | Wt / Wt | Single | None (consumed immediately) |
| CB_16 | 2*Wt / Wt | Double | Compute/Writer overlap |

## Index Calculations

### Reader Kernel
- **Stick indexing**: Linear `stick_id` from 0 to (H-1)
- **NoC address**: Uses `TensorAccessor::get_noc_addr(stick_id)` for DRAM interleaved access
- **L1 write**: Sequential within CB at `input_stick_size` stride

### Writer Kernel
- **Stick indexing**: Linear `stick_id` from 0 to (H-1), matching reader
- **NoC address**: Uses `TensorAccessor::get_noc_addr(stick_id)` for DRAM interleaved access
- **L1 read**: Sequential within CB at `output_stick_size` stride

### Compute Kernel (Reduce)
- **Tile indexing**: PERSISTENT mode uses indexed access `wt + index_offset` where `index_offset = 0` for single tile-row
- **CB_1 tiles**: Accessed as indices 0 to Wt-1

### Compute Kernel (Broadcast Subtract)
- **Input A (CB_1)**: Preloaded tiles accessed by index 0 to Wt-1
- **Input B (CB_3)**: Single mean tile, accessed at index 0, broadcast as COL

## Memory Access Patterns

### Read Pattern

| Stage | Pattern | Description |
|-------|---------|-------------|
| Reader | Sequential strided | 32 sticks per tile-row, each `input_stick_size` bytes |
| Tilize (CB_0->CB_1) | Block | tilize_block processes Wt tiles at once |
| Reduce (CB_1) | Indexed sequential | PERSISTENT: index 0 to Wt-1 |
| Sub (CB_1, CB_3) | A: Indexed, B: Single | A preloaded, B waits upfront |

### Write Pattern

| Stage | Pattern | Description |
|-------|---------|-------------|
| Tilize | Block | Wt tiles written to CB_1 |
| Reduce | Single | 1 mean tile to CB_3 |
| Sub | Block | Wt tiles to CB_4 |
| Untilize | Block | Wt pages to CB_16 |
| Writer | Sequential strided | 32 sticks, each `output_stick_size` bytes |

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Single core (initial implementation) |
| **Grid dimensions** | 1 x 1 |
| **Total cores** | 1 |
| **Work per core** | All Ht tile-rows |
| **Load balancing** | N/A (single core) |

Note: This is a single-core implementation. Multi-core distribution would split Ht tile-rows across cores.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_stick_size_aligned | uint32_t | NoC-aligned stick size in bytes |
| 1 | packed_scaler_value | uint32_t | Two bfloat16 (1/W) packed into uint32 |
| 2 | Ht | uint32_t | Height in tiles |
| 3 | Wt | uint32_t | Width in tiles |
| 4+ | TensorAccessorArgs | multiple | Buffer address mode, page size, etc. |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Height in tiles (outer loop count) |
| 1 | Wt | uint32_t | Width in tiles (tiles per row) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_stick_size_aligned | uint32_t | NoC-aligned stick size in bytes |
| 1 | Ht | uint32_t | Height in tiles |
| 2 | Wt | uint32_t | Width in tiles |
| 3+ | TensorAccessorArgs | multiple | Buffer address mode, page size, etc. |

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
| reader_centralize_w_rm | RISCV_0 (BRISC) | NOC0 | DRAM | CB_0, CB_2 | Read sticks, generate scaler |

**File**: `ttnn/cpp/ttnn/operations/reduction/centralize_w_rm/device/kernels/dataflow/reader_centralize_w_rm.cpp`

**Key Logic**:
1. **Phase 1 (once)**: Generate scaler tile using `generate_reduce_scaler(cb_scaler, packed_scaler_value)`
   - Creates a tile with 1/W value for mean calculation
   - Pushed to CB_2 (persistent for entire program)

2. **Phase 2 (per tile-row)**: Read 32 sticks into CB_0
   ```cpp
   for (ht = 0; ht < Ht; ++ht) {
       cb_reserve_back(cb_in_rm, Wt);  // Reserve Wt pages (tilize expects this)
       for (s = 0; s < 32; ++s) {      // 32 sticks per tile-row
           noc_async_read(accessor.get_noc_addr(stick_id), l1_write_addr, input_stick_size);
           l1_write_addr += input_stick_size;
           stick_id++;
       }
       noc_async_read_barrier();
       cb_push_back(cb_in_rm, Wt);     // Push Wt pages to match tilize wait
   }
   ```

**CB Synchronization**:
- Reserve/push Wt pages (not 32 sticks) because tilize helper expects tile count
- Memory equivalence: 32 sticks * W = Wt tiles * tile_size

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| centralize_w_rm_compute | RISCV_2,3,4 | N/A | CB_0,1,2,3 | CB_1,3,4,16 | tilize, reduce, sub, untilize |

**File**: `ttnn/cpp/ttnn/operations/reduction/centralize_w_rm/device/kernels/compute/centralize_w_rm_compute.cpp`

**Key Logic** (per tile-row):

```cpp
for (uint32_t block = 0; block < Ht; ++block) {
    // Phase 1: Tilize (32 RM sticks -> Wt tiles)
    compute_kernel_lib::tilize(cb_in_rm, Wt, cb_in_tiled, 1);

    // Phase 2: Reduce with PERSISTENT mode
    // - Waits for Wt tiles, does NOT pop them
    compute_kernel_lib::reduce<SUM, REDUCE_ROW, PERSISTENT>(
        cb_in_tiled, cb_scaler, cb_mean_tiled, TileShape::row(Wt));

    // Phase 3: Broadcast subtract
    // - A (cb_in_tiled): tiles still present from Phase 2
    // - B (cb_mean_tiled): COL broadcast (REDUCE_ROW produces column-shaped output)
    using PreloadedPopAtEnd = cb_policies::InputPolicy<WaitCallerManaged, PopAtEnd>;
    using WaitUpfrontPopAtEnd = cb_policies::InputPolicy<WaitUpfront, PopAtEnd>;
    compute_kernel_lib::sub<COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>(
        cb_in_tiled, cb_mean_tiled, cb_centralized_tiled, BinaryTileShape::row(Wt));

    // Phase 4: Untilize (Wt tiles -> 32 RM sticks)
    compute_kernel_lib::untilize<Wt, cb_centralized_tiled, cb_out_rm>(1);
}
```

**Critical CB Policy Details**:

1. **Reduce PERSISTENT mode**:
   - `cb_wait_front(cb_in_tiled, Wt)` at start
   - NO `cb_pop_front` - tiles remain in CB_1

2. **Sub Input A Policy (`PreloadedPopAtEnd`)**:
   - `WaitCallerManaged`: No wait (tiles already there from reduce)
   - `PopAtEnd`: Pop all Wt tiles after processing

3. **Sub Input B Policy (`WaitUpfrontPopAtEnd`)**:
   - `WaitUpfront`: Wait for 1 mean tile
   - `PopAtEnd`: Pop after all A tiles processed

4. **Why COL broadcast after REDUCE_ROW**:
   - REDUCE_ROW reduces along width -> output has valid data in column 0 only
   - COL broadcast takes column 0 and replicates it across all columns
   - This applies the single mean value to every element in the row

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_centralize_w_rm | RISCV_1 (NCRISC) | NOC1 | CB_16 | DRAM | Write sticks |

**File**: `ttnn/cpp/ttnn/operations/reduction/centralize_w_rm/device/kernels/dataflow/writer_centralize_w_rm.cpp`

**Key Logic**:
```cpp
for (ht = 0; ht < Ht; ++ht) {
    cb_wait_front(cb_out_rm, Wt);     // Wait for Wt pages from untilize
    for (s = 0; s < 32; ++s) {        // 32 sticks per tile-row
        noc_async_write(l1_read_addr, accessor.get_noc_addr(stick_id), output_stick_size);
        l1_read_addr += output_stick_size;
        stick_id++;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out_rm, Wt);      // Release Wt pages
}
```

## Implementation Notes

### Kernel Helper Usage Summary

| Helper | Location | Purpose | Key Parameters |
|--------|----------|---------|----------------|
| `tilize()` | tilize_helpers.hpp | RM->Tile conversion | cb_in, Wt, cb_out, num_blocks=1 |
| `reduce<SUM, REDUCE_ROW, PERSISTENT>()` | reduce_helpers.hpp | Row-wise mean | TileShape::row(Wt), keeps tiles in CB |
| `sub<COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>()` | binary_op_helpers.hpp | Broadcast subtract | BinaryTileShape::row(Wt) |
| `untilize<Wt, cb_in, cb_out>()` | untilize_helpers.hpp | Tile->RM conversion | num_rows=1 |
| `generate_reduce_scaler()` | generate_reduce_scaler.hpp | Create 1/W scaler | packed bfloat16 value |

### Why REDUCE_ROW + BroadcastDim::COL

This seemingly counterintuitive pairing makes sense when you understand the semantics:

1. **REDUCE_ROW** means "reduce along the row direction" (i.e., sum across width W)
2. The output is a **single column** of values (one per row in H dimension)
3. Tile representation: output has valid data only in column 0 of the tile
4. **BroadcastDim::COL** takes column 0 and replicates it across all columns
5. Result: the mean value is subtracted from every element in the original row

### Scaler Value Computation

```cpp
const float scaler_value = 1.0f / static_cast<float>(W);
bfloat16 bfloat_scaler_value = bfloat16::truncate(scaler_value);
const uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
```

The scaler is pre-computed on the host and embedded as a compile-time argument. The reader kernel uses `generate_reduce_scaler()` to create a tile filled with this value.

### Memory Efficiency

- CB_1 reuse avoids re-reading Wt tiles from DRAM
- PERSISTENT mode enables this pattern with minimal CB capacity
- Double buffering on CB_0 and CB_16 allows overlap with DRAM transfers

### Extension to Variance

For `variance_w_rm`, the pattern extends to:
1. **Centralize** (as above): output = input - mean
2. **Square**: squared = (input - mean)^2
3. **Reduce (Variance)**: variance = mean(squared)

The centralize_w_rm pattern provides the foundation, and variance adds a square operation between phases 3 and 4.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the purpose and behavior of the generate_reduce_scaler function in dataflow kernels?"
   **Reason**: Needed to understand how the scaler tile is created for reduce operations
   **Key Findings**: `generate_reduce_scaler` creates a tile filled with a scalar value (1/W for mean). It reserves CB space, zero-fills the tile, then writes the scalar to specific positions within the tile structure.

2. **Query**: "What is BroadcastDim::COL and why is it used after REDUCE_ROW?"
   **Reason**: The compute kernel uses COL broadcast after ROW reduction, which seems counterintuitive
   **Key Findings**: REDUCE_ROW produces column-shaped output (valid in column 0). BroadcastDim::COL then replicates column 0 across all columns. This is the correct pairing - the reduce dimension and broadcast dimension are complementary.

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding circular buffer semantics and tile-based operations
   **Key Information**: CB operations (reserve_back, push_back, wait_front, pop_front) for producer-consumer synchronization

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp`
   **Reason**: Understanding ReduceInputMode::PERSISTENT semantics
   **Key Information**: PERSISTENT mode waits for all tiles upfront but does NOT pop them, allowing subsequent operations to reuse the same data

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`
   **Reason**: Understanding CB policies for broadcast subtract
   **Key Information**: Custom InputPolicy combinations (WaitCallerManaged + PopAtEnd) enable preloaded tile reuse patterns

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/cb_policies.hpp`
   **Reason**: Understanding the policy type system
   **Key Information**: Policies separate timing/pattern (when to wait/pop) from tile counts (derived from broadcast dimension)
