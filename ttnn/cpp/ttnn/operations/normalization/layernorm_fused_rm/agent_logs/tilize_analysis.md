# Tilize Multi-Core Interleaved Implementation Analysis

## Overview

The **tilize** operation transforms row-major (RM) tensor data into tiled (32x32) format suitable for Tenstorrent hardware compute operations. This analysis covers the multi-core interleaved variant, which reads row-major data from interleaved DRAM and writes tiled output to interleaved memory.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

### Core Concept

Tilization groups contiguous row-major elements into 32x32 tiles. Each tile is further divided into four 16x16 "faces" stored contiguously in memory for efficient matrix engine access.

```
Row-Major Input (logical view):
Row 0:  [e0,  e1,  e2,  ... e31, e32, e33, ... e63]
Row 1:  [e64, e65, e66, ... e95, e96, e97, ... e127]
...
Row 31: [...]

Tiled Output (per 32x32 tile):
Face 0 (16x16) -> Face 1 (16x16) -> Face 2 (16x16) -> Face 3 (16x16)
[rows 0-15, cols 0-15] [rows 0-15, cols 16-31] [rows 16-31, cols 0-15] [rows 16-31, cols 16-31]
```

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (tile row) |
| **Unit size** | `ntiles_per_block` tiles (one tile-row of output) |
| **Total units** | `nblocks = ceil(ntiles / ntiles_per_block)` |
| **Loop structure** | Each core processes `nblocks_per_core` blocks. Each block = 32 input sticks -> 1 tile-row output |

A **block** represents:
- **Input**: 32 contiguous sticks (rows) spanning the full tensor width
- **Output**: One row of tiles (`ntiles_per_block` tiles), where `ntiles_per_block = padded_width / TILE_WIDTH`

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, ...outer_dims..., H, W] |
| **Dimension convention** | Last dim = width (contiguous in memory) |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typically) |
| **Data type** | BFLOAT16 (or FLOAT32) |
| **Page definition** | One stick (row) = W elements |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input (with padding to tile alignment) |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typically) |
| **Data type** | Same as input |
| **Page definition** | One tile = 32x32 elements |

### Key Calculations

```cpp
// From program factory:
ntiles = physical_volume / TILE_HW;           // Total output tiles
ntiles_per_block = padded_width / TILE_WIDTH; // Tiles per tile-row
nblocks = ceil(ntiles / ntiles_per_block);    // Total tile-rows
block_size_nbytes = padded_width * element_size; // Bytes per stick
```

---

## Data Flow Pattern

```
Stage 1: Reader Kernel (RISCV_0 / NoC0)
------------------------------------------------------------
For each block (tile_height=32 sticks):
  1. Compute NoC addresses for 32 consecutive sticks
  2. For each width-block in row:
     a. cb_reserve_back(cb_in0, ntiles_per_block)
     b. Read 32 sticks sequentially into CB (row-major layout)
     c. noc_async_read_barrier()
     d. cb_push_back(cb_in0, ntiles_per_block)

Stage 2: Compute Kernel (UNPACK/MATH/PACK RISC-Vs)
------------------------------------------------------------
For each block:
  1. cb_wait_front(cb_in0, ntiles_per_block)
  2. cb_reserve_back(cb_out, ntiles_per_block)
  3. tilize_block(cb_in0, ntiles_per_block, cb_out)
     - Unpacker reads 32 RM rows, reorders to tile format
     - Math datacopy moves data through pipeline
     - Packer writes tiled data to output CB
  4. cb_push_back(cb_out, ntiles_per_block)
  5. cb_pop_front(cb_in0, ntiles_per_block)

Stage 3: Writer Kernel (RISCV_1 / NoC1)
------------------------------------------------------------
For each tile in output:
  1. cb_wait_front(cb_out, 1)
  2. noc_async_write_page(tile_id, accessor, l1_addr)
  3. noc_async_write_barrier()
  4. cb_pop_front(cb_out, 1)
```

### Data Flow Diagram

```
DRAM (Interleaved)           L1 Circular Buffers              DRAM (Interleaved)
     |                              |                               ^
     | 32 sticks                    |                               |
     | (row-major)                  |                               | tiles
     v                              v                               |
[src_buffer] --NoC0--> [CB c_0] --tilize--> [CB c_16] --NoC1--> [dst_buffer]
  (RM pages)           (RM input)            (tiled output)      (tile pages)
```

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input staging (RM sticks) | ntiles_per_block tiles | ntiles_per_block tiles | Single | Reader | Compute | Block |
| c_16 | cb_out | Output staging (tiled) | ntiles_per_block tiles | ntiles_per_block tiles | Single | Compute | Writer | Block |

### CB Sizing Details

```cpp
// Input CB (c_0): sized to hold one block of row-major data
//   - Capacity: ntiles_per_block * input_single_tile_size bytes
//   - Conceptually holds 32 sticks * width elements, but allocated as "tiles"
//     because tilize_block expects tile-sized pages in CB semantics

// Output CB (c_16): sized to hold one row of output tiles
//   - Capacity: ntiles_per_block * output_single_tile_size bytes
//   - Each tile = 32*32 * element_size bytes
```

**Note**: The input CB is somewhat unconventional - it stores row-major data but is sized in "tile units" because the tilize operation expects the CB to contain enough data for a complete tilization (32 rows).

---

## Pipeline Pattern Summary

| Pattern | Description |
|---------|-------------|
| **Buffering Type** | Single-buffered (capacity = block size for both CBs) |
| **Overlap Potential** | Limited - single buffering means reader must complete before compute can start |
| **Pipeline Depth** | 1 block in flight per stage |

The single-buffered configuration means:
- Reader fills CB c_0, then waits
- Compute processes CB c_0 to CB c_16, then waits
- Writer drains CB c_16, then waits

For higher throughput, double-buffering could be considered.

---

## Index Calculations

### Reader: Stick-to-NoC Address Mapping

```cpp
// TensorAccessor handles interleaved bank distribution
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);

// For each block of 32 sticks:
for (uint32_t j = 0; j < tile_height; j++) {
    base_src_noc_addr[j] = get_noc_addr(stick_id, s);
    stick_id++;
}
```

The `TensorAccessor` with interleaved configuration:
1. Takes `stick_id` (logical page index)
2. Computes `bank_id = stick_id % num_banks` (round-robin distribution)
3. Computes `bank_offset = (stick_id / num_banks) * stick_size`
4. Returns 64-bit NoC address encoding bank coordinates + offset

### Compute: Tilize Block Index

The `tilize_block` function internally:
1. Reads 32 rows from input CB (row-major format)
2. Reorders elements into tile format (face0, face1, face2, face3)
3. Writes tiled output to destination CB

### Writer: Tile-to-NoC Address Mapping

```cpp
const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

for (uint32_t i = start_id; i < end_id; ++i) {
    noc_async_write_page(i, s, l1_read_addr);
}
```

Output tiles are written sequentially starting from `tile_start_id`.

---

## Memory Access Patterns

### Read Pattern

| Attribute | Value |
|-----------|-------|
| **Pattern Type** | Sequential with stride |
| **Granularity** | Full stick (padded width) |
| **Access Order** | Row-major: stick 0, stick 1, ..., stick 31 for each block |
| **Bank Distribution** | Interleaved round-robin across DRAM banks |
| **Reads per Block** | 32 NoC reads (one per stick) |

```cpp
// Read pattern within read_tiles lambda:
for (uint32_t k = 0; k < tile_height; k++) {
    noc_async_read(base_src_noc_addr[k], l1_write_addr, width_size);
    l1_write_addr += width_size;  // Sequential L1 placement
}
```

### Write Pattern

| Attribute | Value |
|-----------|-------|
| **Pattern Type** | Sequential |
| **Granularity** | Single tile (32x32 elements) |
| **Access Order** | Linear tile index: tile 0, tile 1, ..., tile N-1 |
| **Bank Distribution** | Interleaved round-robin across DRAM banks |
| **Writes per Block** | ntiles_per_block NoC writes |

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from available grid) |
| **Grid dimensions** | Variable based on `sub_core_grids` or `compute_with_storage_grid_size()` |
| **Total cores** | `ncores` (computed by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (or `nblocks_per_core_cliff` for last core) |
| **Load balancing** | Equal + cliff (last core handles remainder) |

### Work Split Algorithm

```cpp
auto [ncores, all_cores, core_range, core_range_cliff,
      nblocks_per_core, nblocks_per_core_cliff] =
    ttnn::split_blocks_for_tilize(available_grid, nblocks);

// Algorithm from work_split_tilize.hpp:
// 1. nblocks_per_core = ceil(nblocks / grid_area)
// 2. ncores = ceil(nblocks / nblocks_per_core)
// 3. nblocks_per_core_cliff = nblocks % nblocks_per_core (if non-zero)
```

### Per-Core Work Assignment

```cpp
// Full cores:
row_start_id = i * nblocks_per_core * TILE_HEIGHT;
tile_start_id = i * nblocks_per_core * ntiles_per_block;

// Cliff core (if exists):
// Same formula but with nblocks_per_core_cliff
```

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one row-major stick in bytes |
| 1+ | TensorAccessorArgs | multiple | Interleaved buffer configuration (bank base addresses, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | Output circular buffer ID (c_16) |
| 1+ | TensorAccessorArgs | multiple | Interleaved buffer configuration |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block (ntiles_per_block) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_sticks | uint32_t | Total sticks to read (nblocks * TILE_HEIGHT) |
| 2 | block_size_nbytes | uint32_t | Bytes per stick |
| 3 | num_tiles_per_block | uint32_t | Tiles per output row |
| 4 | block_width_size | uint32_t | Same as block_size_nbytes |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 for this factory |
| 6 | num_leftover_tiles | uint32_t | 0 (not used) |
| 7 | leftover_width | uint32_t | 0 (not used) |
| 8 | start_stick_id | uint32_t | Starting stick index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_tiles | uint32_t | Total tiles to write (ntiles_per_block * nblocks_per_core) |
| 2 | start_id | uint32_t | Starting tile index for this core |

---

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | Interleaved DRAM (RM) | CB c_0 | Read sticks, stage for tilize |

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

**Key Logic**:
1. Pre-computes NoC addresses for 32 consecutive sticks
2. Reads all 32 sticks in sequence, filling CB row-by-row
3. Uses `TensorAccessor` for interleaved address generation
4. Barrier after each block to ensure data availability

```cpp
// Address pre-computation for one block:
for (uint32_t j = 0; j < tile_height; j++) {
    base_src_noc_addr[j] = get_noc_addr(stick_id, s);
    stick_id++;
}

// Sequential read of 32 sticks:
for (uint32_t k = 0; k < tile_height; k++) {
    noc_async_read(base_src_noc_addr[k], l1_write_addr, width_size);
    l1_write_addr += width_size;
}
```

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize | RISCV_UNPACK/MATH/PACK | N/A | CB c_0 | CB c_16 | tilize_block |

**File**: `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp`

**Key Logic**:
- Uses `compute_kernel_lib::tilize()` helper
- Processes one block at a time: wait for input, reserve output, tilize, push/pop

```cpp
compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
compute_kernel_lib::tilize(tt::CBIndex::c_0, per_core_block_tile_cnt,
                           tt::CBIndex::c_16, per_core_block_cnt);
```

The `tilize_block` operation:
1. **Unpack**: `llk_unpack_tilize_block` reads 32 RM rows, reorders to tile format
2. **Math**: `llk_math_eltwise_unary_datacopy` passes data through
3. **Pack**: `llk_pack` writes tiled data to output CB

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_16 | Interleaved DRAM (tiled) | Write tiles |

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

**Key Logic**:
- Single-page write pattern for simplicity
- Uses `TensorAccessor` for interleaved address generation
- Barrier after each write (could be optimized)

```cpp
for (uint32_t i = start_id; i < end_id; ++i) {
    cb_wait_front(cb_id_out, onepage);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);
    noc_async_write_page(i, s, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out, onepage);
}
```

---

## Implementation Notes

### Design Decisions

1. **Single-Buffered CBs**: Simple implementation prioritizes correctness over throughput. For high-performance scenarios, double-buffering would allow reader/compute/writer overlap.

2. **Block Granularity**: Processing exactly TILE_HEIGHT (32) sticks per block aligns with tile dimensions, ensuring clean tilization without partial tiles.

3. **Row-Major Stick Reading**: The reader pre-computes all 32 NoC addresses before issuing reads. This enables potential read coalescing by the NoC hardware.

4. **Generic Writer**: Uses `writer_unary_interleaved_start_id` which is shared across many ops. Single-tile granularity simplifies implementation but may not be optimal for bandwidth.

### Interleaved Memory Considerations

- Pages are distributed round-robin across DRAM banks
- Each stick/tile may reside in a different bank
- `TensorAccessor` abstracts bank selection: `bank_id = page_id % num_banks`
- NoC reads/writes to different banks can proceed in parallel

### Potential Optimizations

1. **Double-Buffering**: Use 2x capacity for CBs to overlap stages
2. **Batch Writes**: Writer could accumulate multiple tiles before barrier
3. **Fast Tilize**: `fast_tilize_block` variant available for non-Blackhole architectures

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the tilize operation and how does it transform row-major data to tiled format? What does tilize_block do in the compute kernel?"
   **Reason**: Needed to understand the fundamental transformation performed by tilize
   **Key Findings**: Tilize groups row-major elements into 32x32 tiles with 16x16 faces. `tilize_block` uses `llk_unpack_tilize_block` for unpacking/reordering, `llk_math_eltwise_unary_datacopy` for data movement, and `llk_pack` for output.

2. **Query**: "What is TensorAccessor and TensorAccessorArgs in tt-metal? How do they work with interleaved memory layouts?"
   **Reason**: Needed to understand how kernels access interleaved DRAM
   **Key Findings**: `TensorAccessor` maps logical page IDs to physical NoC addresses. For interleaved buffers, it computes `bank_id = page_id % num_banks` and generates appropriate NoC coordinates.

3. **Query**: "How does the circular buffer work for tilize operations? How many sticks are read before tilization can happen?"
   **Reason**: Needed to understand CB sizing and data requirements
   **Key Findings**: 32 sticks (one TILE_HEIGHT worth of rows) must be accumulated before tilization can produce a complete tile. The CB is sized to hold `ntiles_per_block` tiles worth of row-major data.

4. **Query**: "What is the noc_async_read function and how does it work with get_noc_addr to read from interleaved DRAM banks?"
   **Reason**: Needed to understand low-level NoC data movement
   **Key Findings**: `noc_async_read` initiates DMA from NoC address to L1. `get_noc_addr` with `TensorAccessor` computes the 64-bit address encoding bank coordinates and local offset.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding row-major vs tiled layout semantics
   **Key Information**: Row-major: each row is one page. Tiled: 32x32 blocks with 16x16 faces. Interleaved distributes pages round-robin across banks.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor API and configuration
   **Key Information**: `TensorAccessorArgs` configures compile-time vs runtime parameters. Device-side `TensorAccessor` provides `get_noc_addr()` and data transfer helpers.

3. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding core distribution algorithm
   **Key Information**: `split_blocks_for_tilize` divides blocks evenly with optional cliff core for remainder. Returns `BlockSplit` with core ranges and work counts.

4. **Source**: `tt_metal/include/compute_kernel_api/tilize.h`
   **Reason**: Understanding tilize compute API
   **Key Information**: `tilize_init` configures unpacker/math/packer. `tilize_block` processes one block through UNPACK->MATH->PACK pipeline.

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
   **Reason**: Understanding the unified tilize helper API
   **Key Information**: `compute_kernel_lib::tilize()` is a template-based helper that handles various tilize patterns (simple, activation, fast, DT-aware) through compile-time dispatch.
