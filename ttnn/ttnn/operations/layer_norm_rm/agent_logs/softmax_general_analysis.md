# Softmax General (W-Dimension) Implementation Analysis

## Overview

The softmax general operation computes softmax along an arbitrary dimension of a tensor. This analysis focuses on the **W-dimension variants** (`SoftmaxProgramFactoryGeneralWSmall` and `SoftmaxProgramFactoryGeneralWLarge`), which perform row-wise softmax reduction -- the pattern most relevant to `layer_norm_rm`.

**Program factory paths**:
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp`
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_large.cpp`

**Kernel paths** (shared with moreh_softmax):
- Reader: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w.cpp` (small) / `reader_moreh_softmax_w_large.cpp` (large)
- Compute: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp` (small) / `moreh_softmax_w_large.cpp` (large)
- Writer: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w.cpp` (small) / `writer_moreh_softmax_w_large.cpp` (large)

**WSmall vs WLarge selection criterion**: WSmall is selected when all circular buffers (with `Wt`-sized CBs for input, output, exp, x-max) fit within 512KB of L1 memory. Otherwise WLarge is used, which streams tiles one-at-a-time and re-reads input data multiple times.

**Mathematical formula** (SOFTMAX variant, `#define SOFTMAX`):
```
softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
```

This requires: (1) row-wise max, (2) subtract max, (3) exp, (4) row-wise sum, (5) reciprocal, (6) multiply.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Row of tiles (one tile-row = Wt tiles) |
| **Unit size** | Wt tiles (all tile columns spanning one tile-row) |
| **Total units** | `num_kernel_rows = (physical_volume / H / W) * Ht` |
| **Loop structure** | Outer loop over N tile-rows, inner loop over Wt tile-columns |

A "work unit" or "kernel row" is one tile-row: all Wt tiles sharing the same batch+height position. Each core processes `num_tiles_per_core` such kernel rows. The softmax reduction is performed independently per kernel row.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [..., H, W] (arbitrary rank) | Same as input |
| **Dimension convention** | Last dim = W (reduction dim) | Same as input |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (typical) | DRAM (typical) |
| **Data type** | BFLOAT16 or FLOAT32 | Same as input |

### Layout Transformations
- No explicit tilize/untilize within the operation
- When logical W is not a multiple of 32, a **mask tile** is generated to zero out padding columns in the last tile of each row. This is critical for correct max/sum computation

## Data Flow Pattern

### WSmall Variant (all Wt tiles fit in L1)

The WSmall variant loads an entire tile-row into L1 at once, enabling the compute kernel to access any tile by index without re-reading from DRAM.

**Stage 1 -- Reader**: For each kernel row, reads all Wt tiles from DRAM into `cb_in0` in one batch (`cb_reserve_back(cb_in, Wt)` then `cb_push_back(cb_in, Wt)`).

**Stage 2 -- Compute**: Processes the entire tile-row with all tiles resident in L1:
1. **Find max**: Reduce all Wt tiles with `REDUCE_ROW` + `MAX` to get a single tile containing row-wise maximums
2. **Subtract max**: Broadcast-subtract the max tile from each of the Wt input tiles using `sub_tiles_bcast<COL>`
3. **Exp**: Compute `exp(x - max)` for all Wt tiles (mask applied to last tile)
4. **Sum**: Reduce the Wt exp tiles with `REDUCE_ROW` + `SUM`, then apply `recip_tile` to get `1/sum`
5. **Multiply**: Broadcast-multiply each exp tile by the `1/sum` scalar tile using `mul_tiles_bcast_cols`

**Stage 3 -- Writer**: For each kernel row, waits for all Wt output tiles then writes them to DRAM in one batch.

### WLarge Variant (tiles streamed one-at-a-time)

When Wt is too large for all tiles to fit in L1 simultaneously, the WLarge variant uses CBs of size 2 (double-buffered) and **reads the input row 3 times** from DRAM:

**Pass 1 (max reduction)**: Reader streams Wt tiles one-by-one. Compute reduces them to find the row-wise max.

**Pass 2 (exp + accumulate sum)**: Reader re-reads all Wt tiles. Compute subtracts max, computes exp, applies mask to last tile, and accumulates a running sum into `cb_add` (tile-by-tile addition).

**Pass 3 (normalize)**: Reader re-reads all Wt tiles. Compute subtracts max, computes exp again, and multiplies by `1/sum` to produce the final output.

The writer streams output tiles one-at-a-time (`cb_wait_front(cb_out, 1)` per tile).

## Circular Buffer Configuration

### WSmall Variant

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles (entire row) | Wt tiles | Wt tiles | Single | Reader | Compute | Row |
| c_1 | cb_mask | Width mask tile (padding zeroing) | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_2 | cb_bcast_scaler | Reduce scaler (all 1.0) | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_16 | cb_out0 | Output tiles | Wt tiles | Wt tiles | Single | Compute | Writer | Row |
| c_24 | cb_exps | exp(x - max) intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Row |
| c_25 | cb_recipsumexps | 1/sum(exp) result | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_26 | cb_max | Row-wise max result | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_27 | cb_x_m_max | x - max intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Row |
| c_28 | cb_tmp | Temporary (masked tile) | 1 tile | 1 tile | Single | Compute | Compute | Block |

**Data format**: CBs c_0, c_1, c_2, c_16 use the input data format (BFLOAT16 or FLOAT32). CBs c_24 through c_28 use `intermed_data_format` which is FLOAT32 if `fp32_dest_acc_en` is true, otherwise matches the input format.

### WLarge Variant

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles (streamed) | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_1 | cb_mask | Width mask tile | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_2 | cb_bcast_scaler | Reduce scaler | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_16 | cb_out0 | Output tiles (streamed) | 2 tiles | 1 tile | Double | Compute | Writer | Block |
| c_24 | cb_exps | exp(x - max) intermediate | 2 tiles | 1 tile | Double | Compute | Compute | Block |
| c_25 | cb_recipsumexps | 1/sum(exp) result | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_26 | cb_add | Running sum accumulator | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_27 | cb_max | Row-wise max result | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_28 | cb_tmp | Temporary | 1 tile | 1 tile | Single | Compute | Compute | Block |

**Key difference**: WLarge uses capacity=2 for input, output, and exps CBs (double-buffering for streaming), while WSmall uses capacity=Wt (all tiles resident).

## Pipeline Pattern Summary

### WSmall
- **Input (c_0)**: Single-buffered at Wt tiles -- Reader fills entire row, compute processes entire row
- **Output (c_16)**: Single-buffered at Wt tiles -- Compute fills entire row, writer drains entire row
- No overlap between reader and compute within a single row (compute needs all tiles for max reduction)

### WLarge
- **Input (c_0)**: Double-buffered at 1 tile -- enables reader/compute overlap during streaming passes
- **Output (c_16)**: Double-buffered at 1 tile -- enables compute/writer overlap during output
- Three-pass architecture means input is read from DRAM 3 times per kernel row

## Index Calculations

### Tile Addressing
Tiles are addressed using a linear tile index through `TensorAccessor`. The reader computes:
```
curr_tile = tile_offset + (row_index * Wt) + w
```
where `tile_offset` is the starting tile for this core (from the work split), `row_index` iterates over the N kernel rows assigned to this core, and `w` iterates from 0 to Wt-1.

### TensorAccessor Usage
Both reader and writer use `TensorAccessor` initialized from compile-time `TensorAccessorArgs`. The reader creates it as:
```cpp
constexpr auto in_args = TensorAccessorArgs<1>();  // CT arg index 1 onwards
const auto src_in = TensorAccessor(in_args, src_addr, src_in_tile_bytes);
```
The `noc_async_read_tile(curr_tile, src_in, l1_write_addr)` call uses the accessor to translate the linear tile index to a physical NoC address, handling interleaved bank mapping internally.

### Mask Index Calculation
```cpp
uint32_t mask_w = input.logical_shape()[-1] % TILE_WIDTH;
if (mask_w == 0) mask_w = TILE_WIDTH;
```
This computes how many valid columns exist in the last tile of each row. The mask tile is generated once in the reader and persists for the entire program.

## Memory Access Patterns

### Read Pattern
- **WSmall**: Sequential read of Wt contiguous tiles per row, one complete row at a time. Single read pass per row.
- **WLarge**: Sequential read of Wt tiles per row, one tile at a time. **Three read passes per row** (max pass, exp+sum pass, normalize pass). Each pass re-reads the same tiles from DRAM, with the read offset reset to `curr_offset_i` at the beginning of each pass.

### Write Pattern
- **WSmall**: Sequential write of Wt contiguous tiles per row, one complete row at a time.
- **WLarge**: Sequential write of 1 tile at a time, Wt tiles per row.

### DRAM vs L1 Access
- Input/output data resides in DRAM (interleaved)
- All intermediate results (max, exp, sum, x-max) reside in L1 circular buffers
- Mask and scaler tiles are generated in L1 by the reader and never touch DRAM

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major traversal) |
| **Grid dimensions** | `grid_coord.x` x `grid_coord.y` (full compute grid) |
| **Total cores** | min(num_kernel_rows, grid_coord.x * grid_coord.y) |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` kernel rows |
| **Load balancing** | Near-equal: group1 gets ceil(N/cores) rows, group2 gets floor(N/cores) |

Core indexing uses column-major order:
```cpp
CoreCoord core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset};
```
where `core_h` is the number of rows in the grid. This means cores fill column-by-column (y varies fastest).

The `tile_offset` accumulates across cores:
```cpp
tile_offset += num_tiles_per_core * Wt;
```
Each core's tile_offset tells it where its work begins in the linear tile space.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | is_fp32 | uint32_t (bool) | 1 if input dtype is FLOAT32, 0 for BFLOAT16 |
| 1+ | TensorAccessorArgs | uint32_t[] | Buffer descriptor for NoC addressing (auto-appended) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of kernel rows assigned to this core |
| 1 | Wt | uint32_t | Number of tiles along the W dimension |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Buffer descriptor for NoC addressing |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | N (num_tiles_per_core) | uint32_t | Number of kernel rows for this core |
| 2 | tile_offset | uint32_t | Starting tile index for this core |
| 3 | Wt | uint32_t | Tiles along W dimension |
| 4 | scaler | uint32_t | Bit-cast float scaler value (1.0f for softmax) |
| 5 | mask_w | uint32_t | Number of valid columns in last tile (1-32) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | N (num_tiles_per_core) | uint32_t | Number of kernel rows for this core |
| 2 | tile_offset | uint32_t | Starting tile index for this core |
| 3 | Wt | uint32_t | Tiles along W dimension |

### Compile Defines

| Define | Value | Description |
|--------|-------|-------------|
| SOFTMAX | 1 | Enables softmax path (vs softmin which negates before exp) |
| FP32_DEST_ACC_EN | 1 (conditional) | Enables FP32 accumulation in dest registers |

## Kernel Implementations

### Reader Kernel (WSmall): `reader_moreh_softmax_w.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM | cb_in0, cb_mask, cb_scaler | Read input tiles, generate mask and scaler |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w.cpp`
- **Key Logic**:
  1. Generates the **broadcast scaler tile** (all elements = 1.0) into `cb_scaler` using `generate_bcast_scaler<T>(cb_scaler, scaler)`. This fills the first row of each face (indices k*256+j for k=0..3, j=0..15) with the scaler value, rest zeros. For bfloat16 the upper 16 bits of the float are used.
  2. Generates the **width mask tile** into `cb_mask` using `generate_mask_w<T>(cb_mask, mask_w)`. This creates a tile where columns 0..mask_w-1 are 1.0 and columns mask_w..31 are 0.0, applied per-subtile.
  3. Main loop: For each of N kernel rows, reserves Wt slots in cb_in0, reads Wt tiles sequentially from DRAM using `noc_async_read_tile`, issues a barrier, then pushes all Wt tiles.

### Reader Kernel (WLarge): `reader_moreh_softmax_w_large.cpp`

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w_large.cpp`
- **Key Logic**: Same scaler/mask generation as WSmall. But the main loop reads the same row **three times**:
  1. First pass: Reads Wt tiles one-at-a-time (for max reduction)
  2. Second pass: Re-reads Wt tiles one-at-a-time (for exp+sum)
  3. Third pass: Re-reads Wt tiles one-at-a-time (for normalize+output)

  Each pass resets `curr_tile = curr_offset_i` before starting. Tiles are read/pushed individually (`cb_reserve_back(cb_in, 1)` then `cb_push_back(cb_in, 1)`).

### Compute Kernel (WSmall): `moreh_softmax_w.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | cb_in0, cb_mask, cb_scaler | cb_out0 | MAX reduce, subtract, exp, SUM reduce, recip, multiply |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`
- **Key Logic** (SOFTMAX path, per kernel row):

**Step 1: Find row-wise max**
```
if Wt == 1:
    mask the single input tile -> cb_tmp
    reduce MAX REDUCE_ROW: cb_tmp -> cb_max
else:
    reduce MAX REDUCE_ROW (WaitUpfrontNoPop): first Wt-1 tiles of cb_in0 -> cb_max
    mask last tile (tile index Wt-1) -> cb_tmp
    reduce MAX REDUCE_ROW with Accumulate::at(cb_max, 1): cb_tmp -> cb_max
```
The two-phase max (Wt-1 tiles then 1 masked tile with accumulation) ensures padding columns do not contribute to the max. The `WaitUpfrontNoPop` policy means the reduce waits for all Wt tiles upfront but does NOT pop them -- they remain available for the subtraction step.

**Step 2: Compute x - max(x)**
```
for w in 0..Wt:
    sub_tiles_bcast<COL>(cb_in0[w], cb_max[0]) -> cb_x_m_max[w]
pop cb_max (1 tile), pop cb_in0 (Wt tiles)
push cb_x_m_max (Wt tiles)
```
`BroadcastType::COL` means: the max tile (which has one reduced value per row of 32) is broadcast across all 32 columns. Each row in the 32x32 tile sees its own max value subtracted from all its columns. This is exactly what `sub_bcast_cols_init_short_with_dt` + `sub_tiles_bcast<BroadcastType::COL>` accomplish.

**Step 3: Compute exp(x - max(x))**
```
for w in 0..Wt:
    copy tile from cb_x_m_max[w] to DST
    exp_tile(DST)
    if w == Wt-1: apply mask_tile to zero out padding
    pack DST -> cb_exps[w]
push cb_exps (Wt tiles)
```

**Step 4: Reduce sum + reciprocal**
```
reduce SUM REDUCE_ROW (WaitUpfrontNoPop): cb_exps (Wt tiles) -> cb_recipsumexps
    post_reduce_op: recip_tile(dst)  // 1/sum
```
The `WaitUpfrontNoPop` policy keeps exp tiles in cb_exps for the final multiplication step. The `recip_tile` post-reduce operation is applied inline after the reduction completes, giving `1/sum(exp(x-max))`.

**Step 5: Final result -- exp(x-max) * (1/sum)**
```
for w in 0..Wt:
    mul_tiles_bcast_cols(cb_exps[w], cb_recipsumexps[0]) -> cb_out0[w]
pop cb_recipsumexps (1), pop cb_x_m_max (Wt), pop cb_exps (Wt)
push cb_out0 (Wt tiles)
```
`mul_tiles_bcast_cols` broadcasts the single 1/sum tile (which has one value per row) across all columns of each exp tile.

### Compute Kernel (WLarge): `moreh_softmax_w_large.cpp`

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp`
- **Key Logic**: Same algorithmic steps but tiles arrive one-at-a-time. Key differences:
  - **Max phase**: Uses default `WaitAndPopPerTile` policy for first Wt-1 tiles (pops each after reducing). Then masks and reduces the last tile with accumulation.
  - **Exp+Sum phase**: For each tile: sub_max -> exp -> (mask last tile) -> accumulate into `cb_add` (tile-by-tile `add_tiles_to_cb`). Then reduces `cb_add` with SUM+recip.
  - **Output phase**: For each tile: sub_max -> exp -> multiply by 1/sum -> write to cb_out0. Each output tile is produced and popped individually.
  - `cb_max` and `cb_recipsumexps` are popped only at the end of all three phases.

### Writer Kernel (WSmall): `writer_moreh_softmax_w.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | cb_out0 | DRAM | Write output tiles |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w.cpp`
- **Key Logic**: For each of N rows, waits for Wt tiles in cb_out0, writes them sequentially to DRAM using `noc_async_write_tile`, barriers, then pops all Wt.

### Writer Kernel (WLarge): `writer_moreh_softmax_w_large.cpp`

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w_large.cpp`
- **Key Logic**: For each row, writes tiles one-at-a-time: waits for 1 tile, writes, barriers, pops 1.

## Implementation Notes

### Scaler Tile Construction
The scaler tile is generated by the reader on each core with `generate_bcast_scaler<T>(cb_scaler, scaler)`. For softmax, the scaler is 1.0 (the reduce hardware multiplies by this scaler after accumulation). The function writes the scaler value only to the first row of each 16x16 face (indices 0..15 within each 256-element face block), with the remaining 15 rows set to zero. For bfloat16, the value is stored as the upper 16 bits of the float32 representation.

**Important for layer_norm_rm**: The scaler can be set to `1/W` for mean computation, avoiding a separate divide step. The runtime arg `scaler` is passed as a bit-cast `uint32_t` from a `float`:
```cpp
float scaler = 1.0f;
*reinterpret_cast<uint32_t*>(&scaler)  // bit-pattern as uint32
```

### Mask Tile Construction
`generate_mask_w<T>(cb_mask, mask_w)` creates a 32x32 tile where:
- Columns 0 through mask_w-1: filled with 1.0
- Columns mask_w through 31: filled with 0.0
This is used with `mask_tile()` which zeroes out elements where the mask is 0.0, preventing padding columns from contributing to reductions.

### BroadcastType::COL Semantics
When `sub_tiles_bcast<BroadcastType::COL>(cb_A, cb_B, tile_A, tile_B, dst)` is called:
- `cb_B` is expected to contain a tile with meaningful values only in column 0 (the reduce output has values in column 0 of each face)
- The operation computes `dst[h][w] = A[h][w] - B[h][0]` for all h, w
- This effectively broadcasts each row's scalar across all columns

This is the mechanism by which a 1-tile reduced result (max or 1/sum) is applied back to all Wt tiles of a row.

### Numerical Stability
The two-phase max-then-exp approach (`exp(x - max(x))`) ensures numerical stability by preventing overflow in the exp operation. This is the standard numerically stable softmax algorithm.

### FP32 Destination Accumulation
When `fp32_dest_acc_en` is true:
- Intermediate CBs use `Float32` data format (4 bytes per element vs 2 for bfloat16)
- `pack_tile_with_dt` calls `pack_reconfig_data_format` before packing
- `copy_tile_init_with_dt` calls `reconfig_data_format_srca` before copy init
- All intermediate computations benefit from higher precision

### WLarge Three-Pass Trade-off
The WLarge variant reads from DRAM 3x per row, which is a significant bandwidth cost. However, it avoids the L1 memory pressure that would come from storing all Wt tiles simultaneously. This is a classic memory-bandwidth trade-off. For `layer_norm_rm`, if the W dimension is small enough, the "small" pattern (all tiles in L1) is strongly preferred.

### Relevance to layer_norm_rm
The softmax W-dimension pattern provides a direct template for layer_norm_rm:
1. **Row-wise mean**: Replace MAX reduction with SUM, use scaler=1/W (or reduce with scaler=1.0 then multiply by 1/W)
2. **Subtract mean**: Same `sub_tiles_bcast<COL>` pattern as softmax's `x - max(x)`
3. **Variance**: Square the (x - mean) result, then SUM reduce again
4. **Normalize**: Multiply by `rsqrt(variance + epsilon)` using `mul_tiles_bcast_cols`
5. **Scale/Bias**: Additional `mul_tiles_bcast_cols` and `add_tiles_bcast_cols` for gamma/beta

The CB layout, work splitting, and reader/writer patterns can be largely reused.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the REDUCE_ROW operation work in the compute kernel? Specifically, what does the reduce hardware do when reducing a row of tiles (W dimension) - does it sum/max all 32 columns within each tile and then combine across tiles? What role does the scaler tile play in reduce operations?"
   **Reason**: Needed to understand the hardware-level semantics of REDUCE_ROW and the scaler tile
   **Key Findings**: REDUCE_ROW reduces the W dimension by combining values across all 32 columns within each tile, then combines across tiles in sequence. The scaler tile is multiplied with the result after reduction -- for SUM/MAX it's typically 1.0, for AVG it should be 1/N. The `generate_reduce_scaler` fills the first row of each face with the scaler value. The reduce operation is configured via `reduce_init` which sets up unpacker, math, and packer hardware.

2. **Query**: "What does BroadcastType::COL mean in sub_tiles_bcast operations? When we do sub_tiles_bcast<BroadcastType::COL>(cb_in, cb_scalar, ...), does it broadcast the column-reduced scalar across all columns of the input tile?"
   **Reason**: Needed to understand the broadcast semantics for subtract-max and multiply-recipsum operations
   **Key Findings**: `BroadcastType::COL` broadcasts column 0 of the second operand across all columns. For `sub_tiles_bcast<COL>`, this computes `C[h,w] = A[h,w] - B[h,0]`. The 1-tile reduced result (which has per-row values in column 0) is broadcast to all Wt tiles by calling the operation in a loop for each tile column.

3. **Query**: "How does split_work_to_cores work in moreh operations? What do core_group_1 and core_group_2 represent?"
   **Reason**: Needed to understand the work distribution strategy across cores
   **Key Findings**: `split_work_to_cores` divides N work units across available cores. `core_group_1` gets `ceil(N/cores)` units and `core_group_2` gets `floor(N/cores)` units. The number of cores in group_1 equals `N % total_cores` (the remainder). If N divides evenly, group_2 is empty. This ensures near-equal load balancing with at most 1 unit difference between groups.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
   **Reason**: Understanding the reduce library API, input policies, and accumulation patterns
   **Key Information**: The reduce function supports multiple policies: `WaitAndPopPerTile` (streaming), `BulkWaitBulkPop` (batch), `WaitUpfrontNoPop` (persistent for reuse), `NoWaitNoPop` (caller-managed). Supports post-reduce operations as lambdas (e.g., `recip_tile` for softmax). Accumulation via `Accumulate::at(cb, iteration)` for multi-phase reductions.

2. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding helper functions like `mask_tile_to_cb`, `sub_tiles_bcast_cols_to_cb`, `exp_tile_to_cb`, etc.
   **Key Information**: These are convenience wrappers that handle the full DST register dance (acquire/commit/wait/release), CB synchronization (wait/reserve/pop/push), and FP32 data format reconfiguration. Each function processes one tile at a time. The `_to_cb` suffix means the result is written to a specified output CB.

3. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: Understanding `generate_bcast_scaler` and `generate_mask_w` tile generation
   **Key Information**: `generate_bcast_scaler` fills the first row (16 elements) of each of the 4 faces (at offsets 0, 256, 512, 768) with the scaler value. For bfloat16, the upper 16 bits of the float32 are used. `generate_mask_w` creates a width mask with 1.0 for valid columns and 0.0 for padding, split across subtiles based on whether mask_w is greater or less than 16.

4. **Source**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_device_operation.cpp`
   **Reason**: Understanding WSmall vs WLarge selection criteria
   **Key Information**: WSmall is selected when the total CB memory (input + mask + scaler + output + 3 intermediate Wt-sized + 3 single-tile) fits within 512KB of L1 memory starting from the base allocator address. The check is performed by `is_softmax_general_w_small_available()`.
