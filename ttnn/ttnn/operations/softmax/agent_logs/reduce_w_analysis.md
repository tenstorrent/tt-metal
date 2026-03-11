# Reduce W (Width-Dimension Reduction) Implementation Analysis

## Overview

The reduce_w operation performs reduction along the W (width) dimension of a 4D tensor.
For each row of tiles (fixed N, C, H position), all Wt tiles along the width are reduced
to a single output tile. This means the output width is always 1 tile (32 columns), while
height, batch, and channel dimensions are preserved.

**Program factory path**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp`

**Supported reduce operations**: SUM, AVG, MAX (via `ReduceOpMath` enum, mapped to `PoolType`)

**Critical design choice**: For SUM and AVG reductions along W, a **matmul-based path**
(`REDUCE_ROW_SUM_VIA_MM`) is used instead of `reduce_tile`, reportedly for better precision.
For MAX reductions along W, the standard `reduce_tile` path via the helper library is used.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-row |
| **Unit size** | Wt tiles (one row of tiles along W, reduced to 1 output tile) |
| **Total units** | NC * Ht (one unit per row of tiles across all batches/channels/height positions) |
| **Loop structure** | Outer: rows assigned to this core; Inner: Wt tiles accumulated per row |

One "work unit" is one complete row of Wt tiles that are reduced into a single output tile.
The total number of work units is `num_rows = NC * Ht` where `NC = shape[0] * shape[1]` and
`Ht = shape[2] / TILE_HEIGHT`.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N*C, H, W] (internally flattened to NC, Ht, Wt) | [N*C, H, 1] (internally NC, Ht, 1 tile) |
| **Dimension convention** | NHWC flattened: dim0*dim1=NC, dim2=H, dim3=W | Same, with W reduced to 1 tile |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (interleaved) | DRAM (interleaved) |
| **Data type** | Configurable (src0_cb_data_format derived from input dtype) | Configurable (dst_cb_data_format derived from output dtype) |

### Layout Transformations

No explicit tilize/untilize or reshard operations. Input and output are both in TILE_LAYOUT.
The reduction collapses W from Wt tiles to 1 tile per row, so the output has `NC * Ht` tiles total
(compared to `NC * Ht * Wt` input tiles).

## Data Flow Pattern

### High-Level Flow

1. **Reader kernel** reads input tiles sequentially from DRAM into CB0 (one tile at a time),
   and prepares the scaler tile in CB2 at the start.
2. **Compute kernel** consumes Wt tiles from CB0 per row, accumulates them in DST register,
   and produces 1 output tile per row into CB3.
3. **Writer kernel** reads output tiles from CB3 and writes them sequentially to DRAM.

### Reader Kernel (Summary)

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp`

The reader performs two tasks:
1. **Scaler preparation** (once, at kernel start): Calls either
   `dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2>(scaler_f)` (for the reduce_tile path)
   or `generate_mm_scaler(cb_id_in2, packed_bf16)` (for the matmul path).
   The scaler is pushed into CB2 and remains there for the entire kernel lifetime.
2. **Input tile streaming**: Reads `num_tiles` tiles sequentially starting from `start_id`,
   one tile at a time via `noc.async_read` into CB0, using TensorAccessor for address resolution.

### Writer Kernel (Summary)

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

Simple sequential writer: waits for one tile in CB3 (the output CB), reads it from L1,
writes it to DRAM via `noc_async_write_page`, pops, and repeats for all output tiles.

## Compute Kernel Structure: Two Paths

The compute kernel has two distinct implementations selected at compile time via the
`REDUCE_ROW_SUM_VIA_MM` define:

### Path A: reduce_tile via Helper Library (MAX reduction, or when REDUCE_ROW_SUM_VIA_MM is not defined)

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`

When `REDUCE_ROW_SUM_VIA_MM` is **not** defined (i.e., MAX reduction), the kernel delegates
entirely to the `compute_kernel_lib::reduce` helper:

```cpp
compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

compute_kernel_lib::reduce<
    REDUCE_OP,                                           // e.g. PoolType::MAX
    REDUCE_DIM,                                          // ReduceDim::REDUCE_ROW
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
    tt::CBIndex::c_0,                                    // input_cb
    tt::CBIndex::c_2,                                    // scaler_cb
    tt::CBIndex::c_3,                                    // output_cb
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
```

**What the helper does internally** (from `reduce_helpers_compute.inl`, REDUCE_ROW branch):

For each batch (nc in 0..NC) and each row (ht in 0..Ht):
1. `tile_regs_acquire()` -- acquire DST registers
2. For each tile along width (wt in 0..Wt):
   - `cb_wait_front(input_cb, 1)` -- wait for reader to provide 1 tile in CB0
   - `reduce_tile<REDUCE_OP, REDUCE_ROW>(input_cb, scaler_cb, 0, 0, dst_idx)`
     This unpacks tile 0 from CB0 and scaler tile 0 from CB2, performs the
     row-wise reduction, and **accumulates** the result into DST[dst_idx].
     The key insight: reduce_tile with REDUCE_ROW collapses each 32-column
     tile into a single column, multiplied by the scaler. When called
     repeatedly on tiles in the same row, results accumulate in DST[0].
   - `cb_pop_front(input_cb, 1)` -- free the consumed tile from CB0
3. `cb_reserve_back(output_cb, 1)` -- reserve space in output CB3
4. `tile_regs_commit()` / `tile_regs_wait()` -- synchronize DST between math and pack
5. `pack_tile(dst_idx, output_cb)` -- pack accumulated result from DST[0] to CB3
6. `tile_regs_release()` -- release DST registers
7. `cb_push_back(output_cb, 1)` -- signal output tile is ready

After all processing: `reduce_uninit()` resets packer edge masks.

### Path B: matmul_tiles (SUM/AVG reduction, when REDUCE_ROW_SUM_VIA_MM is defined)

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`

When `REDUCE_ROW_SUM_VIA_MM` **is** defined (SUM or AVG), the kernel uses matrix multiplication:

```cpp
mm_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

cb2.wait_front(1);  // scaler tile from the reader -- stays in CB for entire kernel
for (uint32_t nc = 0; nc < NC; nc++) {
    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        acquire_dst();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb0.wait_front(onetile);
            matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, 0);
            cb0.pop_front(onetile);
        }
        cb3.reserve_back(onetile);
        pack_tile(reduce_dst_idx, tt::CBIndex::c_3);
        cb3.push_back(onetile);
        release_dst();
    }
}
```

**How matmul achieves row reduction**:

The scaler tile (CB2) is specially constructed by `generate_mm_scaler`:
- The tile is zeroed out, then only the **first column of each left face** is filled
  with the scaler value (1.0 for SUM, 1/Wt for AVG). Specifically, `ptr[i]` is set
  for `i = 0, 8, 16, ..., 120` (face 0) and `i = 256, 264, ..., 376` (face 2).
  These correspond to column 0 of the two left faces in bf16 format.
- When `matmul_tiles(A, scaler, 0, 0, dst_idx)` computes `DST += A * scaler`,
  only column 0 of the scaler is non-zero, so the result is effectively the
  dot product of each row of A with a column of 1s (or 1/N), producing the
  row sum (or row mean) in column 0 of the output.
- Since `matmul_tiles` accumulates (`DST += ...`), calling it for each of the
  Wt input tiles accumulates all partial row-sums into DST[0].

**Key difference from reduce_tile path**: The matmul path uses `acquire_dst()`/`release_dst()`
instead of `tile_regs_acquire()`/`tile_regs_commit()`/`tile_regs_wait()`/`tile_regs_release()`.
The `acquire_dst` / `release_dst` pair is the simplified full-DST-ownership model where one
kernel phase (math+pack) owns DST exclusively.

### Negate Variant (reduce_w_neg.cpp)

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w_neg.cpp`

This variant is selected when `operation_attributes.negate == true`. It does NOT use the
helper library and does NOT use the matmul path. Instead it performs a manual tile-by-tile
reduction with explicit negation:

For each row of Wt tiles:
1. **Negate each input tile**: `copy_tile(cb_input, 0, dst_idx)` then `negative_tile(dst_idx)`,
   pack result into cb_ineg (CB5)
2. **Accumulate using reduce_tile**: For wt > 0, reload the running accumulator from cb_acc (CB4)
   via `copy_tile(cb_acc, 0, dst_idx)`. Then call
   `reduce_tile<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, 0, 0, dst_idx)` to reduce the
   negated tile and accumulate. Pack result back to cb_acc (CB4).
3. **Final negation**: After all Wt tiles are accumulated, negate the result again:
   `copy_tile(cb_acc, 0, dst_idx)` then `negative_tile(dst_idx)`, pack into cb_output (CB3).

This implements `reduce(-x)` then negates the result, i.e., `-reduce(-x)`.
Uses two extra CBs: CB4 (accumulator, 1 tile) and CB5 (negated intermediate, 1 tile).

## Circular Buffer Configuration

### Standard Path (non-negate)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 (0) | cb_input | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_2 | cb_scaler | Scaler/constant tile | 2 tiles | 1 tile | Double | Reader | Compute | Program (entire kernel) |
| c_3 | cb_output | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Block (per row output) |

### Negate Path (additional CBs)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_4 | cb_acc | Running accumulator | 1 tile | 1 tile | Single | Compute | Compute | Row (across Wt iterations) |
| c_5 | cb_ineg | Negated input intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block (per tile within row) |

**Note on CB2 (scaler)**: Although allocated with capacity for 2 tiles, only 1 tile is ever
pushed. It stays in the CB for the entire program execution -- the compute kernel calls
`cb_wait_front(scaler_cb, 1)` once at the start and never pops it.

**Note on CB capacity**: CB0 and CB3 both have capacity=2, block_size=1, enabling double-buffering.
This allows the reader to push one tile while compute consumes the previous, and compute to push
output while writer drains the previous.

## Scaler Tile Construction

### reduce_tile Path (MAX): `prepare_reduce_scaler`

`dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2>(scaler_f)` (from `reduce_helpers_dataflow.inl`):
1. `cb_reserve_back(cb_id, 1)` -- reserve 1 tile slot in CB2
2. `zero_faces(write_addr)` -- zero out entire tile (all 4 faces)
3. `fill_row0<data_format, half_tile>(ptr, scaler)` -- fill row 0 of each face with the scaler value.
   For bf16: fills 8 uint32 values per face (= 16 bf16 values = one row of 16 columns) across all 4 faces.
   The scaler value is packed as `(bf16 << 16) | bf16` so each uint32 holds two copies.
4. `cb_push_back(cb_id, 1)` -- make the tile available to compute

The scaler tile has the scaler value in row 0 of every face. When `reduce_tile<..., REDUCE_ROW>`
processes an input tile, it performs a dot-product of each input column with the scaler row,
effectively multiplying the row sum by the scaler. For SUM, scaler=1.0; for MAX, scaler=1.0;
for AVG, scaler=1/N.

### matmul Path (SUM/AVG): `generate_mm_scaler`

`generate_mm_scaler(cb_id_in2, packed_bf16)` (from `generate_mm_scaler.hpp`):
1. Zeroes entire 2048-byte tile via NoC reads from MEM_ZEROS
2. Sets `ptr[i]` for `i in {0, 8, 16, ..., 120}` (face 0, column 0, 16 rows)
   and `i in {256, 264, ..., 376}` (face 2, column 0, 16 rows) to `single_packed_scalar`
   (only lower 16 bits, i.e., one bf16 value per uint32 position).
3. This creates a tile that is essentially a column vector of scaler values in the left column,
   zeros everywhere else. When used as the B operand in `matmul_tiles(A, B, ...)`, the result
   `A * B` produces the dot product of each row of A with this column vector, yielding row sums.

## Pipeline Pattern Summary

- **CB0 (input)**: Double-buffered (capacity=2, block=1). Reader and compute can overlap.
- **CB2 (scaler)**: Effectively persistent single-buffered. Written once, read throughout.
- **CB3 (output)**: Double-buffered (capacity=2, block=1). Compute and writer can overlap.
- **CB4 (acc, negate path only)**: Single-buffered. Self-producing, self-consuming by compute.
- **CB5 (ineg, negate path only)**: Single-buffered. Self-producing, self-consuming by compute.

## Index Calculations

Tile indexing is linear. The program factory computes:
- `num_rows = NC * Ht` total rows across all batches
- Each core is assigned `num_rows_per_core` consecutive rows
- `num_tensor_tiles_per_core = num_rows_per_core * Wt` (total input tiles for this core)
- Reader: reads tiles from `start_id` to `start_id + num_tensor_tiles_per_core - 1`
- Writer: writes tiles from `start_id / Wt` to `start_id / Wt + num_rows_per_core - 1`
- Tiles arrive at compute in natural order: row 0 tile 0, row 0 tile 1, ..., row 0 tile Wt-1, row 1 tile 0, ...

The reader uses TensorAccessor for physical address resolution (maps linear tile ID to bank/offset).

## Memory Access Patterns

### Read Pattern
Sequential tile reads from DRAM. Each tile is read individually (one `noc.async_read` per tile)
with a barrier before pushing. Tiles are read in linear order starting from `start_id`.

### Write Pattern
Sequential tile writes to DRAM. One output tile per row. Writer processes tiles one at a time
with `noc_async_write_page` and a `noc_async_writes_flushed()` between pages.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (cores enumerated linearly) |
| **Grid dimensions** | Up to compute_with_storage_grid_size.x * y |
| **Total cores** | min(num_rows, max_cores) |
| **Work per core** | num_rows_per_core complete rows (each row = Wt input tiles, 1 output tile) |
| **Load balancing** | Two-group split: core_group_1 gets ceil(num_rows/num_cores) rows, core_group_2 gets floor |

Work splitting uses `tt::tt_metal::split_work_to_cores`:
- `core_group_1` cores each process `num_rows_per_core_group_1` rows
- `core_group_2` cores each process `num_rows_per_core_group_2` rows (one fewer)
- If `num_rows` divides evenly, core_group_2 is empty

Optional `sub_core_grids` parameter allows restricting the core set.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | scaler_bits | uint32_t | Bit-cast float scaler value (1.0 for SUM/MAX, 1/N for AVG) |
| 1+ | tensor_accessor_args | uint32_t[] | TensorAccessorArgs for input buffer (bank mapping, page size, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (always c_3 = 3) |
| 1+ | tensor_accessor_args | uint32_t[] | TensorAccessorArgs for output buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile-rows per core (num_rows_per_core_group_{1,2}) |
| 1 | Wt | uint32_t | Number of tiles along width dimension |
| 2 | NC | uint32_t | Always 1 (batch dimension folded into Ht via core distribution) |

**Note on NC=1**: The program factory always passes NC=1 to the compute kernel because
the NC*Ht rows are distributed across cores, so each core sees its share as pure row count.
The triple-loop (NC, Ht, Wt) in the compute kernel is effectively a double-loop (Ht rows, Wt tiles per row).

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_tensor_tiles_per_core | uint32_t | Total input tiles this core reads (rows * Wt) |
| 2 | start_id | uint32_t | Global tile index where this core starts reading |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_output_tiles | uint32_t | Number of output tiles (= num_rows_per_core) |
| 2 | output_start_id | uint32_t | Global output tile index where this core starts writing |

### Preprocessor Defines (passed to all kernels)

| Define | Value (for W reduction) | Description |
|--------|------------------------|-------------|
| REDUCE_OP | PoolType::SUM / PoolType::AVG / PoolType::MAX | Reduction operation type |
| REDUCE_DIM | ReduceDim::REDUCE_ROW | Reduction dimension (W = row reduction) |
| REDUCE_ROW_SUM_VIA_MM | 1 (only for SUM/AVG) | Selects matmul path over reduce_tile path |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_reduce_universal_start_id | RISCV_0 | NOC0 | DRAM (input tensor) | CB0 (input tiles), CB2 (scaler tile) | Read tiles sequentially, prepare scaler |

- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp`
- **Key Logic**: Prepares scaler first (once), then streams input tiles one at a time.

### Compute Kernel (Standard Path)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reduce_w | RISCV_2 (unpack+math+pack) | N/A | CB0 (input), CB2 (scaler) | CB3 (output) | reduce_tile or matmul_tiles accumulation |

- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`
- **Key Logic**:
  - **matmul path (SUM/AVG)**: `mm_init` -> for each row: `acquire_dst` -> for each Wt tile: `cb0.wait_front(1)`, `matmul_tiles(c_0, c_2, 0, 0, 0)`, `cb0.pop_front(1)` -> `pack_tile(0, c_3)` -> `release_dst`
  - **reduce_tile path (MAX)**: Delegates to `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, WaitAndPopPerTile, NONE>(c_0, c_2, c_3, ReduceInputBlockShape::of(Ht, Wt, NC))`

### Compute Kernel (Negate Variant)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reduce_w_neg | RISCV_2 | N/A | CB0 (input), CB2 (scaler) | CB3 (output) | copy_tile, negative_tile, reduce_tile with intermediate CBs |

- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w_neg.cpp`
- **Key Logic**: Per-tile negate -> reduce_tile accumulate into CB4 -> final negate -> CB3

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB3 (output tiles) | DRAM (output tensor) | Write tiles sequentially |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential tile write with TensorAccessor.

## How Width-Dimension Reduction Works Tile-by-Tile

This section details the accumulation mechanism, which is the core pattern needed for
implementing width reduction in a new operation like softmax.

### Conceptual Model

Given a row of Wt tiles, each tile is 32x32 elements. We want to produce a single 32x32
output tile where column `c` contains the reduction of all values at column `c` across all
Wt input tiles (for SUM), or the maximum value at column `c` across all tiles (for MAX).

### matmul_tiles Accumulation (SUM/AVG Path)

The matmul approach treats each input tile as a 32x32 matrix and multiplies it by a
specially constructed scaler tile that acts as a "sum-reduction matrix":

```
scaler tile B = [ s 0 0 ... 0 ]
                [ s 0 0 ... 0 ]
                [ s 0 0 ... 0 ]   (32 rows, 32 columns; only column 0 has value s)
                [ ...           ]
```

For each input tile A_wt:
- `matmul_tiles(A_wt, B) -> DST += A_wt * B`
- Since B has non-zero values only in column 0, `(A_wt * B)[row][0] = sum(A_wt[row][c] * B[c][0] for c in 0..31) = s * sum(A_wt[row][c])`
- All other output columns are 0 because B[c][j]=0 for j>0.

After processing all Wt tiles, `DST[0]` contains `s * sum over all Wt tiles and all 32 columns`.
The output tile has row sums in column 0 and zeros elsewhere.

For SUM: s = 1.0, result = sum of all values per row across all Wt tiles.
For AVG: s = 1/W (set by host), result = mean per row.

### reduce_tile Accumulation (MAX Path)

`reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(input_cb, scaler_cb, itile, itile_scaler, idst)`

This is a hardware-level operation that:
1. Unpacks input tile from `input_cb[itile]` into SRCA
2. Unpacks scaler tile from `scaler_cb[itile_scaler]` into SRCB
3. Performs reduction math: for REDUCE_ROW, reduces each row of the 32x32 input
   tile to a single value, producing a column vector (stored in appropriate positions
   in the destination). The scaler is applied as a multiplicative factor.
4. **Accumulates** into DST[idst]: subsequent calls to reduce_tile with the same
   `idst` continue accumulating (max operation: takes element-wise max with existing DST value)

The key accumulation property: `reduce_tile` does NOT clear DST before operating. Each call
adds to (SUM) or takes max with (MAX) the existing value in DST[idst]. This is how the
Wt tiles are combined: calling reduce_tile Wt times with the same dst_idx accumulates the
row-reduced values from all tiles.

### DST Register Management for Both Paths

**matmul path**: Uses `acquire_dst()` before the Wt loop and `release_dst()` after packing.
This is the simpler full-sync model.

**reduce_tile via helper path**: Uses `tile_regs_acquire()` before the Wt loop and
`tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()` after. This is the
half-sync model where math and pack phases handoff via the commit/wait protocol.

In both cases, DST[0] is the accumulation register. After each row, DST is released
(which implicitly clears it), and a fresh acquisition starts for the next row.

## Implementation Notes

1. **NC is always 1 in compute args**: The host flattens batch/channel into the row count
   assigned per core. The compute kernel's NC loop is technically redundant for this factory
   but maintained for interface consistency with the helper library.

2. **Scaler CB persistence**: The scaler tile is pushed once by the reader and never popped.
   The compute kernel waits for it once and then accesses it by index (tile 0) throughout.
   This works because `reduce_tile` and `matmul_tiles` both access CB tiles by index without
   consuming them -- only `cb_pop_front` removes tiles, and it is never called on CB2.

3. **REDUCE_ROW_SUM_VIA_MM selection**: This define is set in `reduce_op_utils::get_defines()`
   only when `reduce_dim == W && (reduce_op == SUM || reduce_op == AVG)`. For MAX, the
   standard `reduce_tile` path is used. The matmul approach is preferred for SUM/AVG
   reportedly for precision reasons.

4. **Two compute kernel groups**: The program creates two compute kernel instances
   (core_group_1 and core_group_2) with different `Ht` compile-time args to handle
   uneven work distribution. core_group_2 may be empty if work divides evenly.

5. **override_runtime_arguments**: Only buffer addresses are updated on cache hits,
   not tile counts or start IDs. This means the operation can only be cached for
   same-shaped tensors (same NC, Ht, Wt).

6. **Relevance to softmax**: For a softmax operation needing width reduction (dim=-1),
   the matmul_tiles path demonstrates how to sum all tiles along W into a single tile.
   Softmax needs: (a) max reduction along W, (b) sum of exp(x - max) along W. Both
   can use the patterns shown here. The `compute_kernel_lib::reduce` helper with
   `ReduceInputPolicy::WaitAndPopPerTile` and `ReduceInputPolicy::WaitUpfrontNoPop`
   (for tile reuse) are directly applicable.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does matmul_tiles work for reducing a row of tiles along the W dimension? What is the relationship between a scaler tile and the input tile in a row-reduction via matmul?"
   **Reason**: Needed to understand why matmul is used instead of reduce_tile for SUM/AVG
   **Key Findings**: matmul_tiles with a column-vector scaler tile produces row sums. The scaler tile has 1.0 only in column 0 of left faces. The matmul approach is preferred over reduce_tile for precision. Each call accumulates via DST += A * B.

2. **Query**: "What is the reduce_tile function signature in tt-metal compute kernels?"
   **Reason**: Needed exact function signature and accumulation semantics
   **Key Findings**: Query failed; information obtained directly from source file `tt_metal/hw/inc/api/compute/reduce.h`

3. **Query**: "What are tile_regs_acquire/commit/wait/release?"
   **Reason**: Needed to understand DST register management protocol
   **Key Findings**: Query failed; information obtained from METALIUM_GUIDE.md code example showing the acquire/commit/wait/release protocol

4. **Query**: "What is compute_kernel_hw_startup?"
   **Reason**: Needed to understand required initialization
   **Key Findings**: Query failed; information obtained from `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`

### Documentation References

1. **Source**: `tt_metal/hw/inc/api/compute/reduce.h`
   **Reason**: Authoritative API documentation for reduce_init, reduce_tile, reduce_uninit
   **Key Information**: reduce_tile signature `(icb, icb_scaler, itile, itile_scaler, idst)`, accumulates into DST[idst]. Scaler CB must have row 0 of each face filled. reduce_uninit resets packer edge masks.

2. **Source**: `tt_metal/hw/inc/api/compute/matmul.h`
   **Reason**: Authoritative API for mm_init and matmul_tiles
   **Key Information**: `mm_init(in0_cb, in1_cb, out_cb)` initializes unpack+math+pack for matmul. `matmul_tiles(in0_cb, in1_cb, in0_tile_idx, in1_tile_idx, dst_idx)` performs DST += A * B.

3. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`
   **Reason**: Understanding hardware initialization requirements
   **Key Information**: Must be called once at kernel start before any other compute API. Takes (icb0, icb1, ocb) to configure unpack, math, and pack hardware. Cannot be called mid-kernel due to MMIO writes.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` and `.inl`
   **Reason**: Understanding the helper library that wraps reduce_tile with CB management
   **Key Information**: Provides `compute_kernel_lib::reduce<>()` with multiple input policies (WaitAndPopPerTile, BulkWaitBulkPop, WaitUpfrontNoPop, NoWaitNoPop), accumulation support, post-reduce op callbacks. For REDUCE_ROW: processes Wt tiles per row, outputs Ht tiles per batch.

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` and `.inl`
   **Reason**: Understanding scaler tile preparation
   **Key Information**: `prepare_reduce_scaler<cb_id>(scaler_f)` zeros tile then fills row 0 of all faces. `calculate_and_prepare_reduce_scaler` auto-computes scaler based on pool type and dimension.

6. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp`
   **Reason**: Understanding matmul scaler tile layout
   **Key Information**: Creates column-vector scaler tile: zeros entire tile, then fills column 0 of left faces (faces 0 and 2) with single_packed_scalar. Pattern: ptr[i] for i in {0,8,16,...,120} and {256,264,...,376}.

7. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register capacity and DEST_AUTO_LIMIT
   **Key Information**: DEST capacity depends on sync mode (half/full) and accumulation mode (16/32-bit). Half-sync bf16: 8 tiles, Half-sync fp32: 4 tiles. DEST_AUTO_LIMIT is auto-detected constexpr.

8. **Source**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp` (get_defines function)
   **Reason**: Understanding which defines are set for W-dimension reduction
   **Key Information**: W maps to REDUCE_ROW. SUM/AVG+W sets REDUCE_ROW_SUM_VIA_MM=1. MAX+W does not set it.

9. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding DST register management protocol
   **Key Information**: The tile_regs_acquire/commit/wait/release protocol synchronizes math (producer) and pack (consumer) threads sharing DST registers. acquire locks for math, commit signals data ready, wait blocks pack until committed, release frees for next acquire.
