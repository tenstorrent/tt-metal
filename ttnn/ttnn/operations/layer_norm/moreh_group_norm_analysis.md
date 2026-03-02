# Moreh Group Norm Implementation Analysis

## Overview

**moreh_group_norm** implements Group Normalization on Tenstorrent hardware. Given an input tensor of shape `(N, C, H, W)`, it divides the `C` channels into `num_groups` groups and normalizes each group independently across the spatial `(H, W)` dimensions and the channels within that group. Optionally, it applies an affine transformation with learnable parameters gamma (scale) and beta (bias), and can output the per-group mean and reciprocal standard deviation (rstd).

**Program factory**: `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/moreh_group_norm_program_factory.cpp`

The operation reuses the **moreh_layer_norm** compute kernels but with the `is_group_norm` flag set to `true`, which changes how gamma/beta broadcasting works. It has two algorithmic variants:
- **Small algorithm**: All `num_inner_tiles` for one row fit in L1 at once (single read pass for input).
- **Large algorithm**: Input is re-read from DRAM in blocks of `block_size` tiles for each computational phase (three read passes per row).

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Row (one "row" = one group for one batch element) |
| **Unit size** | `num_inner_tiles` = `(C / num_groups) * Ht * Wt` tiles per row |
| **Total units** | `num_rows` = `N * num_groups` |
| **Loop structure** | Outer loop over rows assigned to this core; inner loop over `num_inner_tiles` in blocks of `block_size` |

A single "row" corresponds to all spatial tiles for one group in one batch element. For a tensor of shape `(N, C, H, W)` with `num_groups` groups, each row spans `(C / num_groups) * ceil(H/32) * ceil(W/32)` tiles.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | `[N, C, H, W]` |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (accessed via TensorAccessor) |
| **Data type** | Matches `cb_data_format` (bfloat16 or float32) |

### Gamma Tensor (optional)

| Property | Gamma |
|----------|-------|
| **Logical shape** | `[1, 1, 1, C]` |
| **Dimension convention** | NCHW (broadcast over N, H, W) |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input |

### Beta Tensor (optional)

Same format as Gamma.

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | `[N, C, H, W]` |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input |

### Mean / Rstd Tensors (optional)

| Property | Mean / Rstd |
|----------|-------------|
| **Logical shape** | `[1, 1, N, num_groups]` |
| **Dimension convention** | 2D stored in last two dims |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input |

### Layout Transformations

No tilize/untilize operations are performed. All tensors are expected in TILE_LAYOUT. Padding masks are applied in the compute kernel when `origin_h` or `origin_w` are not multiples of 32, to zero out padded elements so they do not affect the normalization statistics.

## Data Flow Pattern

### Small Algorithm

The small algorithm loads ALL `num_inner_tiles` of a row into L1 at once. The compute kernel can then access tiles by index within the CB without re-reading from DRAM.

**Per-row flow:**

1. **Reader** loads all `num_inner_tiles` into `cb_input` (one bulk read).
2. **Compute** computes `Sum[x]` by iterating over all tiles in `cb_input` (tiles remain in CB).
3. **Compute** reduces `Sum[x]` with scaler to get `E[x]` -> `cb_ex`. Optionally copies to `cb_mean`.
4. **Compute** computes `x - E[x]` for all tiles -> `cb_xmm`. Pops `cb_input`.
5. **Compute** computes `(x - E[x])^2` and accumulates `Sum[(x-E[x])^2]` -> `cb_xmm2sum`.
6. **Compute** reduces to get `Var[x]` -> `cb_var`.
7. **Compute** computes `1/sqrt(Var[x] + eps)` -> `cb_recip_std`. Optionally copies to `cb_rstd`.
8. **Compute** computes `(x - E[x]) * recip_std` -> output (or intermediate if gamma/beta apply).
9. **Reader** loads gamma/beta tiles (in blocks) -> `cb_gamma`, `cb_beta`.
10. **Compute** applies affine: `result * gamma + beta` -> `cb_out`.
11. **Writer** reads `cb_mean`, `cb_rstd`, `cb_out` and writes to DRAM.

### Large Algorithm

The large algorithm cannot hold all tiles in L1, so input is re-read from DRAM three times per row:

1. **Reader Pass 1**: Reads input in blocks of `block_size` for `Sum[x]` computation.
2. **Compute**: Accumulates `Sum[x]`, then reduces to `E[x]`.
3. **Reader Pass 2**: Re-reads input in blocks for `Var[x]` computation.
4. **Compute**: Computes `x - E[x]`, squares, accumulates variance. Masks applied inline.
5. **Reader Pass 3**: Re-reads input in blocks for final normalization + affine.
6. **Compute**: Computes `(x - E[x]) * recip_std`, applies gamma/beta.
7. **Reader** interleaves gamma/beta reads during Pass 3.
8. **Writer**: Outputs results to DRAM.

## Circular Buffer Configuration

### Small Algorithm CB Sizes

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|--------------------|-----------| ---------|----------|----------|
| c_0 | cb_input | Input tiles | `num_inner_tiles` | `num_inner_tiles` | Single | Reader | Compute | Row |
| c_1 | cb_scaler | Normalization scaler | 1 | 1 | Single | Reader | Compute | Program |
| c_2 | cb_eps | Epsilon value | 1 | 1 | Single | Reader | Compute | Program |
| c_3 | cb_gamma | Gamma (scale) | `block_size` (if present, else 0) | `block_size` | Single | Reader | Compute | Block |
| c_4 | cb_beta | Beta (bias) | `block_size` (if present, else 0) | `block_size` | Single | Reader | Compute | Block |
| c_5 | cb_mask_h | H-dimension mask | 1 (if needed, else 0) | 1 | Single | Reader | Compute | Program |
| c_6 | cb_mask_w | W-dimension mask | 1 (if needed, else 0) | 1 | Single | Reader | Compute | Program |
| c_16 | cb_out | Normalized output | `block_size` | `block_size` | Single | Compute | Writer | Block |
| c_17 | cb_mean | Mean output | 1 (if requested, else 0) | 1 | Single | Compute | Writer | Row |
| c_18 | cb_rstd | Rstd output | 1 (if requested, else 0) | 1 | Single | Compute | Writer | Row |
| c_24 | cb_ex | E[x] (mean) | 1 | 1 | Single | Compute | Compute | Row |
| c_25 | cb_xmm | x - E[x] | `num_inner_tiles` | `block_size` | Single | Compute | Compute | Row |
| c_26 | cb_xmm2 | (x - E[x])^2 | 1 | 1 | Single | Compute | Compute | Block |
| c_27 | cb_xmm2sum | Sum[(x-E[x])^2] | 1 | 1 | Single | Compute | Compute | Row |
| c_28 | cb_var | Var[x] | 1 | 1 | Single | Compute | Compute | Row |
| c_29 | cb_recip_std | 1/sqrt(Var+eps) | 1 | 1 | Single | Compute | Compute | Row |
| c_30 | cb_gamma_beta | Intermediate for affine | `2 * block_size` (if gamma or beta, else 0) | `block_size` | Double | Compute | Compute | Block |
| c_31 | cb_xsum | Sum[x] | 2 | 1 | Double | Compute | Compute | Row |

### Large Algorithm Differences

When `use_large_algorithm` is true, the following CB sizes change:
- `cb_input` (c_0): reduced from `num_inner_tiles` to `block_size`
- `cb_xmm` (c_25): reduced from `num_inner_tiles` to `2 * block_size`
- `cb_xmm2` (c_26): increased from 1 to `2 * block_size`

## Pipeline Pattern Summary

All CBs use **single-buffered** patterns (capacity equals block size) except:
- `cb_gamma_beta` (c_30): **double-buffered** (`2 * block_size` capacity, `block_size` block). This allows overlapping the gamma multiplication output with the beta addition input.
- `cb_xsum` (c_31): **double-buffered** (capacity=2, block=1). Used for running accumulation.
- In large algorithm: `cb_xmm` and `cb_xmm2` become double-buffered (`2 * block_size` capacity, `block_size` block).

Since all input/output CBs are single-buffered, there is no overlap between DRAM reads/writes and compute. The compute kernel blocks until the reader finishes, and the writer blocks until compute finishes.

## Index Calculations

### Input Tile Indexing

Input tiles are addressed linearly in NCHW order:
```
input_tile_idx = n * C * Ht * Wt + c * Ht * Wt + h * Wt + w
```

Each core is assigned a contiguous range of rows via `tile_offset`. Within each row:
```
input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx
```

### Gamma/Beta Tile Indexing

Gamma and beta have shape `(1, 1, 1, C)`. The reader must map from the input tile index to the corresponding channel index:

```cpp
// get_gamma_beta_tile_idx: which tile in gamma/beta to read
c_idx = (input_tile_idx / HtWt) % C;   // extract channel index
tile_idx = c_idx / TILE_W;             // which 32-wide tile
```

After reading the gamma/beta tile, the reader extracts the correct scalar value within the tile using `get_tilized_gamma_beta_idx_in_tile`, which computes the tilized offset for the channel's position within the tile:
```cpp
w_idx_in_tile = c_idx % TILE_W;
tilized_idx = get_tilized_idx(0, w_idx_in_tile, TILE_H, TILE_W);
```

The value at `tilized_idx` is then moved to position 0 in the tile, so the compute kernel can use `bcast_scalar` operations.

### Mean/Rstd Tile Indexing

Mean and rstd have shape `(1, 1, N, num_groups)`. The writer computes:
```cpp
mean_rstd_idx = start_mean_rstd_idx + outer_idx;  // linear index into (N x num_groups)
mean_rstd_n_idx = mean_rstd_idx / num_groups;      // batch index -> row in tile
mean_rstd_g_idx = mean_rstd_idx % num_groups;      // group index -> col in tile
```

It then computes the tile coordinates and the tilized position within that tile using `get_tilized_idx`. The writer performs a **single-element write** -- it writes only the specific bfloat16 (2-byte) element within the tile at the computed tilized offset. This is because multiple rows/groups may map to the same tile, and each core only writes its assigned element.

## Memory Access Patterns

### Read Pattern

**Input**: Sequential tile reads within each row. Each tile is read by its linear index using `noc_async_read_tile` through a `TensorAccessor`. In the small algorithm, all tiles for a row are bulk-read once. In the large algorithm, the same tiles are read three times (once per computational phase) in blocks of `block_size`.

**Gamma/Beta**: Non-sequential. For each block of input tiles, the corresponding gamma/beta tile is determined by the channel index. Multiple input tiles sharing the same channel share the same gamma/beta tile, but tiles are re-read for each block. Within each tile, a scalar extraction pattern moves the relevant channel value to position 0.

### Write Pattern

**Output**: Sequential tile writes in blocks of `block_size`, matching the order of input tiles.

**Mean/Rstd**: Single-element writes per row. Each write targets a specific 2-byte location within a tile, using a computed tilized index. This is a fine-grained sub-tile write pattern.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major traversal) |
| **Grid dimensions** | Up to `grid.x` x `grid.y` (device compute grid) |
| **Total cores** | `num_cores_to_be_used` (at most `grid.x * grid.y`) |
| **Work per core** | `num_rows_per_core_group_1` or `num_rows_per_core_group_2` rows |
| **Load balancing** | Two-group split via `split_work_to_cores` |

The total work is `num_rows = N * num_groups`. The `split_work_to_cores` utility divides rows across available cores, producing two core groups:
- **core_group_1**: cores with `num_rows_per_core_group_1` rows each
- **core_group_2**: cores with `num_rows_per_core_group_2` rows each (handles remainder; may be empty)

Cores are enumerated in column-major order: `core = {i / num_cores_y, i % num_cores_y}`.

Each core receives a `tile_offset` that advances by `num_rows_per_core * num_inner_tiles` for successive cores, ensuring non-overlapping tile ranges.

## Arguments

### Reader Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | gamma_has_value | uint32_t (bool) | Whether gamma tensor is provided |
| 1 | beta_has_value | uint32_t (bool) | Whether beta tensor is provided |
| 2+ | input TensorAccessorArgs | multiple | Accessor params for input tensor (bank mapping, page size, etc.) |
| N+ | gamma TensorAccessorArgs | multiple | Accessor params for gamma tensor |
| M+ | beta TensorAccessorArgs | multiple | Accessor params for beta tensor |

### Reader Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_addr | uint32_t | Base address of input tensor in DRAM |
| 1 | gamma_addr | uint32_t | Base address of gamma tensor (0 if absent) |
| 2 | beta_addr | uint32_t | Base address of beta tensor (0 if absent) |
| 3 | scaler | uint32_t | Bit-cast float: `1.0 / (TILE_W * sqrt((C/G) * Ht_orig * Wt_orig))` |
| 4 | eps | uint32_t | Bit-cast float: epsilon for numerical stability |
| 5 | tile_offset | uint32_t | Starting tile index for this core's work |
| 6 | num_rows_per_core | uint32_t | Number of rows (groups) this core processes |
| 7 | num_inner_tiles | uint32_t | Tiles per row: `(C/G) * Ht * Wt` |
| 8 | num_channels | uint32_t | Total number of channels `C` |
| 9 | origin_h | uint32_t | Original (unpadded) height |
| 10 | origin_w | uint32_t | Original (unpadded) width |
| 11 | block_size | uint32_t | Tile block size for inner loop (1..8) |

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | mean_has_value | uint32_t (bool) | Whether mean output is requested |
| 1 | rstd_has_value | uint32_t (bool) | Whether rstd output is requested |
| 2+ | output TensorAccessorArgs | multiple | Accessor params for output tensor |
| N+ | mean TensorAccessorArgs | multiple | Accessor params for mean tensor |
| M+ | rstd TensorAccessorArgs | multiple | Accessor params for rstd tensor |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_addr | uint32_t | Base address of output tensor |
| 1 | mean_addr | uint32_t | Base address of mean tensor (0 if absent) |
| 2 | rstd_addr | uint32_t | Base address of rstd tensor (0 if absent) |
| 3 | tile_offset | uint32_t | Starting tile index for this core |
| 4 | num_rows_per_core | uint32_t | Number of rows this core processes |
| 5 | num_inner_tiles | uint32_t | Tiles per row |
| 6 | num_groups | uint32_t | Number of groups (for mean/rstd indexing) |
| 7 | block_size | uint32_t | Tile block size |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_rows_per_core | uint32_t | Rows for this core group |
| 1 | origin_H | uint32_t | Original (unpadded) height |
| 2 | origin_W | uint32_t | Original (unpadded) width |
| 3 | num_inner_tiles | uint32_t | Tiles per row |
| 4 | block_size | uint32_t | Block size for tiled processing |
| 5 | gamma_has_value | uint32_t (bool) | Whether gamma applies |
| 6 | beta_has_value | uint32_t (bool) | Whether beta applies |
| 7 | mean_has_value | uint32_t (bool) | Whether to output mean |
| 8 | rstd_has_value | uint32_t (bool) | Whether to output rstd |
| 9 | is_lastdim_layernorm | uint32_t (bool) | Always 0 for group norm |
| 10 | is_group_norm | uint32_t (bool) | Always 1 for group norm |

### Compute Defines

| Define | Value | Purpose |
|--------|-------|---------|
| `REDUCE_OP` | `PoolType::AVG` | Reduction type for scalar reduce |
| `REDUCE_DIM` | `ReduceDim::REDUCE_SCALAR` | Reduces entire tile to scalar |

## Kernel Implementations

### Reader Kernel (Small)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_small | RISCV_0 | NOC0 | DRAM (input, gamma, beta) | c_0, c_1, c_2, c_3, c_4, c_5, c_6 | Read input tiles, fill scaler/eps, generate masks, read gamma/beta with scalar extraction |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/dataflow/reader_moreh_group_norm_small.cpp`
- **Key Logic**:
  - Fills `cb_scaler` and `cb_eps` with constant values using `fill_cb_with_value`.
  - Generates H/W masks using `generate_mask_h` / `generate_mask_w` if padding exists.
  - Reads ALL `num_inner_tiles` into `cb_input` at once (bulk load).
  - For gamma/beta: reads the tile containing the relevant channel, then extracts the scalar at the tilized position and moves it to position 0 in the tile. This allows the compute kernel to use `bcast_scalar` operations.

### Reader Kernel (Large)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_large | RISCV_0 | NOC0 | DRAM (input, gamma, beta) | c_0, c_1, c_2, c_3, c_4, c_5, c_6 | Three-pass input read, fill scaler/eps, generate masks, read gamma/beta |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/dataflow/reader_moreh_group_norm_large.cpp`
- **Key Logic**:
  - Same initialization as small (scaler, eps, masks).
  - Reads input tiles THREE times per row, each time in blocks of `block_size`:
    1. Pass 1: For `Sum[x]` computation
    2. Pass 2: For `Var[x]` computation
    3. Pass 3: For final normalization, interleaved with gamma/beta reads
  - Gamma/beta extraction logic is identical to the small reader.

### Compute Kernel (Small)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute_small | RISCV_2 | N/A | c_0, c_1, c_2, c_3, c_4, c_5, c_6 | c_16, c_17, c_18 | Full group norm computation |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_small_kernel.cpp` (shared with layer_norm)
- **Key Logic**: see "How Mean and Variance Are Computed" and "How Affine Transform Is Applied" sections below.
- In small mode, `cb_x` holds all `num_inner_tiles` simultaneously. The kernel reads tiles by index within the CB. After computing `x - E[x]` (stored in `cb_xmm` for all tiles), it computes variance from `cb_xmm` directly without re-reading input.

### Compute Kernel (Large)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute_large | RISCV_2 | N/A | c_0, c_1, c_2, c_3, c_4, c_5, c_6 | c_16, c_17, c_18 | Full group norm computation (streaming) |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp` (shared with layer_norm)
- **Key Logic**: Same math as small, but processes input in blocks of `block_size` and pops `cb_x` after each block. Requires three separate read passes from DRAM (coordinated with the large reader kernel):
  1. First pass: accumulate `Sum[x]` block by block, popping input after each block.
  2. Second pass: re-read input, compute `x - E[x]`, apply masks, square, accumulate variance. `cb_xmm` is produced and consumed in blocks.
  3. Third pass: re-read input, recompute `x - E[x]`, multiply by `recip_std`, apply gamma/beta.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | c_16, c_17, c_18 | DRAM (output, mean, rstd) | Write output tiles, write mean/rstd scalars |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/dataflow/writer_moreh_group_norm.cpp`
- **Key Logic**:
  - For each row, first writes mean and rstd (if requested), then writes output tiles in blocks.
  - Mean/rstd writes are **sub-tile**: only 1 element (2 bytes for bfloat16) is written per tile. The writer computes the tilized index within the `(N, num_groups)` output tile and writes at that specific byte offset.
  - The scalar value produced by compute is at position 0 in the tile. The writer moves it to the correct tilized position before writing: `rstd_ptr[tilized_idx] = rstd_ptr[0]`.

## How Mean and Variance Are Computed

### Step 1: Sum[x]

For each row (group), the compute kernel accumulates a running sum of all input tiles into `cb_xsum`. The first tile is copied directly; subsequent tiles are added using `add_tiles`. If the original H or W is not a multiple of 32, mask tiles are applied (element-wise multiply with 0/1 mask) to zero out padding before summing.

### Step 2: E[x] (Mean)

The accumulated tile-sum in `cb_xsum` is reduced to a scalar using `reduce_tile` with the scaler:
```
scaler = 1.0 / (TILE_W * sqrt((C/G) * Ht_orig * Wt_orig))
```

The `reduce_tile` with `REDUCE_SCALAR` and `PoolType::AVG` sums all elements within the tile and multiplies by the scaler. This two-step process (first sum across tiles, then scalar-reduce within the accumulated tile) computes the mean across all elements in the group.

Note: The scaler incorporates both the tile-internal reduction factor (`1/TILE_W` comes from the AVG reduce) and the cross-element normalization factor. The `sqrt` in the denominator accounts for the fact that `reduce_tile` applies the scaler in a specific way related to the AVG pooling implementation.

### Step 3: x - E[x]

Each input tile has the mean subtracted using `sub_tiles_bcast_scalar` (for group norm). This broadcasts the scalar mean across all elements of each tile.

### Step 4: Var[x] = E[(x - E[x])^2]

The centered tiles are squared element-wise (`mul_tiles` of `cb_xmm` with itself), producing `cb_xmm2`. These are accumulated into `cb_xmm2sum` using the same running-sum pattern. Finally, `reduce_tile` with the same scaler converts the sum to variance.

### Step 5: Reciprocal Standard Deviation

```
recip_std = 1.0 / sqrt(Var[x] + eps)
```

Computed as: add eps to variance (`add_tiles`), then apply `rsqrt_tile` (hardware reciprocal square root on SFPU).

## How the Affine Transform (Gamma/Beta) Is Applied

### Gamma/Beta Data Preparation (Reader)

Gamma and beta have shape `(1, 1, 1, C)`. For each block of input tiles, the reader:

1. Computes which gamma/beta tile to read based on the channel index: `tile_idx = c_idx / TILE_W`.
2. Reads the tile from DRAM.
3. Extracts the specific channel's scalar value from its tilized position within the tile.
4. Moves that value to position 0 (element `[0][0]` in the tile).

This preparation allows the compute kernel to use efficient broadcast-scalar operations.

### Gamma Multiplication (Compute)

For group norm (`is_groupnorm = true`), the kernel uses:
```cpp
mul_tiles_bcast_scalar(cb_gamma_beta_or_out, cb_gamma, j, j, j);
```

This multiplies every element of the normalized tile by the scalar at position 0 of the gamma tile. This is correct because within a single input tile, all elements belong to the same channel (or the reader has extracted the right per-channel scalar).

### Beta Addition (Compute)

Similarly:
```cpp
add_tiles_bcast_scalar(cb_gamma_beta, cb_beta, j, j, j);
```

Adds the beta scalar to every element of the scaled tile.

### CB Routing for Affine

The compute kernel uses a conditional routing strategy:
- If neither gamma nor beta: normalized result goes directly to `cb_out`.
- If gamma only: result goes to `cb_gamma_beta`, gamma multiply writes to `cb_out`.
- If both: result goes to `cb_gamma_beta`, gamma multiply stays in `cb_gamma_beta`, beta add writes to `cb_out`.
- If beta only: result goes to `cb_gamma_beta`, beta add writes to `cb_out`.

## Memory Management Patterns

### Algorithm Selection

The program factory computes total CB usage and compares against available L1:
```cpp
const auto cb_usage = (sum of all CB tile counts) * single_tile_size;
const auto available_L1 = device->l1_size_per_core() - base_allocator_addr;
const bool use_large_algorithm = cb_usage >= available_L1;
```

If the small algorithm's CB requirements exceed L1, the large algorithm is selected, which reduces `cb_input` from `num_inner_tiles` to `block_size`, and reduces `cb_xmm` from `num_inner_tiles` to `2 * block_size`, at the cost of reading input from DRAM three times.

### Block Size Selection

`block_size` is the largest power of 2 (up to 8) that evenly divides `num_inner_tiles`. This ensures clean loop boundaries without remainder handling in the inner loops.

```cpp
for (uint32_t current_block_size = 8; current_block_size >= 1; current_block_size >>= 1) {
    if (num_inner_tiles % current_block_size == 0) { block_size = current_block_size; break; }
}
```

### CB as Temporary Storage

The compute kernel reuses `cb_ex` as a temporary buffer (`cb_tmp`) during the Sum[x] accumulation phase. When `w_idx != 0`, a masked input tile is packed into `cb_ex` (acting as temp), then immediately added to `cb_xsum`. This avoids allocating a separate CB for temporary values.

### Sub-tile Mean/Rstd Writes

Mean and rstd output tensors are written element-by-element (2 bytes per write) rather than tile-by-tile. This is because each core computes only one scalar per row, but the output tile may contain values from multiple batch elements and groups. The writer computes the exact byte offset within the destination tile and performs a targeted `noc_async_write` of just that element.

### Program Caching

The operation supports program caching via `override_runtime_arguments`, which updates only buffer addresses (indices 0-2 in reader/writer runtime args) without recreating the program. This avoids recompilation when tensor addresses change between invocations.

## Implementation Notes

1. **Shared compute kernel with layer_norm**: The compute kernel (`moreh_layer_norm_small/large_kernel.cpp`) is shared between moreh_layer_norm and moreh_group_norm. The `is_groupnorm` flag controls broadcasting behavior: group norm uses `bcast_scalar` for gamma/beta (since gamma/beta values are pre-extracted to scalar position by the reader), while layer norm uses `bcast_rows` or element-wise operations.

2. **Three-pass DRAM access in large algorithm**: The large algorithm trades memory for bandwidth -- it reads the entire input three times from DRAM. This is necessary because intermediate results (`x - E[x]`) cannot all fit in L1.

3. **Masking strategy**: Padding masks are applied at tile boundaries where `origin_h` or `origin_w` don't align to 32. The `need_to_do_mask_h` function checks if a tile is at the last row of H-tiles, and mask_w is applied when a tile is at the last column of W-tiles.

4. **Scaler value derivation**: The scaler `1.0 / (TILE_W * sqrt((C/G) * Ht_orig * Wt_orig))` accounts for the number of valid elements in the normalization group. The `TILE_W` factor is related to how `reduce_tile` with `PoolType::AVG` internally handles the reduction. The `sqrt` compensates for the reduce being applied twice (once for sum, once for variance).

5. **Column-major core enumeration**: Cores are assigned work in column-major order (`core = {i / num_cores_y, i % num_cores_y}`), which means work fills down columns first, then across rows.

6. **Redundant cb_reserve_back in small reader**: The small reader has `cb_reserve_back(cb_id_input, num_inner_tiles)` both at the outer loop level (line 83) and inside the inner loop (line 85). The inner reserve is redundant since the outer already reserved all tiles. This appears to be a minor code issue that does not affect correctness because `cb_reserve_back` is a no-op when space is already available.

## External Knowledge Sources

### Documentation References

1. **Source**: `METALIUM_GUIDE.md` (referenced implicitly)
   **Reason**: Understanding Tensix core architecture, RISC-V thread assignment, NoC0/NoC1 conventions, and circular buffer semantics.
   **Key Information**: Reader runs on RISC-V data movement processor 0 (NOC0), writer on processor 1 (NOC1), compute on the 3 compute threads (unpack/math/pack). CBs synchronize via `cb_reserve_back`/`cb_push_back` (producer) and `cb_wait_front`/`cb_pop_front` (consumer).

2. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: Understanding helper functions `fill_cb_with_value`, `get_gamma_beta_tile_idx`, `get_tilized_gamma_beta_idx_in_tile`, `get_tilized_idx`, `generate_mask_h`, `generate_mask_w`.
   **Key Information**: These functions handle constant-filling CBs, mapping channel indices to gamma/beta tile positions, converting row-major indices to tilized (face-interleaved) layout indices, and generating binary masks for H/W padding.

3. **Source**: `ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.hpp`
   **Reason**: Understanding `CreateCircularBuffer`, `CreateReadKernel`, `CreateWriteKernel`, `CreateComputeKernel` helper wrappers.
   **Key Information**: These are convenience wrappers around tt-metal APIs. `CreateCircularBuffer` takes a vector of `{CBIndex, num_tiles}` pairs. `CreateReadKernel`/`CreateWriteKernel` create data movement kernels on the appropriate RISC-V processors. `CreateComputeKernel` supports per-core-group compile-time args.

4. **Source**: `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/moreh_group_norm_device_operation.hpp`
   **Reason**: Understanding operation attributes, tensor args, and shared variables for program caching.
   **Key Information**: Operation takes `num_groups`, `eps`, optional gamma/beta/mean/rstd tensors. Shared variables cache kernel handles and core counts for runtime arg override.
