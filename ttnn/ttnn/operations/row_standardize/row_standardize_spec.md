# Row Standardize Functional Specification

## Overview
- **Operation Name**: `row_standardize`
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), softmax (compute_core), untilize (output_stage)
- **Reference Analyses**:
  - `ttnn/ttnn/operations/row_standardize/tilize_analysis.md` (role: input_stage)
  - `ttnn/ttnn/operations/row_standardize/softmax_analysis.md` (role: compute_core)
  - `ttnn/ttnn/operations/row_standardize/untilize_analysis.md` (role: output_stage)

## Mathematical Definition

### Formula
```
mean_row = (1/W) * sum(x[..., j] for j in 0..W-1)
var_row  = (1/W) * sum((x[..., j] - mean_row)^2 for j in 0..W-1)
output[..., i] = (x[..., i] - mean_row) * rsqrt(var_row + epsilon)
```

### Semantic Description
Row standardize computes per-row (last dimension) standardization of an input tensor. For each row, it subtracts the row mean and divides by the row standard deviation (with an epsilon for numerical stability). This is equivalent to Layer Normalization without learnable affine parameters (no gamma/beta), applied along dim=-1.

The PyTorch reference is:
```python
(x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, unbiased=False, keepdim=True) + epsilon)
```

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | At least 2D, W and H multiples of 32 | - | Input tensor in ROW_MAJOR layout |
| epsilon | float | No | > 0 | 1e-5 | Small constant for numerical stability in rsqrt |

### Input Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Rank | Must be >= 2 | "Input tensor must have rank >= 2" |
| Layout | ROW_MAJOR | "Input tensor must be in ROW_MAJOR layout" |
| Dtype | bfloat16 or float32 | "Input tensor must be bfloat16 or float32" |
| Device | Must be on device | "Input tensor must be on device" |
| Memory | INTERLEAVED | "Input tensor must use interleaved memory" |
| Last dim (W) | Must be a multiple of 32 | "Last dimension must be a multiple of 32 (tile width)" |
| Second-to-last dim (H) | Must be a multiple of 32 | "Second-to-last dimension must be a multiple of 32 (tile height)" |

### Output Tensor Specification
- **Shape**: Same as input tensor
- **Dtype**: Same as input tensor (bfloat16 or float32)
- **Layout**: ROW_MAJOR
- **Memory**: INTERLEAVED on DRAM

## Component Sources (Hybrid Mode)

This operation is composed from three references. The overall data flow is:
```
RM input --> [Reader: read sticks] --> [Compute: tilize -> row_standardize -> untilize] --> [Writer: write sticks] --> RM output
```

### Input Stage (from tilize reference)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Reader kernel | tilize.reader (`reader_unary_stick_layout_split_rows_interleaved.cpp` pattern) | Adapted: reads RM sticks from DRAM into `cb_rm_in`. Also generates scalar tiles (reduce scaler 1/W into `cb_scaler`, epsilon scalar into `cb_eps`). |
| CB_rm_in (c_0) | tilize.CB_c_0 | Same sizing: Wt pages capacity per block (holds 32 sticks = one tile-row). Page size = tile_size for the dtype. |
| Compute (tilize phase) | tilize.compute (`tilize_block`) | Reused via `compute_kernel_lib::tilize<>()`. Converts 32 sticks in cb_rm_in to Wt tiles in cb_tilized. |

### Compute Stage (from softmax reference, adapted)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Row SUM reduce (mean) | softmax.Phase1 (MAX reduce pattern) | Changed from MAX to SUM with `WaitUpfrontNoPop` policy. Uses `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>`. Scaler = 1/W (not 1.0). Output in cb_mean. |
| Subtract mean (x - mean) | softmax.Phase2 (sub_tiles_bcast COL) | Identical pattern: `sub<COL, WaitUpfrontNoPop>` from binary_op_helpers. Subtracts cb_mean column vector from each tile in cb_tilized. Output in cb_xmm. |
| Square (x-mean)^2 | New (not in softmax) | Uses `square<>` from binary_op_helpers on cb_xmm tiles. Output in cb_xmm_sq. Or in-place: compute (x-mean)^2 using mul_tiles of cb_xmm with itself. |
| Row SUM reduce (variance) | softmax.Phase4 (SUM reduce pattern) | SUM reduce with scaler 1/W over cb_xmm_sq. Uses `compute_kernel_lib::reduce<SUM, REDUCE_ROW>`. Output in cb_var. |
| Add epsilon + rsqrt | New (not in softmax, replaces recip) | Add epsilon to cb_var using `add_tiles_bcast_scalar`, then `rsqrt_tile`. Output in cb_invstd. |
| Final normalize | softmax.Phase5 (mul_tiles_bcast COL) | `mul<COL>` from binary_op_helpers: multiply cb_xmm by cb_invstd. Output in cb_tilized_out (reuse cb_tilized or new CB). |

### Output Stage (from untilize reference)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Compute (untilize phase) | untilize.compute (`untilize_block` / `pack_untilize_block`) | Reused via `compute_kernel_lib::untilize<>()`. Converts Wt tiles in cb_tilized_out to RM sticks in cb_rm_out. |
| CB_rm_out (c_16) | untilize.CB_c_16 | Same sizing: Wt pages capacity per block. Page size = tile_size. |
| Writer kernel | untilize.writer (`writer_unary_stick_layout_split_rows_multi_core.cpp` pattern) | Adapted: writes RM sticks from cb_rm_out to DRAM. Row-by-row stick writes via TensorAccessor. |

### Interface Compatibility
| Interface | Component A | Component B | Format A | Format B | Compatible? |
|-----------|------------|-------------|----------|----------|-------------|
| Reader -> Compute (tilize) | reader.cb_rm_in | compute.tilize_block | RM sticks (32 x W bytes) | RM sticks in CB pages | Yes (same as tilize reference) |
| Compute (tilize) -> Compute (standardize) | tilize_block.cb_tilized | reduce/sub/mul pipeline | Tile format (Wt tiles) | Tile format (Wt tiles) | Yes (all compute ops work on tiles) |
| Compute (standardize) -> Compute (untilize) | normalize.cb_tilized_out | untilize_block.input | Tile format (Wt tiles) | Tile format (Wt tiles) | Yes |
| Compute (untilize) -> Writer | untilize_block.cb_rm_out | writer.cb_rm_out | RM sticks (32 rows x W) | RM sticks | Yes (same as untilize reference) |

### CB ID Resolution
| Logical CB | Source Ref | Original ID | Final ID | Notes |
|------------|-----------|-------------|----------|-------|
| cb_rm_in | tilize | c_0 | c_0 | Input: RM sticks from reader, consumed by tilize phase |
| cb_scaler | softmax | c_2 | c_1 | Reduce scaler tile (1/W), generated by reader, persistent |
| cb_eps | New | - | c_2 | Epsilon scalar tile, generated by reader, persistent |
| cb_tilized | tilize->softmax | c_16->c_0 | c_3 | Tilized input tiles, produced by tilize phase, consumed by compute |
| cb_mean | softmax | c_26 | c_24 | Per-row mean (column vector), 1 tile |
| cb_xmm | softmax | c_27 | c_25 | x - mean intermediate, Wt tiles |
| cb_xmm_sq | New | - | c_26 | (x-mean)^2 intermediate, Wt tiles |
| cb_var | softmax | c_25 | c_27 | Per-row variance (column vector), 1 tile |
| cb_invstd | New | - | c_28 | rsqrt(var + eps), 1 tile |
| cb_tilized_out | softmax->untilize | c_16->c_0 | c_4 | Normalized tiles before untilize, Wt tiles |
| cb_rm_out | untilize | c_16 | c_16 | Output: RM sticks for writer |

## Design Decisions

### Decision 1: Single Compute Kernel (tilize + standardize + untilize combined)
- **Choice**: Use a single compute kernel that performs tilize, all row-standardize compute phases, and untilize sequentially for each block (tile-row).
- **Rationale**: Keeping tilize/untilize in the compute kernel allows the RM data to be tilized, processed, and untilized entirely within L1 without ever writing tiles to DRAM. This avoids 2 extra DRAM round-trips (tilized intermediate to DRAM and back). The softmax reference shows that multi-phase compute within a single kernel is standard practice.
- **Alternatives Considered**: (1) Separate tilize/compute/untilize operations chained via DRAM -- rejected due to unnecessary DRAM bandwidth. (2) Reader does tilize via DMA -- not supported, tilize requires compute hardware.
- **Tradeoffs**: Single kernel is more complex but far more efficient. CB count is higher but all fit within L1 for reasonable W dimensions.

### Decision 2: Single-Core Prototype
- **Choice**: Start with single-core execution (all tile-rows processed by one core).
- **Rationale**: Simplifies the prototype significantly. Multi-core distribution (like tilize/untilize references) can be added later by distributing tile-rows across cores. The softmax reference's `split_work_to_cores` pattern is directly applicable for future multi-core support.
- **Alternatives Considered**: Multi-core from the start -- deferred to avoid complexity in the prototype phase.
- **Tradeoffs**: Lower performance for large tensors, but correct behavior for all shapes. Multi-core extension is straightforward since each tile-row is independent.

### Decision 3: Row-Wise Processing (one tile-row at a time)
- **Choice**: Process one tile-row (Wt tiles spanning the full W dimension) at a time, matching the softmax reference pattern.
- **Rationale**: Row standardize is a per-row operation. Processing one complete tile-row ensures all W-dimension tiles are available for the reduction operations (sum for mean, sum for variance). This matches the softmax w_small pattern exactly.
- **Alternatives Considered**: Processing individual tiles -- impossible since reductions span the full W dimension.
- **Tradeoffs**: Requires CBs large enough to hold Wt tiles. For very large W, this may exceed L1. A "large-W" variant (like softmax's w_large) could be added later.

### Decision 4: Reduce Scaler = 1/W
- **Choice**: Use `1/W` as the reduce scaler for both SUM reductions (mean and variance), where W is the logical (unpadded) last dimension.
- **Rationale**: The hardware reduce operation multiplies each reduced element by the scaler. Using `1/W` directly computes the mean: `scaler * sum(x) = (1/W) * sum(x) = mean`. Similarly for variance: `(1/W) * sum((x-mean)^2) = var`. This is the standard pattern used in softmax (where scaler is 1.0 for plain sum).
- **Alternatives Considered**: Using scaler=1.0 and dividing afterward -- adds an extra mul operation.
- **Tradeoffs**: Requires computing `1/W` on the host and packing it into bfloat16 format. The packed format is `(bf16 << 16 | bf16)` as expected by `generate_reduce_scaler()`.

### Decision 5: Epsilon as Scalar Broadcast Tile
- **Choice**: Generate an epsilon scalar broadcast tile in the reader kernel using `generate_bcast_scalar_bfloat16` (or float32 variant), then use `add_tiles_bcast_scalar` in compute to add epsilon to the variance tile.
- **Rationale**: The variance tile from the reduce is a column vector (per-row variance values). Adding epsilon as a scalar broadcast ensures every element of the variance tile gets epsilon added. The `add_tiles_bcast_scalar` hardware operation handles this efficiently.
- **Alternatives Considered**: (1) Fused `add_rsqrt_tile` -- exists but unclear if it handles the scalar broadcast correctly for column-vector inputs. Safer to use separate add + rsqrt. (2) Pass epsilon as a runtime arg and construct the tile in compute -- compute kernels should not do data generation, reader is the appropriate place.
- **Tradeoffs**: Requires one extra CB (cb_eps) and one extra reader step, but is clean and correct.

### Decision 6: Epsilon Passed as Runtime Argument
- **Choice**: Epsilon is a runtime argument to the reader kernel (not compile-time).
- **Rationale**: The user specifies epsilon per call, and it may vary between calls. Making it compile-time would require recompilation for each epsilon value, defeating program caching. It is passed as a `uint32_t` containing the reinterpreted float bits (for float32) or packed bfloat16 (for bfloat16).
- **Alternatives Considered**: Compile-time arg -- rejected because epsilon is a user-facing parameter that should be runtime.
- **Tradeoffs**: Minimal overhead; reader generates the scalar tile once per program invocation.

### Decision 7: CB Sizing Strategy
- **Choice**: All tile-row CBs (cb_tilized, cb_xmm, cb_xmm_sq, cb_tilized_out) have capacity = Wt tiles. Scalar CBs (cb_mean, cb_var, cb_invstd) have capacity = 1 tile. Persistent CBs (cb_scaler, cb_eps) have capacity = 1 tile.
- **Rationale**: Matches the softmax w_small pattern exactly. One full tile-row must be resident for reductions. Scalar results are single tiles. The scaler and epsilon are generated once and persist for the entire program.
- **Alternatives Considered**: Double-buffering -- not needed for single-core prototype where there is no overlap opportunity.
- **Tradeoffs**: Total CB memory = approximately `(4*Wt + 3 + 2 + Wt + Wt) * tile_size = (7*Wt + 5) * tile_size`. For Wt=32 (W=1024) with bfloat16: `(224+5)*2048 = ~469KB`. This fits within the L1 512KB budget. For larger W, a large-W variant would be needed.

### Decision 8: Dtype-Aware CB Page Sizes
- **Choice**: CB page sizes are computed based on input dtype: `tile_size = 32 * 32 * datum_size` where `datum_size` is 2 for bfloat16 and 4 for float32.
- **Rationale**: The hardware requires correct page sizes for tile read/write operations. Using the wrong page size causes data corruption.
- **Alternatives Considered**: None -- this is a hard requirement.
- **Tradeoffs**: Float32 uses 2x the CB memory of bfloat16, limiting the maximum W for float32.

### Decision 9: FP32 Dest Accumulation for Float32 Input
- **Choice**: When input dtype is float32, enable `fp32_dest_acc_en` in the compute kernel configuration. All intermediate CBs use Float32 data format.
- **Rationale**: Required for correctness with float32 inputs. The destination registers must accumulate in float32 to avoid precision loss. This is the standard pattern from both the softmax and tilize/untilize references.
- **Alternatives Considered**: None -- this is a hard requirement.
- **Tradeoffs**: Halves DEST register capacity (from 8 to 4 tiles in half-sync mode).

### Decision 10: WaitUpfrontNoPop Policy for Reductions on Tilized Input
- **Choice**: The first SUM reduce (for mean) uses `WaitUpfrontNoPop` policy on cb_tilized, because the same tiles are needed again for the subtraction step (x - mean).
- **Rationale**: This matches the softmax pattern exactly: MAX reduce uses `WaitUpfrontNoPop` so tiles persist for the subsequent subtraction. After subtraction, tiles are explicitly popped.
- **Alternatives Considered**: Reading tiles from DRAM again -- wasteful.
- **Tradeoffs**: Tiles remain in the CB, consuming memory. This is acceptable since we already sized the CB for Wt tiles.

## Work Distribution

### Work Unit Definition
One work unit = one **tile-row** = Wt tiles spanning the full W dimension for one tile-height (32 rows).

Total work units = `nblocks = total_tiles / Wt` where `total_tiles = (product of all dims) / (32 * 32)` and `Wt = W / 32`.

Equivalently: `nblocks = (N_batch * H) / 32` where `N_batch` is the product of all batch dimensions and H is the second-to-last dimension.

### Parallelization Strategy
- **Grid**: Single core (1x1) for prototype
- **Work per core**: All `nblocks` tile-rows
- **Load balancing**: N/A for single core

**Future multi-core extension**: Distribute tile-rows across cores using `split_blocks_for_tilize(grid, nblocks)`. Each core processes its assigned tile-rows independently. Per-core runtime args specify `start_stick_id` and `start_tile_id`.

## Data Flow

### High-Level Flow
For each tile-row (block of 32 sticks spanning full W):

```
1. Reader: Read 32 RM sticks from DRAM -> cb_rm_in
2. Compute: Tilize cb_rm_in -> cb_tilized (Wt tiles)
3. Compute: SUM reduce cb_tilized (with 1/W scaler) -> cb_mean (1 tile, column vector)
4. Compute: sub<COL> cb_tilized - cb_mean -> cb_xmm (Wt tiles)
   (pop cb_tilized and cb_mean after this step)
5. Compute: square cb_xmm -> cb_xmm_sq (Wt tiles)
   (cb_xmm stays for final multiply - WaitUpfrontNoPop on square input)
6. Compute: SUM reduce cb_xmm_sq (with 1/W scaler) -> cb_var (1 tile)
   (pop cb_xmm_sq after reduce)
7. Compute: add_bcast_scalar cb_var + epsilon -> in-place or cb_var
   Then: rsqrt_tile -> cb_invstd (1 tile)
   (pop cb_var after this step)
8. Compute: mul<COL> cb_xmm * cb_invstd -> cb_tilized_out (Wt tiles)
   (pop cb_xmm and cb_invstd after this step)
9. Compute: Untilize cb_tilized_out -> cb_rm_out (Wt pages of RM sticks)
   (pop cb_tilized_out after untilize)
10. Writer: Write 32 RM sticks from cb_rm_out -> DRAM
```

### Kernel Data Movement

| Kernel | Core | NOC | Actual Function |
|--------|------|-----|-----------------|
| Reader | RISCV_0 (BRISC) | NOC0 | Read RM sticks from input DRAM buffer via TensorAccessor. Generate scaler (1/W) and epsilon scalar tiles once at start. |
| Compute | RISCV_2 (TRISC) | N/A | Tilize sticks -> tiles, row-standardize compute pipeline (6 phases), untilize tiles -> sticks |
| Writer | RISCV_1 (NCRISC) | NOC1 | Write RM sticks to output DRAM buffer via TensorAccessor, row-by-row within each block. |

### Circular Buffer Requirements

| CB ID | Name | Purpose | Capacity (pages) | Page Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-----------|-----------|----------|----------|----------|
| c_0 | cb_rm_in | Input RM sticks staging | Wt | tile_size(dtype) | Single | Reader | Compute (tilize) | Block |
| c_1 | cb_scaler | Reduce scaler tile (1/W) | 1 | tile_size(dtype) | Single | Reader (once) | Compute | Program |
| c_2 | cb_eps | Epsilon scalar tile | 1 | tile_size(dtype) | Single | Reader (once) | Compute | Program |
| c_3 | cb_tilized | Tilized input tiles | Wt | tile_size(intermed_fmt) | Single | Compute (tilize) | Compute (reduce, sub) | Block |
| c_4 | cb_tilized_out | Normalized tiles before untilize | Wt | tile_size(intermed_fmt) | Single | Compute (mul) | Compute (untilize) | Block |
| c_16 | cb_rm_out | Output RM sticks staging | Wt | tile_size(dtype) | Single | Compute (untilize) | Writer | Block |
| c_24 | cb_mean | Per-row mean (column vector) | 1 | tile_size(intermed_fmt) | Single | Compute (reduce) | Compute (sub) | Block |
| c_25 | cb_xmm | x - mean intermediate | Wt | tile_size(intermed_fmt) | Single | Compute (sub) | Compute (square, mul) | Block |
| c_26 | cb_xmm_sq | (x-mean)^2 intermediate | Wt | tile_size(intermed_fmt) | Single | Compute (square) | Compute (reduce) | Block |
| c_27 | cb_var | Per-row variance (column vector) | 1 | tile_size(intermed_fmt) | Single | Compute (reduce) | Compute (add+rsqrt) | Block |
| c_28 | cb_invstd | rsqrt(var + eps) | 1 | tile_size(intermed_fmt) | Single | Compute (rsqrt) | Compute (mul) | Block |

**Data Format Notes**:
- `dtype` = input tensor data format (BFLOAT16 or FLOAT32)
- `intermed_fmt` = FLOAT32 if `fp32_dest_acc_en` else same as `dtype`
- `tile_size(BFLOAT16)` = 2048 bytes (32 x 32 x 2)
- `tile_size(FLOAT32)` = 4096 bytes (32 x 32 x 4)
- cb_rm_in (c_0): Uses `dtype` format (matches input tensor)
- cb_scaler (c_1): Uses `dtype` format (scaler matches reduce input format)
- cb_eps (c_2): Uses `intermed_fmt` (epsilon added to intermediate variance)
- cb_tilized (c_3) through cb_invstd (c_28): Use `intermed_fmt`
- cb_tilized_out (c_4): Uses `intermed_fmt` (untilize will convert)
- cb_rm_out (c_16): Uses `dtype` format (matches output tensor)

**L1 Memory Budget Estimate** (for bfloat16, intermed = bfloat16):
```
cb_rm_in:       Wt * 2048
cb_scaler:      1 * 2048
cb_eps:         1 * 2048
cb_tilized:     Wt * 2048
cb_tilized_out: Wt * 2048
cb_rm_out:      Wt * 2048
cb_mean:        1 * 2048
cb_xmm:        Wt * 2048
cb_xmm_sq:     Wt * 2048
cb_var:         1 * 2048
cb_invstd:      1 * 2048
Total:          (5*Wt + 5) * 2048

For W=1024 (Wt=32): (160+5) * 2048 = 337,920 bytes ~ 330 KB (fits in L1)
For W=2048 (Wt=64): (320+5) * 2048 = 665,600 bytes ~ 650 KB (exceeds 512KB L1)
```

**Maximum supported W for bfloat16**: approximately W = 1568 (Wt = 49) before exceeding 512KB.
**Maximum supported W for float32**: approximately W = 768 (Wt = 24) before exceeding 512KB.

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access

**One-time setup** (at program start):
1. Generate reduce scaler tile: `generate_reduce_scaler(cb_scaler, packed_1_over_W)` -- fills cb_scaler with 1/W value in correct tile format. For bfloat16: scaler is `(bf16_val << 16 | bf16_val)`. For float32: scaler is reinterpreted float bits.
2. Generate epsilon scalar tile: `generate_bcast_scalar_bfloat16(cb_eps, packed_epsilon)` for bfloat16 or `generate_bcast_scalar(cb_eps, epsilon_bits)` for float32. This creates a scalar broadcast tile.

**Per-block loop** (for each tile-row):
1. Pre-compute 32 NoC addresses for sticks using TensorAccessor: `base_src_noc_addr[j] = get_noc_addr(stick_id + j, s)`
2. Reserve Wt pages in cb_rm_in
3. Read 32 sticks sequentially via `noc_async_read` -- each stick is `W * datum_size` bytes
4. `noc_async_read_barrier()`
5. Push Wt pages to cb_rm_in

### RISCV_1 ("writer" / NCRISC) Access

**Per-block loop** (for each tile-row):
1. Wait for Wt pages in cb_rm_out
2. Compute L1 base read address from `get_read_ptr(cb_rm_out)`
3. For each of 32 rows within the block:
   a. Compute output stick page ID: `output_page_id = block_index * 32 + j`
   b. Compute L1 read offset: `base_l1_addr + j * W * datum_size`
   c. Write stick to DRAM via `noc_async_write` using TensorAccessor
4. `noc_async_write_barrier()`
5. Pop Wt pages from cb_rm_out

### Compute Access

**Per-block sequence** (for each tile-row):

1. **Tilize phase**: `cb_wait_front(cb_rm_in, Wt)` -> `cb_reserve_back(cb_tilized, Wt)` -> `tilize_block(cb_rm_in, Wt, cb_tilized)` -> `cb_push_back(cb_tilized, Wt)` -> `cb_pop_front(cb_rm_in, Wt)`. Uses `compute_kernel_lib::tilize<c_0, c_3>()`.

2. **Mean reduce**: `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>(cb_tilized, cb_scaler, cb_mean, {1, Wt, 1})`. Tiles stay in cb_tilized. Result in cb_mean (1 tile, column vector of per-row means).

3. **Subtract mean**: `sub<COL, NoWaitNoPop>(cb_tilized, cb_mean, cb_xmm, Wt)`. Then manually `cb_pop_front(cb_tilized, Wt)` and `cb_pop_front(cb_mean, 1)`.

4. **Square (x-mean)**: `square<NoWaitNoPop>(cb_xmm, cb_xmm_sq, Wt)` or equivalently multiply cb_xmm with itself. Tiles stay in cb_xmm for step 8.

5. **Variance reduce**: `reduce<SUM, REDUCE_ROW, BulkWaitBulkPop>(cb_xmm_sq, cb_scaler, cb_var, {1, Wt, 1})`. Pops cb_xmm_sq. Result in cb_var (1 tile, column vector of per-row variances).

6. **Add epsilon + rsqrt**: Wait for cb_var (1 tile) and cb_eps (1 tile, persistent). `add_tiles_bcast_scalar(cb_var, cb_eps, 0, 0, dst0)` -> `rsqrt_tile(dst0)` -> pack to cb_invstd. Pop cb_var.

7. **Final normalize**: `mul<COL, NoWaitNoPop>(cb_xmm, cb_invstd, cb_tilized_out, Wt)`. Then pop cb_xmm (Wt tiles) and cb_invstd (1 tile).

8. **Untilize phase**: `cb_wait_front(cb_tilized_out, Wt)` -> `cb_reserve_back(cb_rm_out, Wt)` -> `untilize_block(cb_tilized_out, Wt, cb_rm_out)` or `pack_untilize_block` -> `cb_push_back(cb_rm_out, Wt)` -> `cb_pop_front(cb_tilized_out, Wt)`. Uses `compute_kernel_lib::untilize<Wt, c_4, c_16>()`.

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size_bytes | uint32_t | Size of one RM stick in bytes: `W * datum_size` |
| 1 | is_float32 | uint32_t | 1 if input is FLOAT32, 0 if BFLOAT16 |
| 2+ | TensorAccessorArgs (src) | uint32_t[] | Bank-interleaved addressing parameters for input tensor |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Number of tiles in the W dimension (tiles per row) |
| 1 | nblocks | uint32_t | Total number of tile-rows (blocks) to process |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size_bytes | uint32_t | Size of one output RM stick in bytes: `W * datum_size` |
| 1 | Wt | uint32_t | Number of tiles per row (used to compute CB wait count) |
| 2+ | TensorAccessorArgs (dst) | uint32_t[] | Bank-interleaved addressing parameters for output tensor |

### Compute Kernel Defines

| Define | Condition | Effect |
|--------|-----------|--------|
| FP32_DEST_ACC_EN | input dtype == FLOAT32 | Enables FP32 accumulation in DST registers |

## Runtime Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_sticks | uint32_t | Total number of sticks to read: `nblocks * 32` |
| 2 | start_stick_id | uint32_t | First stick ID for this core (0 for single-core) |
| 3 | Wt | uint32_t | Tiles per row |
| 4 | scaler | uint32_t | Reduce scaler (1/W) as packed uint32 (bf16<<16\|bf16 for bfloat16, float bits for float32) |
| 5 | epsilon | uint32_t | Epsilon as packed uint32 (bf16<<16\|bf16 for bfloat16, float bits for float32) |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | num_blocks | uint32_t | Number of tile-row blocks to write |
| 2 | start_stick_id | uint32_t | First output stick ID for this core (0 for single-core) |

## Edge Cases

| Condition | Expected Behavior |
|-----------|-------------------|
| Single tile (W=32, H=32) | Wt=1: reduce across 1 tile, standardize correctly. Simplest case. |
| Single tile-row, multi-column (W=1024, H=32) | Wt=32: one block, 32 tiles per reduction. Standard case. |
| Multi tile-row (H=128, W=64) | nblocks=4, Wt=2: 4 blocks of 2 tiles each. |
| Large tensor (1024x1024) | nblocks=32, Wt=32: 32 blocks of 32 tiles. L1 budget: ~330KB for bf16. |
| Batched 3D (2, 32, 64) | N_batch=2, H=32, W=64: nblocks=2, Wt=2. Each tile-row is independent. |
| Batched 4D (2, 4, 32, 64) | N_batch=8, H=32, W=64: nblocks=8, Wt=2. |
| Float32 input | All CBs use 4096-byte pages. FP32_DEST_ACC_EN enabled. DEST capacity halved. |
| Zero variance row (constant row) | var=0, var+eps=eps, rsqrt(eps)=1/sqrt(eps). Output = (x-mean)/sqrt(eps) which is ~0 for constant rows. Numerically stable due to epsilon. |
| Very small epsilon | rsqrt may have precision issues for very small epsilon with bfloat16. Default 1e-5 is safe. |
| W not multiple of 32 | Rejected by input validation. Error: "Last dimension must be a multiple of 32" |
| H not multiple of 32 | Rejected by input validation. Error: "Second-to-last dimension must be a multiple of 32" |
| 1D tensor | Rejected by input validation. Error: "Input tensor must have rank >= 2" |
| Wrong layout | Rejected by input validation. Error: "Input tensor must be in ROW_MAJOR layout" |
| Unsupported dtype (e.g., bfloat8) | Rejected by input validation. Error: "Input tensor must be bfloat16 or float32" |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **generic_op_builder** | API Specification, Input Tensor Requirements, Output Tensor Specification, Circular Buffer Requirements, Work Distribution, Compile-Time Arguments, Runtime Arguments, Component Sources |
| **ttnn-kernel-designer** | Circular Buffer Requirements, Data Flow, Memory Access Patterns, Compute Access, Mathematical Definition, Component Sources, Design Decisions |
| **ttnn-kernel-writer** | Kernel Data Movement, Memory Access Patterns, Compile-Time Arguments, Runtime Arguments, Edge Cases, Design Decisions |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

### Validation Behavior
- Tensor rank < 2 -> error containing "rank >= 2"
- Wrong layout (TILE_LAYOUT) -> error containing "ROW_MAJOR layout"
- Unsupported dtype (bfloat8_b) -> error containing "bfloat16 or float32"
- W not multiple of 32 -> error containing "multiple of 32"
- H not multiple of 32 -> error containing "multiple of 32"
- Tensor not on device -> error containing "on device"

### Shape Behavior
- Output shape == input shape for all valid inputs
- Output dtype == input dtype
- Output layout == ROW_MAJOR
- Output memory == INTERLEAVED

### Functional Behavior
- **PyTorch reference**: `(x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, unbiased=False, keepdim=True) + epsilon)`
- **Accuracy for bfloat16**: PCC > 0.99
- **Accuracy for float32**: PCC > 0.999

### Test Shapes (cartesian product with dtypes)
- 2D: (32, 32), (32, 64), (64, 128), (128, 128), (32, 1024), (128, 1024), (1024, 32), (1024, 1024)
- 3D: (2, 32, 64), (4, 64, 128)
- 4D: (2, 4, 32, 64)

### Test Dtypes
- ttnn.bfloat16
- ttnn.float32

## Open Questions

None. All design decisions have been made with reasonable assumptions for a prototype implementation. Future improvements to consider:
1. Multi-core distribution for performance
2. Large-W variant for tensors exceeding L1 memory budget
3. Sharded memory support
4. Optional gamma/beta parameters (full layer norm)

## References
- Reference analyses:
  - `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/row_standardize/tilize_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/row_standardize/softmax_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/row_standardize/untilize_analysis.md`
- DeepWiki queries:
  - generic_op infrastructure and ProgramDescriptor: Confirmed Python-based approach with KernelDescriptor, CBDescriptor
  - rsqrt_tile and add_tiles_bcast_scalar: Confirmed availability for epsilon + rsqrt pattern
  - generate_reduce_scaler format: Confirmed packed bfloat16 `(bf16 << 16 | bf16)` format
- Documentation consulted:
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` - reduce<SUM, REDUCE_ROW> with policies
  - `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp` - sub<COL>, mul<COL>, square<>
  - `ttnn/cpp/ttnn/kernel_lib/scalar_helpers.hpp` - generate_bcast_scalar_bfloat16, generate_bcast_col_scalar
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` - generate_reduce_scaler
  - `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` - compute_kernel_lib::tilize<>()
  - `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` - compute_kernel_lib::untilize<>()
  - `.claude/references/table-templates.md` - table formatting
