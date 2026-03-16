# moreh_sum_w Implementation Analysis

## Overview

**moreh_sum_w** reduces a tensor along the W (last / innermost) dimension by summing all elements within each row of tiles. Given an input of shape `[..., H, W]`, it produces an output of shape `[..., H, 1]` (padded to tile alignment, so the output W dimension is one tile wide).

The reduction is implemented using the **reduction-via-matmul** pattern: each input tile is multiplied against a pre-constructed scaler tile whose first column contains 1.0 and all other positions are 0. The matmul accumulates partial sums across destination registers, and the final tile per row is packed to the output CB.

**Program factory path**: `ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_program_factory.cpp`

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-row |
| **Unit size** | Wt tiles (one complete tile-row along W) |
| **Total units** | `other_dims_product * Ht` (all tile-rows across batch, channel, and H dims) |
| **Loop structure** | Outer: NC (=1, collapsed batch dims are handled via row count); Middle: Ht; Inner: Wt tiles per row |

One "work unit" is a single tile-row: the Wt tiles that share the same (batch, h_tile) coordinate. Processing one tile-row produces exactly one output tile.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | `[N0, N1, ..., H, W]` (arbitrary rank, minimum 2D) |
| **Dimension convention** | Last two dims are H, W; all preceding dims are batch |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (via TensorAccessor) |
| **Data type** | bfloat16 or float32 (determined by `input.dtype()`) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | `[N0, N1, ..., H, 1]` (W reduced to 1) |
| **Dimension convention** | Same batch dims, H preserved, W collapsed |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (via TensorAccessor) |
| **Data type** | Same as output tensor dtype (may differ from input) |

### Layout Transformations

No explicit tilize/untilize steps in this operation. Input and output are both in TILE_LAYOUT. The W-dimension masking handles partial tiles when the logical W is not a multiple of 32.

---

## Data Flow Pattern

The operation processes each tile-row (Wt tiles) in three phases within the compute kernel:

### Phase 1: Accumulate first (Wt-1) tiles via matmul

For tiles `wt = 0 .. Wt-2`:
1. **Reader** pushes one input tile to `cb_input` (c_0)
2. **Compute** waits for the tile, calls `matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx)` which multiplies the input tile by the scaler tile and **accumulates** into `dst[reduce_dst_idx]`
3. **Compute** pops the consumed input tile

After all (Wt-1) tiles are processed, the accumulated partial sum is packed from the destination register into `cb_accum_dst` (c_24).

### Phase 2: Mask the last tile (conditional)

If `origin_W % 32 != 0` (partial last tile):
1. **Compute** waits for the last input tile in `cb_input`
2. Copies the input tile and the mask tile into destination registers
3. Applies `mask_tile()` to zero out padding columns
4. Packs the masked tile into `cb_masked_input` (c_25)
5. Redirects `cb_input` to `cb_masked_input` for Phase 3

### Phase 3: Final accumulation and output

1. If `Wt > 1`: copies the partial sum from `cb_accum_dst` back into a destination register
2. Calls `matmul_tiles` on the (possibly masked) last tile with the scaler, accumulating into the same destination register
3. Packs the final result into `cb_out` (c_16)
4. **Writer** consumes from `cb_out` and writes to DRAM

### Special case: Wt == 1

When there is only one tile in the W dimension, Phase 1 is skipped entirely. Phase 2 (masking) may still apply. Phase 3 processes the single (possibly masked) tile directly.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_2 | cb_scaler | Scaler tile (column of 1.0s) | 2 tiles | 1 tile | Double | Reader | Compute | **Program** (loaded once, never popped until end) |
| c_3 | cb_mask_w | W-dimension mask tile | 1 tile | 1 tile | Single | Reader | Compute | **Program** (loaded once if needed, popped at end) |
| c_24 | cb_accum_dst | Intermediate accumulator | 1 tile | 1 tile | Single | Compute | Compute | **Row** (written and consumed within one tile-row) |
| c_25 | cb_masked_input | Masked last-tile intermediate | 1 tile | 1 tile | Single | Compute | Compute | **Row** (written and consumed within one tile-row, only if do_mask_w) |
| c_16 | cb_out | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Block (per output tile) |

### Data Format Details

| CB ID | Data Format | Tile Size | Notes |
|-------|-------------|-----------|-------|
| c_0 | Same as input dtype | `tt::tile_size(src0_cb_data_format)` | bfloat16 or float32 |
| c_2 | Float16_b (bfloat16) | `tt::tile_size(Float16_b)` | Always bfloat16 regardless of input type |
| c_3 | Float16_b (bfloat16) | `tt::tile_size(Float16_b)` | Mask is always bfloat16 |
| c_24 | Float32 if `fp32_dest_acc_en`, else Float16_b | `tt::tile_size(intermed_cb_data_format)` | Matches dest accumulation precision |
| c_25 | Float16_b | `tt::tile_size(Float16_b)` | Always bfloat16 regardless of fp32 mode |
| c_16 | Same as output dtype | `tt::tile_size(dst_cb_data_format)` | bfloat16 or float32 |

### Multi-Pass Data Reuse Patterns

**CBs that persist across row iterations (Program lifetime):**

- **cb_scaler (c_2)**: The scaler tile is generated once by the reader at program start. The compute kernel calls `cb_wait_front(cb_scaler, 1)` before the main loop and `cb_pop_front(cb_scaler, 1)` only at the very end after all rows are processed. This means the scaler tile remains resident in the CB throughout the entire kernel execution. Every `matmul_tiles` call references it at tile offset 0 without consuming it.

- **cb_mask_w (c_3)**: Similarly loaded once by the reader (if `do_mask_w`). The compute kernel waits for it before the main loop and pops it only at the end. It is reused across all row iterations.

**CBs that are produced and consumed within each row (Row lifetime):**

- **cb_accum_dst (c_24)**: Written by compute (pack from Dst register) after Phase 1, then read back by compute (copy to Dst register) in Phase 3, then popped. This CB acts as a "spill buffer" to save the partial sum while the mask operation uses the Dst registers. It is produced and consumed entirely within compute -- no reader or writer involvement.

- **cb_masked_input (c_25)**: Written by compute after the mask operation in Phase 2, then consumed by compute in Phase 3 as the replacement for cb_input. Also purely internal to compute.

**Key insight for softmax**: The pattern of loading constant tiles (scaler, mask) once and keeping them resident across all iterations avoids redundant DRAM reads. For softmax, a similar approach could be used for the scaler tile used in exp-sum reduction.

---

## Scalar/Constant CB Setup

### Scaler Tile (cb_scaler, c_2)

The scaler tile implements the reduction-via-matmul pattern. It is created by the reader kernel using `generate_mm_scaler(cb_id_in2, scaler)`:

1. The tile is first zero-filled (reading from hardware zero memory via NoC)
2. The scalar value (1.0 for sum, packed as bfloat16) is placed at specific positions:
   - `ptr[i]` for `i in range(0, 128, 8)` -- first column of face 0 (top-left 16x16 subtile)
   - `ptr[i]` for `i in range(256, 384, 8)` -- first column of face 2 (bottom-left 16x16 subtile)

This creates a tile where only the first column of the left faces contains 1.0, effectively forming a column vector. When an input tile is multiplied by this scaler tile via `matmul_tiles`, the result is the row-wise sum of the input tile packed into the first column of the output.

The scaler value is passed as a compile-time argument (`packed_scaler_value = pack_two_bfloat16_into_uint32({1.0, 1.0})`), appended after the TensorAccessor args in the reader's compile-time args.

### Mask Tile (cb_mask_w, c_3)

The mask tile handles the case where the logical W dimension is not a multiple of 32 (tile width). It is created by `generate_mask_w(cb_id_mask_w, mask_w)`:

1. For each row in the tile, positions `0..mask_w-1` get value 1.0, positions `mask_w..31` get 0.0
2. This is applied across all four subtiles (top-left, top-right, bottom-left, bottom-right) with appropriate splitting at the 16-element boundary

The mask tile is used in compute via `mask_tile(reduce_dst_idx, mask_dst_idx)` which zeroes elements in the data tile where the mask tile has 0.0.

---

## Compute Kernel Structure and Helper Calls

### File
`ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/moreh_sum_w.cpp`

### Includes
- `api/compute/matmul.h` -- provides `mm_init_short`, `matmul_tiles`
- `ttnn/kernel/compute/moreh_common.hpp` -- provides helper wrappers (not directly used by this kernel, but included for `mask_tile`, `copy_tile` etc. via transitive includes)

### Compile-Time Arguments
```cpp
uint32_t Ht = get_compile_time_arg_val(0);     // tile rows per core
uint32_t Wt = get_compile_time_arg_val(1);     // tiles along W dimension
uint32_t NC = get_compile_time_arg_val(2);     // always 1 (batch dims folded into Ht)
constexpr uint32_t origin_W = get_compile_time_arg_val(3);  // logical W before padding
```

### Initialization

```cpp
binary_op_init_common(cb_input, cb_scaler, cb_out);
// Signature: void binary_op_init_common(uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out)
// Initializes unpacker, math core, and packer for binary operations.
// Sets up hardware state based on data formats of the three CBs.
```

### Constant tile wait (before main loop)

```cpp
cb_wait_front(cb_scaler, 1);   // Wait for scaler tile from reader (persists entire program)
if (do_mask_w) {
    cb_wait_front(cb_mask_w, 1);  // Wait for mask tile from reader (persists entire program)
}
```

### Phase 1: Partial sum accumulation (Wt-1 tiles)

```cpp
tile_regs_acquire();  // Acquire Dst registers (zeroes them for matmul accumulation)
for (uint32_t wt = 0; wt < Wt - 1; ++wt) {
    cb_wait_front(cb_input, onetile);
    reconfig_data_format(cb_input, cb_scaler);     // Only if FP32_DEST_ACC_EN
    mm_init_short(cb_input, cb_scaler, false);     // Reinit matmul for these CB formats
    // Signature: void mm_init_short(uint32_t in0_cb, uint32_t in1_cb, bool transpose=false)
    matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx);
    // Signature: void matmul_tiles(uint32_t cb0, uint32_t cb1, uint32_t tile0, uint32_t tile1, uint32_t dst_idx)
    // ACCUMULATES into dst[reduce_dst_idx] -- does NOT overwrite
    cb_pop_front(cb_input, onetile);
}
tile_regs_commit();                                // Hand Dst to packer
cb_reserve_back(cb_accum_dst, onetile);
tile_regs_wait();                                  // Wait for packer readiness
pack_reconfig_data_format(cb_accum_dst);           // Only if FP32_DEST_ACC_EN
pack_tile(reduce_dst_idx, cb_accum_dst);
// Signature: void pack_tile(uint32_t dst_idx, uint32_t cb_out)
tile_regs_release();
cb_push_back(cb_accum_dst, onetile);
```

**Key pattern**: `tile_regs_acquire()` zeroes destination registers for matmul. Multiple `matmul_tiles` calls accumulate partial products in the same Dst slot. The partial sum is then packed to an intermediate CB.

### Phase 2: Mask last tile (conditional on `do_mask_w`)

```cpp
tile_regs_acquire();
cb_wait_front(cb_input, onetile);
reconfig_data_format_srca(cb_input);               // Only if FP32_DEST_ACC_EN
copy_tile_to_dst_init_short(cb_input);
// Signature: void copy_tile_to_dst_init_short(uint32_t in_cb_id)
copy_tile(cb_input, 0, reduce_dst_idx);
// Signature: void copy_tile(uint32_t cb, uint32_t tile_offset, uint32_t dst_idx)
copy_tile(cb_mask_w, 0, mask_dst_idx);             // mask_dst_idx = reduce_dst_idx + 1
mask_tile_init();
// Signature: void mask_tile_init()
mask_tile(reduce_dst_idx, mask_dst_idx);
// Signature: void mask_tile(uint32_t dst_data_idx, uint32_t dst_mask_idx)
// Zeroes elements in dst[dst_data_idx] where dst[dst_mask_idx] has 0
tile_regs_commit();
cb_reserve_back(cb_masked_input, onetile);
tile_regs_wait();
pack_reconfig_data_format(cb_masked_input);        // Only if FP32_DEST_ACC_EN
pack_tile(reduce_dst_idx, cb_masked_input);
tile_regs_release();
cb_push_back(cb_masked_input, onetile);

cb_pop_front(cb_input, onetile);                   // Done with original input
cb_input = cb_masked_input;                        // Redirect for Phase 3
```

**Key pattern**: Two tiles are loaded into adjacent Dst register slots. The mask operation uses both slots. The `copy_tile` for `cb_mask_w` uses tile offset 0 because the mask never advances (it persists across rows).

### Phase 3: Final matmul and output

```cpp
tile_regs_acquire();
cb_wait_front(cb_input, onetile);                  // Either cb_masked_input or cb_input
if (!is_w_single_tile) {
    reconfig_data_format_srca(cb_accum_dst);       // Only if FP32_DEST_ACC_EN
    cb_wait_front(cb_accum_dst, onetile);
    copy_tile_to_dst_init_short(cb_accum_dst);
    copy_tile(cb_accum_dst, 0, reduce_dst_idx);    // Load partial sum into Dst
}
reconfig_data_format(cb_input, cb_scaler);         // Only if FP32_DEST_ACC_EN
mm_init_short(cb_input, cb_scaler, false);
matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx);
// If Wt>1: accumulates last tile's contribution on top of partial sum
// If Wt==1: reduces single tile (which may be masked)
tile_regs_commit();
cb_reserve_back(cb_out, onetile);
tile_regs_wait();
pack_reconfig_data_format(cb_out);                 // Only if FP32_DEST_ACC_EN
pack_tile(reduce_dst_idx, cb_out);
tile_regs_release();
cb_push_back(cb_out, onetile);

cb_pop_front(cb_input, onetile);
if (!is_w_single_tile) {
    cb_pop_front(cb_accum_dst, onetile);
}
```

**Key pattern**: `copy_tile` loads a previously packed partial sum back into Dst. Then `matmul_tiles` adds the last tile's row-reduced values on top. This two-pass approach (pack to CB, then reload) is necessary because the mask operation needs Dst registers (uses 2 slots), and the matmul accumulation needs a clean Dst.

### Cleanup (after all rows)

```cpp
if (do_mask_w) {
    cb_pop_front(cb_mask_w, onetile);              // Release persistent mask tile
}
cb_pop_front(cb_scaler, onetile);                  // Release persistent scaler tile
```

---

## Reduce Helper Parameters and Binary Op Broadcast Pattern

### The Reduction-via-Matmul Pattern

This operation does NOT use the `reduce_tile` API. Instead, it uses `matmul_tiles` with a scaler tile as a "column vector of ones" to achieve row-wise summation. This approach was chosen for better numerical precision.

The define `REDUCE_ROW_SUM_VIA_MM=1` is set by `reduce_op_utils::get_defines(ReduceOpMath::SUM, ReduceOpDim::W)`, along with:
- `REDUCE_OP = "PoolType::SUM"`
- `REDUCE_DIM = "ReduceDim::REDUCE_ROW"`

However, this specific compute kernel (`moreh_sum_w.cpp`) does NOT reference these defines directly in its logic. It uses `matmul_tiles` unconditionally. The defines are passed to the kernel but appear to be used by the included headers for configuration rather than for conditional branching in this specific kernel.

### How matmul_tiles performs reduction

For a 32x32 input tile `A` and scaler tile `S` (column vector of 1s):
- `C = A * S` yields a 32x32 result where column 0 contains the row sums and all other columns are 0
- When `matmul_tiles` is called multiple times on the same Dst slot, it accumulates: `Dst += A_wt * S`, giving `Dst = sum(A_0*S + A_1*S + ... + A_{Wt-1}*S)`
- The final result in column 0 of the Dst tile contains the complete row sums across all W tiles

---

## FP32 Destination Accumulation

When `fp32_dest_acc_en` is true:

1. **Dst registers**: Hold 8 tiles of 32-bit data (instead of 16 tiles of 16-bit)
2. **UnpackToDestMode**: `cb_accum_dst` (c_24) is configured with `UnpackToDestMode::UnpackToDestFp32` to ensure proper format conversion when copying from this CB to Dst
3. **reconfig_data_format calls**: Required before every matmul/copy operation to reconfigure the unpacker for the source CB's format (which may be bfloat16 while Dst is fp32)
4. **pack_reconfig_data_format calls**: Required before every pack operation to configure the packer for the target CB's format (converting from fp32 Dst to the CB's format)

The intermediate CB c_24 is allocated with `Float32` format when fp32 accumulation is enabled, preserving full precision of the partial sum. CB c_25 (masked input) remains `Float16_b` regardless.

---

## Index Calculations

### Reader: Linear tile indexing via TensorAccessor

The reader uses `noc_async_read_tile(i, s, l1_write_addr)` with a linear tile index `i` ranging from `start_id` to `start_id + num_tiles`. The TensorAccessor translates linear tile indices to physical DRAM addresses accounting for bank interleaving.

Tiles are read in linear order (W-contiguous within each row): for a given (batch, h_tile) position, tiles `[wt=0, wt=1, ..., wt=Wt-1]` are read consecutively.

### Writer: Output tile indexing

The writer also uses linear tile indices. For every Wt input tiles, there is 1 output tile. The output start index is `num_tiles_read / Wt` (the input start divided by the W-tile count).

### Compute: No explicit indexing

The compute kernel does not perform address calculations. It consumes tiles from CBs in FIFO order. The NC/Ht/Wt loop structure ensures tiles arrive in the expected order.

---

## Memory Access Patterns

### Read Pattern
- **Sequential tile reads**: Input tiles are read in linear tile order, one at a time
- **DRAM reads via NoC**: Each tile read is a `noc_async_read_tile` with a barrier (no pipelining within the reader loop)
- **Constant tiles read once**: Scaler and mask tiles are generated in L1 (using `generate_mm_scaler` and `generate_mask_w`) before the main read loop, avoiding repeated DRAM access

### Write Pattern
- **Sequential tile writes**: Output tiles are written in linear tile order, one at a time
- **DRAM writes via NoC**: Each tile write is `noc_async_write_tile` with a barrier
- **Reduced output volume**: For every Wt input tiles, only 1 output tile is produced

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major traversal) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | Up to `grid_x * grid_y`, bounded by `num_rows` |
| **Work per core** | `num_rows_per_core_group_1` or `num_rows_per_core_group_2` tile-rows |
| **Load balancing** | Two groups: group_1 gets `ceil(num_rows / num_cores)`, group_2 gets `floor(num_rows / num_cores)` |

### Work Splitting Details

- `num_rows = other_dims_product * Ht` (total tile-rows across all batch dimensions and H)
- `split_work_to_cores` divides `num_rows` across available cores
- Core indexing is column-major: `core = {i / num_cores_y, i % num_cores_y}`
- Each core processes `num_rows_per_core * Wt` input tiles and produces `num_rows_per_core` output tiles
- Two compute kernels are created (one per core group) with different `Ht` compile-time args
- Reader and writer kernels are shared across all cores, with per-core runtime args

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for input buffer (bank mapping, page size, etc.) |
| N+1 | packed_scaler_value | uint32_t | Two bfloat16 1.0 values packed into one uint32 (`pack_two_bfloat16_into_uint32`) |

Additionally, the reader has conditional defines:
- `DO_MASK_W=1` if `origin_W % TILE_WIDTH != 0`

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile-rows this core processes (differs between core_group_1 and core_group_2) |
| 1 | Wt | uint32_t | Number of tiles along W dimension |
| 2 | NC | uint32_t | Always 1 (batch dims folded into Ht) |
| 3 | origin_W | uint32_t | Logical W dimension before tile padding (constexpr, used to compute do_mask_w) |

Defines passed to compute:
- `REDUCE_OP = "PoolType::SUM"`
- `REDUCE_DIM = "ReduceDim::REDUCE_ROW"`
- `REDUCE_ROW_SUM_VIA_MM = "1"`
- `FP32_DEST_ACC_EN = "1"` (conditional on `fp32_dest_acc_en`)

ComputeConfig settings:
- `.math_fidelity = math_fidelity`
- `.fp32_dest_acc_en = fp32_dest_acc_en`
- `.unpack_to_dest_mode[c_24] = UnpackToDestFp32` (if fp32_dest_acc_en)
- `.math_approx_mode = math_approx_mode`

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index to read from (c_16) |
| 1..M | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for output buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address |
| 1 | num_tensor_tiles_per_core | uint32_t | Total input tiles for this core (`num_rows_per_core * Wt`) |
| 2 | start_id | uint32_t | Starting tile index for this core |
| 3 | mask_w | uint32_t | Number of valid columns in last W tile (1-32) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | num_tiles | uint32_t | Output tiles for this core (`num_tensor_tiles_per_core / Wt`) |
| 2 | start_id | uint32_t | Starting output tile index (`input_start_id / Wt`) |

---

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_moreh_sum_w | RISCV_0 | NOC0 | DRAM input | c_0, c_2, c_3 | Read input tiles; generate scaler and mask tiles |
| moreh_sum_w (compute) | RISCV_2 | N/A | c_0, c_2, c_3 | c_16 (via c_24, c_25) | Matmul reduction, masking, accumulation |
| writer_moreh_sum_w | RISCV_1 | NOC1 | c_16 | DRAM output | Write output tiles |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/reader_moreh_sum_w.cpp`
- **Key Logic**: Generates scaler tile via `generate_mm_scaler`, optionally generates mask tile via `generate_mask_w`, then reads input tiles one at a time with TensorAccessor.

### Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/moreh_sum_w.cpp`
- **Key Logic**: Three-phase reduction pattern (accumulate, mask, finalize) detailed in the Data Flow and Compute Kernel Structure sections above.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/writer_moreh_sum_w.cpp`
- **Key Logic**: Simple tile-by-tile writer using TensorAccessor.

---

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Buffering Type | Overlap Potential |
|----|----------|------------|----------------|-------------------|
| c_0 (input) | 2 tiles | 1 tile | Double-buffered | Reader can fill next tile while compute processes current |
| c_2 (scaler) | 2 tiles | 1 tile | Double-buffered (capacity) but single-use | No overlap needed; loaded once |
| c_3 (mask) | 1 tile | 1 tile | Single-buffered | No overlap needed; loaded once |
| c_24 (accum) | 1 tile | 1 tile | Single-buffered | Compute-internal; no overlap with reader/writer |
| c_25 (masked) | 1 tile | 1 tile | Single-buffered | Compute-internal; no overlap with reader/writer |
| c_16 (output) | 2 tiles | 1 tile | Double-buffered | Writer can drain previous tile while compute produces next |

---

## Implementation Notes

### Design Choices Relevant to Softmax

1. **Reduction-via-matmul pattern**: The use of `matmul_tiles` with a scaler column vector avoids precision issues with `reduce_tile`. For softmax, this same pattern can be used for `sum(exp(x))` along W.

2. **Constant CB persistence**: Scaler and mask tiles are loaded once and persist across all row iterations. The compute kernel waits for them once before the main loop and pops them once at the end. This is critical for efficiency and directly applicable to softmax.

3. **Intermediate CB spill pattern**: When the compute kernel needs to perform multiple distinct operations (accumulate, then mask, then combine), it uses intermediate CBs (c_24, c_25) as "spill buffers" to save partial results from Dst registers. This is necessary because different operations need the Dst registers and cannot all fit simultaneously. For softmax, similar spill buffers would be needed between the max-finding, exp computation, sum-reduction, and division phases.

4. **FP32 accumulation support**: The operation cleanly supports both fp32 and fp16 accumulation modes through conditional `reconfig_data_format` calls and appropriate intermediate CB formats. For softmax, fp32 accumulation would be important for numerical stability.

5. **Mask handling as a separate phase**: Rather than applying the mask inline during accumulation, the operation handles it as a distinct phase that produces a masked tile in a separate CB. This simplifies the main accumulation loop. For softmax, a similar approach would work for masking the last partial tile.

6. **NC=1 with folded batch dims**: The batch dimensions are folded into the tile-row count (`num_rows = other_dims_product * Ht`), with NC always set to 1. This simplifies the compute kernel loop structure to just NC * Ht iterations.

### Edge Cases

- **Single tile W (Wt==1)**: Phase 1 is entirely skipped. The single tile (possibly masked) goes directly through Phase 3.
- **No masking needed**: Phase 2 is skipped, and cb_input remains pointing to c_0.
- **Empty core_group_2**: If work divides evenly, only core_group_1 is used and the second compute kernel creation is skipped.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the reduction-via-matmul pattern work in TT-Metal compute kernels? Specifically, how is matmul_tiles used with a scaler tile to perform row reduction?"
   **Reason**: Needed to understand WHY matmul is used instead of reduce_tile for W-dimension reduction
   **Key Findings**: matmul_tiles with a scaler tile (column of 1.0s) performs row reduction with better precision than reduce_tile. The scaler tile has 1.0 in the first column of left faces and 0 elsewhere.

2. **Query**: "What are the exact signatures and semantics of binary_op_init_common, mm_init_short, matmul_tiles, mask_tile_init, mask_tile, copy_tile_to_dst_init_short, copy_tile, pack_tile, tile_regs_acquire/commit/wait/release?"
   **Reason**: Needed exact API semantics to document the compute kernel structure accurately
   **Key Findings**: matmul_tiles ACCUMULATES into Dst (does not overwrite); tile_regs_acquire zeroes Dst registers for matmul; the acquire/commit/wait/release lifecycle manages handoff between math core and packer; Dst has 16 tiles (fp16) or 8 tiles (fp32).

3. **Query**: "How does the scaler tile work for row reduction via matmul? What is the specific layout of a scaler tile with value 1.0?"
   **Reason**: Needed to understand the exact tile layout to document the scaler creation and its effect
   **Key Findings**: generate_mm_scaler zeros the tile, then fills positions at indices 0-127 (step 8) and 256-383 (step 8) with the scalar. This creates a column vector in the first column of the left faces.

4. **Query**: "What do reconfig_data_format, reconfig_data_format_srca, and pack_reconfig_data_format do? Why are they needed with FP32_DEST_ACC_EN?"
   **Reason**: Needed to understand the conditional reconfig calls that appear throughout the compute kernel under FP32_DEST_ACC_EN guards
   **Key Findings**: These functions reconfigure the unpacker/packer data format at runtime to bridge mismatches between CB formats (e.g., bfloat16) and Dst register format (fp32). Required because the hardware needs explicit format conversion setup.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp`
   **Reason**: To understand what defines `get_defines(ReduceOpMath::SUM, ReduceOpDim::W)` produces
   **Key Information**: Sets `REDUCE_OP=PoolType::SUM`, `REDUCE_DIM=ReduceDim::REDUCE_ROW`, `REDUCE_ROW_SUM_VIA_MM=1`

2. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp`
   **Reason**: To understand the exact scaler tile generation logic
   **Key Information**: Zeros the tile first, then places packed scalar at stride-8 positions in ranges [0,128) and [256,384)

3. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: To catalog the available compute helper functions and their signatures
   **Key Information**: Provides wrappers like `pack_tile_with_dt`, `copy_tile_init_with_dt`, `exp_tile_to_cb`, `recip_tile_to_cb`, `mul_tiles_bcast_rows_to_cb` etc. -- many directly useful for softmax

4. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: To understand the mask generation functions
   **Key Information**: `generate_mask_w` creates a tile with 1.0 for valid columns and 0.0 for padding columns

5. **Source**: `ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.cpp`
   **Reason**: To understand `extract_spatial_dims` and `split_work_to_cores_wt_core_range`
   **Key Information**: `extract_spatial_dims` returns (W, H, product_of_other_dims); `split_work_to_cores_wt_core_range` divides work into two groups for load balancing
