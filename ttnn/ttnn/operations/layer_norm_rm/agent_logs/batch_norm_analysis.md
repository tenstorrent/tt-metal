# Batch Norm Implementation Analysis

## Overview

**Operation**: Batch Normalization
**Program Factory**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`
**Formula**: `output = (input - batch_mean) / sqrt(batch_var + eps) * weight + bias`

Batch norm normalizes each input tile by subtracting a pre-computed channel mean and multiplying by the reciprocal square root of (variance + epsilon). Optional affine transform (gamma/beta) via weight and bias tensors follows. The mean and variance are provided as separate input tensors (not computed by this kernel).

**Two compute kernel variants** exist, selected at program creation time based on `fp32_dest_acc_en`:
- **FPU path** (`batch_norm_kernel.cpp`): Uses FPU-based binary ops (`add_tiles`, `sub_tiles`, `mul_tiles`) and the `binary_dest_reuse_tiles` optimization.
- **SFPU path** (`batch_norm_sfpu_kernel.cpp`): Uses SFPU-based binary ops (`add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`) operating directly on DST registers with manual `copy_tile` for data movement.

Both variants use TILE_LAYOUT tensors and process one tile at a time through the same algorithmic pipeline.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `output.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Two-level: outer loop iterates over "channel groups" (complete_iterations), inner loop processes H*W tiles per channel with broadcast of per-channel statistics |

---

## Tensor Format and Layout

### Input Tensors

| Property | Input (x) | batch_mean | batch_var | weight (optional) | bias (optional) |
|----------|-----------|------------|-----------|-------------------|-----------------|
| **Logical shape** | [N, C, H, W] | [1, C, 1, 1] (or matching) | [1, C, 1, 1] | [1, C, 1, 1] | [1, C, 1, 1] |
| **Dimension convention** | NCHW | NCHW | NCHW | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 |

### Output Tensor

| Property | Output |
|----------|--------|
| **Logical shape** | [N, C, H, W] (same as input) |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 |

### Key Shape Relationships

The per-channel broadcast tensors (mean, var, weight, bias) have shape `[1, C, 1, 1]` in tile space. The writer reads one tile from each per-channel tensor per channel, applies `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast the single scalar value across all 32x32 positions in the tile, then the compute kernel reuses that filled tile across all `Ht * Wt` spatial tiles in that channel.

---

## Data Flow Pattern

### High-Level: Per-Channel Broadcast Architecture

The operation processes tiles in N, C, H*W order. For each (N, C) group, per-channel statistics are loaded once and broadcast across all `Ht * Wt` spatial tiles. This is the "multi-pass data reuse" pattern.

### Step-by-Step Flow

```
For each (N, C) combination:
  WRITER: Read batch_mean[c] tile -> FILL_TILE_WITH_FIRST_ELEMENT -> push to CB_1
  WRITER: Read batch_var[c] tile -> FILL_TILE_WITH_FIRST_ELEMENT -> push to CB_3
  WRITER: Read weight[c] tile (optional) -> FILL_TILE_WITH_FIRST_ELEMENT -> push to CB_5
  WRITER: Read bias[c] tile (optional) -> FILL_TILE_WITH_FIRST_ELEMENT -> push to CB_6

  COMPUTE: Wait for batch_var in CB_3 and eps in CB_4
  COMPUTE: den = rsqrt(batch_var + eps) -> push to CB_7
  COMPUTE: Wait for batch_mean in CB_1 and den in CB_7

  For each spatial tile t in [0, Ht*Wt):
    READER: Read input[n,c,t] tile -> push to CB_0
    COMPUTE: result = (input - batch_mean) * den
    COMPUTE: if weight: result = result * weight
    COMPUTE: if bias: result = result + bias
    COMPUTE: push result to CB_2
    WRITER: Wait for result in CB_2 -> write to DRAM

  COMPUTE: Pop batch_mean (CB_1), den (CB_7), weight (CB_5), bias (CB_6)
```

### Key Insight: Reader/Writer Role Naming

Despite naming, the **writer kernel reads** the per-channel tensors (batch_mean, batch_var, weight, bias) and writes the output. The **reader kernel reads** only the input tensor and fills the epsilon CB. This is a common pattern in TT-Metal where kernel names reflect core assignment (RISC-V thread 0 vs 1), not necessarily their data direction.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | input_tensor_cb | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Per-tile (produced/consumed each iteration) |
| c_1 | batch_mean_tensor_cb | Broadcast mean tile | 2 tiles | 1 tile | Double | Writer | Compute | Per-channel group (persists across Ht*Wt tiles) |
| c_2 | output_tensor_cb | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Per-tile (produced/consumed each iteration) |
| c_3 | batch_var_tensor_cb | Broadcast variance tile | 2 tiles | 1 tile | Double | Writer | Compute | Per-channel group (consumed once to produce den) |
| c_4 | eps_cb | Epsilon scalar tile | 2 tiles | 1 tile | Double | Reader | Compute | Program (loaded once, persists entire kernel) |
| c_5 | weight_tensor_cb | Broadcast weight tile | 2 tiles | 1 tile | Double | Writer | Compute | Per-channel group (persists across Ht*Wt tiles) |
| c_6 | bias_tensor_cb | Broadcast bias tile | 2 tiles | 1 tile | Double | Writer | Compute | Per-channel group (persists across Ht*Wt tiles) |
| c_7 | den_cb | Intermediate: 1/sqrt(var+eps) | 2 tiles | 1 tile | Double | Compute | Compute (self) | Per-channel group (produced once, consumed across Ht*Wt tiles) |
| c_8 | temp_1_cb | Intermediate for affine steps | 2 tiles | 1 tile | Double | Compute | Compute (self) | Per-tile (transient between affine stages) |

### CB Persistence and Multi-Pass Data Reuse

This is the central architectural pattern of the batch norm compute kernel:

1. **Program lifetime (CB_4 / eps_cb)**: The epsilon constant is filled once by the reader kernel at program start. The compute kernel does `cb_eps_obj.wait_front(onetile)` at the top and `cb_eps_obj.pop_front(onetile)` only at the very end of `kernel_main`. This means the eps tile stays resident in L1 for the entire kernel execution.

2. **Per-channel-group lifetime (CB_1, CB_3, CB_5, CB_6, CB_7)**: These CBs hold per-channel data that is reused across all `Ht * Wt` spatial tiles within a channel. The `batchnorm_bcast_tiles` function:
   - Calls `wait_front` on batch_mean (CB_1), den (CB_7), weight (CB_5), bias (CB_6) **before** the inner loop
   - Iterates the inner `j` loop over `freq` (= Ht*Wt) tiles, using these values repeatedly
   - Calls `pop_front` on all of them **after** the inner loop completes

3. **Per-tile lifetime (CB_0, CB_2, CB_8)**: Input, output, and temp tiles are produced and consumed once per tile in the inner loop.

**Why this matters for layer_norm_rm**: Layer norm operates over a row (the last dimension) rather than a channel. The broadcast pattern will be different -- instead of broadcasting per-channel stats across spatial tiles, layer norm broadcasts per-row stats across the width dimension. The CB persistence pattern can be adapted: the row-wise mean/variance would persist across the Wt tiles of a single row.

### Scalar/Constant CB Setup (CB_4 / eps_cb)

The epsilon value is packed on the host side:
```cpp
const auto packed_scalar_eps = input_tensor.dtype() == DataType::FLOAT32
    ? std::bit_cast<uint32_t>(scalar)
    : pack_two_bfloat16_into_uint32({scalar, scalar});
```

In the reader kernel, this packed value is used to fill an entire tile with the epsilon constant using `FILL_WITH_VALUE_FLOAT` (for fp32) or `FILL_WITH_VALUE` (for bfloat16). These are macros that expand to `fill_with_val<1024, float>` or `fill_with_val_bfloat16` respectively, which iterate over the 1024 elements (or 512 uint32 words for bfloat16) and write the scalar to every position.

---

## Compute Kernel Structure (PRIMARY FOCUS)

### Initialization

**FPU path** (`batch_norm_kernel.cpp`):
```cpp
binary_op_init_common(cb_other, cb_bcast, cb_output_0);
```
Signature: `binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb)` -- initializes hardware unpack/math/pack pipeline for binary operations with two input CBs and one output CB.

**SFPU path** (`batch_norm_sfpu_kernel.cpp`):
```cpp
unary_op_init_common(cb_other, cb_output_0);
```
Signature: `unary_op_init_common(uint32_t icb_in, uint32_t ocb)` -- initializes for unary operations (single input + output). The SFPU path uses copy_tile to manually load operands into DST registers rather than unpacking from CBs through the FPU pipeline.

### Outer Loop Structure

Both variants share identical loop logic:
```cpp
uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

cb_eps_obj.wait_front(onetile);  // eps persists for entire kernel

for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
    batchnorm_bcast_tiles(..., tile_freq, tile_start, ...);
}
if (remaining_iterations > 0) {
    batchnorm_bcast_tiles(..., remaining_iterations, tile_start, ...);
}

cb_eps_obj.pop_front(onetile);
```

- `tile_freq` = `Ht * Wt` (number of spatial tiles per channel)
- `tile_start` = offset within the first channel group (for cores that start mid-channel)
- `complete_iterations` = number of full channel groups this core processes
- `remaining_iterations` = partial channel group at the end

The `tile_start` parameter handles the case where a core's work range starts partway through a channel's spatial tiles. After the first iteration, `tile_start` resets to 0.

### batchnorm_bcast_tiles -- Detailed Compute Phases

#### CB Aliasing for Conditional Output Routing

```cpp
auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;
```

This aliasing controls where intermediate results go:
- **No weight, no bias**: `cb_affine_or_out = cb_output_0`, `cb_scaled_output = cb_output_0` -- normalized result goes directly to output
- **Weight only**: `cb_affine_or_out = cb_tmp_1`, `cb_scaled_output = cb_output_0` -- normalized goes to temp, weight*temp goes to output
- **Bias only**: `cb_affine_or_out = cb_tmp_1`, `cb_scaled_output = cb_tmp_1` -- normalized goes to temp, temp+bias goes to output
- **Weight and bias**: `cb_affine_or_out = cb_tmp_1`, `cb_scaled_output = cb_tmp_1` -- normalized goes to temp, weight*temp goes to temp, temp+bias goes to output

#### Phase 1: Compute Denominator (rsqrt of variance + epsilon)

**FPU path** (exact calls):
```cpp
cb_den_obj.reserve_back(onetile);
cb_batch_var_obj.wait_front(onetile);

tile_regs_acquire();
add_tiles_init_with_dt(cb_batch_var, cb_eps);    // void add_tiles_init_with_dt(uint32_t icb0, uint32_t icb1)
add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);      // void add_tiles(uint32_t in0_cb, uint32_t in1_cb, uint32_t in0_tile_idx, uint32_t in1_tile_idx, uint32_t dst_tile_idx)
rsqrt_tile_init();                                  // void rsqrt_tile_init()
rsqrt_tile(dst0);                                   // void rsqrt_tile(uint32_t dst_tile_idx)
tile_regs_commit();

tile_regs_wait();
pack_tile_with_dt(dst0, cb_den);                   // void pack_tile_with_dt(uint32_t ifrom_dst, uint32_t icb)
tile_regs_release();

cb_batch_var_obj.pop_front(onetile);
cb_den_obj.push_back(onetile);
```

The FPU path is efficient: `add_tiles` unpacks from both CBs directly into SRCA/SRCB, performs the add, stores in DST[0]. Then `rsqrt_tile` operates in-place on DST[0]. A single acquire/commit/wait/release cycle covers both operations.

**SFPU path** (exact calls):
```cpp
cb_den_obj.reserve_back(onetile);
cb_batch_var_obj.wait_front(onetile);

tile_regs_acquire();
tile_regs_wait();
copy_tile_to_dst_init_short_with_dt(cb_eps, cb_batch_var);   // void copy_tile_to_dst_init_short_with_dt(uint32_t old_cbid, uint32_t new_cbid, uint32_t transpose = 0)
for (uint32_t i = 0; i < onetile; ++i) {
    copy_tile(cb_batch_var, i, i * 2);                         // void copy_tile(uint32_t icb, uint32_t itile, uint32_t idst)
}
add_binary_tile_init();                                         // void add_binary_tile_init()
copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_eps);
for (uint32_t i = 0; i < onetile; ++i) {
    copy_tile(cb_eps, i, i * 2 + 1);                           // copy second operand to DST[1]
    add_binary_tile(i * 2, i * 2 + 1, i * 2);                  // void add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) -- DST[0] = DST[0] + DST[1]
}
rsqrt_tile_init();
for (uint32_t i = 0; i < onetile; ++i) {
    rsqrt_tile(i * 2);                                          // rsqrt in-place on DST[0]
    pack_tile(i * 2, cb_den);                                   // void pack_tile(uint32_t ifrom_dst, uint32_t icb)
}
tile_regs_commit();
tile_regs_release();
cb_den_obj.push_back(onetile);
cb_batch_var_obj.pop_front(onetile);
```

The SFPU path manually copies tiles from CBs into DST registers using `copy_tile`, then uses SFPU binary ops that operate entirely within DST register space. The `copy_tile_to_dst_init_short_with_dt` reconfigures the unpacker data format when switching between different source CBs.

#### Phase 2: Normalize Input Tiles (inner loop, per spatial tile)

Before the inner loop, the compute kernel waits for the broadcast tiles:
```cpp
cb_bcast_obj.wait_front(onetile);   // batch_mean -- stays resident
cb_den_obj.wait_front(onetile);     // 1/sqrt(var+eps) -- stays resident
if (weight_has_value) cb_weight_obj.wait_front(onetile);
if (bias_has_value) cb_bias_obj.wait_front(onetile);
```

**FPU path inner loop** (per tile):
```cpp
// Step 2a: input - batch_mean
cb_other_obj.wait_front(onetile);
cb_affine_or_out_obj.reserve_back(onetile);

tile_regs_acquire();
sub_tiles_init(cb_other, cb_bcast);                 // void sub_tiles_init(uint32_t icb0, uint32_t icb1)
sub_tiles(cb_other, cb_bcast, 0, 0, 0);             // DST[0] = input - mean

// Step 2b: multiply by den (uses dest reuse optimization)
binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den);
binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0);
tile_regs_commit();

tile_regs_wait();
pack_tile_with_dt(0, cb_affine_or_out);
tile_regs_release();

cb_affine_or_out_obj.push_back(onetile);
cb_other_obj.pop_front(onetile);
```

**Key optimization -- `binary_dest_reuse_tiles`**: After `sub_tiles` computes `(input - mean)` into DST[0], the `binary_dest_reuse_tiles` call with `DEST_TO_SRCA` moves DST[0] to SRCA, unpacks `cb_den` into SRCB, and performs the multiply. This avoids an intermediate pack/unpack cycle -- the subtraction result stays in registers and is immediately used as an operand for the multiplication. This is a crucial pattern for chaining operations efficiently.

**SFPU path inner loop** (per tile):
```cpp
// Step 2a: input - batch_mean
cb_other_obj.wait_front(onetile);
tile_regs_acquire();
tile_regs_wait();
copy_tile_to_dst_init_short_with_dt(cb_bcast, cb_other);
copy_tile(cb_other, 0, 0);       // DST[0] = input
sub_binary_tile_init();
copy_tile_to_dst_init_short_with_dt(cb_other, cb_bcast);
copy_tile(cb_bcast, 0, 1);       // DST[1] = mean
sub_binary_tile(0, 1, 0);        // DST[0] = DST[0] - DST[1]
cb_other_obj.pop_front(onetile);

// Step 2b: multiply by den
cb_affine_or_out_obj.reserve_back(onetile);
mul_binary_tile_init();
copy_tile_to_dst_init_short_with_dt(cb_bcast, cb_den);
copy_tile(cb_den, 0, 1);         // DST[1] = den
mul_binary_tile(0, 1, 0);        // DST[0] = DST[0] * DST[1]
pack_tile(0, cb_affine_or_out);
tile_regs_commit();
tile_regs_release();
cb_affine_or_out_obj.push_back(onetile);
```

The SFPU path has no `binary_dest_reuse_tiles` equivalent -- instead it manually copies the den tile into a different DST slot and uses `mul_binary_tile` to combine them. The sub result stays in DST[0] across operations naturally since SFPU ops work entirely in DST register space.

#### Phase 3: Optional Weight Multiply

**FPU path**:
```cpp
if (weight_has_value) {
    cb_scaled_output_obj.reserve_back(onetile);
    cb_affine_or_out_obj.wait_front(1);

    tile_regs_acquire();
    mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
    mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_scaled_output);
    tile_regs_release();

    cb_affine_or_out_obj.pop_front(1);
    cb_scaled_output_obj.push_back(onetile);
}
```

Note: When weight is present but bias is absent, `cb_scaled_output` aliases to `cb_output_0`, so the result goes directly to the output CB. When both are present, `cb_scaled_output` aliases to `cb_tmp_1` for further processing.

**SFPU path** follows the same pattern but uses `copy_tile` + `mul_binary_tile`.

#### Phase 4: Optional Bias Add

**FPU path**:
```cpp
if (bias_has_value) {
    cb_output_0_obj.reserve_back(onetile);
    cb_tmp_1_obj.wait_front(onetile);

    tile_regs_acquire();
    add_tiles_init_with_dt(cb_tmp_1, cb_bias);
    add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_output_0);
    tile_regs_release();

    cb_tmp_1_obj.pop_front(onetile);
    cb_output_0_obj.push_back(onetile);
}
```

This always writes to `cb_output_0` since bias add is the final step.

#### Phase 5: Cleanup (after inner loop)

```cpp
cb_bcast_obj.pop_front(onetile);    // release batch_mean
cb_den_obj.pop_front(onetile);      // release denominator
if (weight_has_value) cb_weight_obj.pop_front(onetile);
if (bias_has_value) cb_bias_obj.pop_front(onetile);
```

This frees the broadcast tiles, allowing the writer to push new per-channel values for the next channel group.

---

## Helper Call Signatures Summary (Compute Kernel)

### FPU Path (batch_norm_kernel.cpp)

| Function | Signature | Purpose |
|----------|-----------|---------|
| `binary_op_init_common` | `(uint32_t icb0, uint32_t icb1, uint32_t ocb)` | Initialize hardware for binary ops |
| `add_tiles_init_with_dt` | `(uint32_t icb0, uint32_t icb1)` | Init add with data format reconfig (moreh_common.hpp wrapper) |
| `add_tiles` | `(uint32_t in0_cb, uint32_t in1_cb, uint32_t in0_tile_idx, uint32_t in1_tile_idx, uint32_t dst_tile_idx)` | Elementwise tile add: DST[dst] = CB0[idx0] + CB1[idx1] |
| `sub_tiles_init` | `(uint32_t icb0, uint32_t icb1)` | Init subtract |
| `sub_tiles` | `(uint32_t in0_cb, uint32_t in1_cb, uint32_t in0_tile_idx, uint32_t in1_tile_idx, uint32_t dst_tile_idx)` | Elementwise tile sub: DST[dst] = CB0[idx0] - CB1[idx1] |
| `mul_tiles_init_with_dt` | `(uint32_t icb0, uint32_t icb1)` | Init multiply with data format reconfig |
| `mul_tiles` | `(uint32_t in0_cb, uint32_t in1_cb, uint32_t in0_tile_idx, uint32_t in1_tile_idx, uint32_t dst_tile_idx)` | Elementwise tile mul: DST[dst] = CB0[idx0] * CB1[idx1] |
| `rsqrt_tile_init` | `()` | Init SFPU rsqrt |
| `rsqrt_tile` | `(uint32_t dst_tile_idx)` | In-place rsqrt on DST: DST[idx] = 1/sqrt(DST[idx]) |
| `binary_dest_reuse_tiles_init` | `<EltwiseBinaryType, EltwiseBinaryReuseDestType>(uint32_t cb)` | Init dest-reuse binary op |
| `binary_dest_reuse_tiles` | `<EltwiseBinaryType, EltwiseBinaryReuseDestType>(uint32_t cb, uint32_t src_tile_idx, uint32_t dst_tile_idx)` | Binary op reusing DST as SRCA: DST[dst] = DST[dst] OP CB[src] |
| `pack_tile_with_dt` | `(uint32_t ifrom_dst, uint32_t icb)` | Pack tile from DST to CB with format reconfig |
| `tile_regs_acquire` | `()` | Acquire DST registers for exclusive compute use |
| `tile_regs_commit` | `()` | Signal computation complete, DST ready for packer |
| `tile_regs_wait` | `()` | Wait for packer readiness |
| `tile_regs_release` | `()` | Release DST registers for next cycle |

### SFPU Path (batch_norm_sfpu_kernel.cpp)

| Function | Signature | Purpose |
|----------|-----------|---------|
| `unary_op_init_common` | `(uint32_t icb_in, uint32_t ocb)` | Initialize hardware for unary/SFPU ops |
| `copy_tile_to_dst_init_short_with_dt` | `(uint32_t old_cbid, uint32_t new_cbid, uint32_t transpose = 0)` | Reconfigure unpacker data format when switching source CBs |
| `copy_tile` | `(uint32_t icb, uint32_t itile, uint32_t idst)` | Copy tile from CB[itile] to DST[idst] |
| `add_binary_tile_init` | `()` | Init SFPU add |
| `add_binary_tile` | `(uint32_t idst0, uint32_t idst1, uint32_t odst)` | SFPU add: DST[odst] = DST[idst0] + DST[idst1] |
| `sub_binary_tile_init` | `()` | Init SFPU subtract |
| `sub_binary_tile` | `(uint32_t idst0, uint32_t idst1, uint32_t odst)` | SFPU sub: DST[odst] = DST[idst0] - DST[idst1] |
| `mul_binary_tile_init` | `()` | Init SFPU multiply |
| `mul_binary_tile` | `(uint32_t idst0, uint32_t idst1, uint32_t odst)` | SFPU mul: DST[odst] = DST[idst0] * DST[idst1] |
| `rsqrt_tile_init` | `()` | Init SFPU rsqrt |
| `rsqrt_tile` | `(uint32_t dst_tile_idx)` | In-place rsqrt: DST[idx] = 1/sqrt(DST[idx]) |
| `pack_tile` | `(uint32_t ifrom_dst, uint32_t icb)` | Pack tile from DST to CB (no format reconfig) |

### Key Differences Between Paths

1. **Operand sourcing**: FPU ops (`add_tiles`, etc.) unpack directly from CBs into SRCA/SRCB. SFPU ops require manual `copy_tile` to load from CB into DST registers first.

2. **Dest reuse**: FPU path has `binary_dest_reuse_tiles` which moves DST to SRCA in hardware, avoiding intermediate pack/unpack. SFPU path naturally keeps values in DST registers since all SFPU binary ops operate within DST space.

3. **Data format management**: FPU path uses `*_init_with_dt` wrappers (from moreh_common.hpp) that conditionally call `reconfig_data_format` when `FP32_DEST_ACC_EN` is defined. SFPU path uses `copy_tile_to_dst_init_short_with_dt` for format reconfig when switching between CBs with different data formats.

4. **Register management**: FPU path uses even DST indices only (DST[0]). SFPU path uses two DST slots per operation (DST[i*2] and DST[i*2+1]) -- slot 0 for the first operand and slot 1 for the second.

---

## Pipeline Pattern Summary

All CBs are allocated with capacity = 2 tiles and block size = 1 tile, so all are nominally **double-buffered**. However, actual overlap depends on the usage pattern:

- **CB_0 (input)**: True double-buffering possible -- reader can push next tile while compute processes current.
- **CB_2 (output)**: True double-buffering possible -- compute can push next result while writer writes current.
- **CB_1, CB_3, CB_5, CB_6 (broadcast)**: Effectively single-buffered in practice because the tile persists for the entire inner loop without being popped.
- **CB_4 (eps)**: Single-use, persists for entire kernel lifetime.
- **CB_7 (den)**: Compute-internal, persists across inner loop iterations.
- **CB_8 (temp_1)**: Compute-internal, single use per tile (produced and consumed within same iteration).

---

## Index Calculations

### How tile_start and tile_freq Drive the Iteration

On the host, the compute kernel receives three runtime args:
```cpp
auto counter = start_tile_id % cHtWt;  // tile_start: offset within first channel group
auto freq = cHtWt;                       // tile_freq: Ht * Wt, spatial tiles per channel
```

The compute kernel uses these to calculate loop bounds:
```cpp
uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;
```

- `complete_iterations`: Full channel groups (each processes `tile_freq` spatial tiles, minus `tile_start` offset for the first)
- `remaining_iterations`: Partial channel group at the end
- `tile_start` resets to 0 after the first iteration (`for (...; tile_start = 0)`)

### Reader's N/C/HW Decomposition

```cpp
uint32_t tiles_per_batch = HtWt * C;
uint32_t start_n = start_tile_id / tiles_per_batch;
uint32_t start_remaining = start_tile_id % tiles_per_batch;
uint32_t start_c = start_remaining / HtWt;
uint32_t start_t = start_remaining % HtWt;
```

The reader uses nested N/C/t loops with stride-based offset tracking:
```cpp
uint32_t tile_offset = start_n * n_stride + start_c * c_stride + start_t;
```

### Writer's Per-Channel Tile Addressing

The writer reads per-channel tensors (mean, var, weight, bias) which have shape `[N, C, 1, 1]` in tile space. It uses the same N/C decomposition but addresses per-channel tiles with `tile_offset = start_n * n_stride + start_c * c_stride` (no spatial component). This same offset is used for batch_mean, batch_var, weight, and bias since they share the same per-channel structure.

---

## Memory Access Patterns

### Read Pattern
- **Input tensor**: Sequential tile reads within each (N,C) group, stride-based jumps between channels and batches. One tile per inner loop iteration.
- **Per-channel tensors**: One tile read per (N,C) group, followed by `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast the scalar. Each value is reused for `Ht * Wt` compute iterations.
- **Epsilon**: Single fill at program start, reused throughout.

### Write Pattern
- **Output tensor**: Sequential tile writes using `start_tile_id + num_tiles_written` addressing. One tile per inner loop iteration.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (flattened to 1D row-major order) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | `num_cores_x * num_cores_y` (device-dependent) |
| **Work per core** | `num_output_tiles / num_cores` tiles (with remainder handling) |
| **Load balancing** | Two-group split via `split_work_to_cores`: group 1 gets `ceil(tiles/cores)` tiles, group 2 gets `floor(tiles/cores)` tiles |

Cores outside both groups receive zero-args and return immediately (`if (num_tiles == 0) return;`).

---

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | 1 if weight tensor provided, 0 otherwise |
| 1 | bias_has_value | uint32_t | 1 if bias tensor provided, 0 otherwise |
| 2 | cb_input | uint32_t | CB index for input tensor (c_0) |
| 3 | cb_batch_mean | uint32_t | CB index for batch mean (c_1) |
| 4 | cb_output_0 | uint32_t | CB index for output (c_2) |
| 5 | cb_batch_var | uint32_t | CB index for batch variance (c_3) |
| 6 | cb_eps | uint32_t | CB index for epsilon constant (c_4) |
| 7 | cb_den | uint32_t | CB index for denominator intermediate (c_7) |
| 8 | cb_weight | uint32_t | CB index for weight tensor (c_5) |
| 9 | cb_tmp_1 | uint32_t | CB index for temp intermediate (c_8) |
| 10 | cb_bias | uint32_t | CB index for bias tensor (c_6) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total number of tiles this core processes |
| 1 | tile_freq | uint32_t | Ht * Wt: spatial tiles per channel (broadcast group size) |
| 2 | tile_start | uint32_t | Starting offset within first channel group |

### Compile-Time Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_src | uint32_t | CB index for input tensor (c_0) |
| 1 | cb_id_eps | uint32_t | CB index for epsilon (c_4) |
| 2+ | TensorAccessorArgs | varies | Tensor accessor config for input buffer |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar_eps | uint32_t | Epsilon value packed as uint32 |
| 1 | src_addr | uint32_t | Input tensor buffer address |
| 2 | start_tile_id | uint32_t | Starting global tile index |
| 3 | num_tiles | uint32_t | Number of tiles to process |
| 4 | HtWt | uint32_t | Ht * Wt: spatial tiles per channel |
| 5 | n_stride | uint32_t | Tile stride between batches |
| 6 | c_stride | uint32_t | Tile stride between channels |
| 7 | N | uint32_t | Number of batches |
| 8 | C | uint32_t | Number of channels |
| 9 | Ht | uint32_t | Height in tiles |
| 10 | Wt | uint32_t | Width in tiles |

### Compile-Time Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | 1 if weight tensor provided |
| 1 | bias_has_value | uint32_t | 1 if bias tensor provided |
| 2 | cb_id_src | uint32_t | CB index for batch mean (c_1) |
| 3 | cb_id_dst | uint32_t | CB index for output (c_2) |
| 4 | cb_id_batch_var | uint32_t | CB index for batch variance (c_3) |
| 5 | cb_id_weight | uint32_t | CB index for weight (c_5) |
| 6 | cb_id_bias | uint32_t | CB index for bias (c_6) |
| 7+ | TensorAccessorArgs | varies | Accessor configs for batch_mean, output, batch_var, weight, bias |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | batch_mean_addr | uint32_t | Batch mean buffer address |
| 1 | batch_var_addr | uint32_t | Batch variance buffer address |
| 2 | weight_addr | uint32_t | Weight buffer address (0 if absent) |
| 3 | bias_addr | uint32_t | Bias buffer address (0 if absent) |
| 4 | dst_addr | uint32_t | Output buffer address |
| 5 | start_tile_id | uint32_t | Starting global tile index |
| 6 | num_tiles | uint32_t | Number of tiles to process |
| 7 | HtWt | uint32_t | Spatial tiles per channel |
| 8 | n_stride | uint32_t | Batch stride for per-channel tensors |
| 9 | c_stride | uint32_t | Channel stride for per-channel tensors |
| 10 | N | uint32_t | Batches |
| 11 | C | uint32_t | Channels |
| 12 | Ht | uint32_t | Height in tiles |
| 13 | Wt | uint32_t | Width in tiles |

---

## Kernel Implementations

| Kernel | Core | NOC | Input CBs | Output CBs | Operations |
|--------|------|-----|-----------|------------|------------|
| reader_batch_norm | RISC-V_0 | NOC0 | DRAM (input) | CB_0 (input), CB_4 (eps) | Read input tiles, fill eps tile |
| compute (kernel or sfpu_kernel) | TRISC | N/A | CB_0, CB_1, CB_3, CB_4, CB_5, CB_6 | CB_2, CB_7, CB_8 | Normalize, rsqrt, optional affine |
| writer_batch_norm | RISC-V_1 | NOC1 | DRAM (mean, var, weight, bias) | CB_1, CB_3, CB_5, CB_6; reads CB_2 and writes to DRAM | Read broadcast tiles with fill, write output |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp`
- **Provides**: Input tiles (CB_0) one at a time in N/C/HW order; epsilon constant tile (CB_4) once at start
- **Key detail**: Uses TensorAccessor for input buffer addressing

### Compute Kernel (FPU variant)
- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`
- **Key Logic**: Uses `binary_dest_reuse_tiles` to chain subtract and multiply without intermediate pack/unpack. Uses `moreh_common.hpp` wrappers for data type management.

### Compute Kernel (SFPU variant)
- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp`
- **Key Logic**: Uses `copy_tile` to load operands into DST registers, then SFPU binary ops. Requires `copy_tile_to_dst_init_short_with_dt` calls to reconfigure unpacker when switching between CBs with different data formats.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp`
- **Consumes**: Output tiles (CB_2) for writing to DRAM
- **Provides**: Per-channel broadcast tiles (CB_1, CB_3, CB_5, CB_6) with `FILL_TILE_WITH_FIRST_ELEMENT`
- **Key detail**: The fill function reads the first element of a tile and replicates it across all 1024 positions, enabling scalar broadcast

---

## Implementation Notes

### FP32 Dest Accumulation Mode

When `fp32_dest_acc_en` is true:
1. The SFPU kernel variant is selected (`batch_norm_sfpu_kernel.cpp`)
2. All input CBs are configured with `UnpackToDestMode::UnpackToDestFp32`
3. The `FP32_DEST_ACC_EN` preprocessor define enables format reconfig wrappers in `moreh_common.hpp`
4. DST registers store full FP32 values, avoiding BF16 truncation between operations

### Conditional Dataflow Defines

The program factory sets up macro defines for fill operations based on data type:
```cpp
if (input_tensor.dtype() == DataType::FLOAT32) {
    dataflow_defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element<float>";
    dataflow_defines["FILL_WITH_VALUE_FLOAT"] = "fill_with_val<1024, float>";
} else {
    dataflow_defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element_bfloat16";
    dataflow_defines["FILL_WITH_VALUE"] = "fill_with_val_bfloat16";
}
```

Note the asymmetry: FP32 uses `FILL_WITH_VALUE_FLOAT` while BF16 uses `FILL_WITH_VALUE` (different macro names). The reader kernel checks for both with `#ifdef`.

### Applicability to Layer Norm RM

Key patterns from batch_norm relevant to layer_norm_rm:

1. **Broadcast reuse pattern**: Batch norm broadcasts per-channel stats across spatial tiles. Layer norm will broadcast per-row stats across width tiles. The CB persistence pattern (wait_front before loop, pop_front after loop) transfers directly.

2. **Compute chain pattern**: The `sub -> mul -> optional mul -> optional add` chain is similar to layer norm's `sub_mean -> mul_rsqrt_var -> optional gamma -> optional beta`.

3. **Two-kernel-variant pattern**: Having both FPU and SFPU compute kernel variants selected by `fp32_dest_acc_en` is a standard pattern worth replicating.

4. **Key difference**: Layer norm computes its own mean and variance via reduction, while batch norm receives them as pre-computed inputs. This means the layer norm compute kernel will need additional reduce phases not present in batch norm.

5. **CB aliasing for conditional paths**: The `cb_affine_or_out` / `cb_scaled_output` aliasing pattern elegantly handles the optional weight/bias with minimal branching.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How do the compute kernel SFPU binary operations work? Specifically: sub_tiles, add_tiles, mul_tiles, rsqrt_tile, binary_dest_reuse_tiles, pack_tile_with_dt, add_tiles_init_with_dt, and the tile_regs_acquire/commit/wait/release pattern."
   **Reason**: Needed to understand exact semantics of all compute API calls used in both kernel variants.
   **Key Findings**: The acquire/commit/wait/release pattern manages DST register ownership between math and pack stages. FPU ops unpack from CBs directly; SFPU ops work in DST register space. `pack_tile_with_dt` handles format conversion on pack.

2. **Query**: "What is the experimental::CircularBuffer API in tt-metal compute kernels?"
   **Reason**: Both kernel variants use this API extensively for CB synchronization.
   **Key Findings**: `wait_front(n)` blocks until n tiles available for reading; `pop_front(n)` marks consumed; `reserve_back(n)` blocks until space for n tiles; `push_back(n)` marks written. The same CB hardware channel is shared between compute and dataflow kernels for producer/consumer coordination.

3. **Query**: "What is binary_dest_reuse_tiles in tt-metal?"
   **Reason**: This is a key optimization in the FPU compute path that chains sub and mul without intermediate pack/unpack.
   **Key Findings**: `DEST_TO_SRCA` copies DST to SRCA register, then fetches the other operand from a CB into SRCB, and performs the binary op. This eliminates a full pack/unpack cycle for intermediate results.

4. **Query**: "What is the difference between the FPU path and the SFPU path in tt-metal compute kernels?"
   **Reason**: Needed to understand why two kernel variants exist and when each is appropriate.
   **Key Findings**: FPU uses the matrix engine, operates on data unpacked from CBs. SFPU uses the vector processing unit, operates on data in DST registers. SFPU is needed for full FP32 accumulation since FPU may truncate intermediates. The `fp32_dest_acc_en` flag selects SFPU path.

5. **Query**: "What is unary_op_init_common vs binary_op_init_common?"
   **Reason**: The two kernel variants use different init functions.
   **Key Findings**: `binary_op_init_common(icb0, icb1, ocb)` sets up hardware for two-input binary ops. `unary_op_init_common(icb, ocb)` sets up for single-input ops. The SFPU path uses unary init because it manually copies tiles to DST rather than having the hardware unpack two sources.

6. **Query**: "What does the split_work_to_cores function do in tt-metal?"
   **Reason**: Understanding core distribution strategy.
   **Key Findings**: Returns 6-tuple (num_cores, all_cores, core_group_1, core_group_2, tiles_per_group_1, tiles_per_group_2). Group 1 gets ceil(tiles/cores), group 2 gets floor(tiles/cores). Standard work distribution utility.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding CB creation helper
   **Key Information**: `create_cb()` is a convenience wrapper that creates a CircularBufferConfig with specified page_size, num_pages, and data format, then calls CreateCircularBuffer.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Understanding how scalar broadcast works in dataflow kernels
   **Key Information**: `fill_tile_with_first_element_bfloat16` reads the first 16-bit element and double-packs it into uint32, then writes to all 512 uint32 words. `fill_with_val_bfloat16` takes a pre-packed uint32 scalar and writes it directly. Both produce a tile where every element equals the scalar.

3. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding `_with_dt` wrapper functions
   **Key Information**: These wrappers conditionally call `reconfig_data_format` / `reconfig_data_format_srca` / `pack_reconfig_data_format` when `FP32_DEST_ACC_EN` is defined. This handles the hardware requirement to reconfigure the data path when operating on mixed-precision data with FP32 destination accumulation.

4. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Exact signatures of SFPU binary ops
   **Key Information**: `add_binary_tile(idst0, idst1, odst)`, `sub_binary_tile(idst0, idst1, odst)`, `mul_binary_tile(idst0, idst1, odst)` -- all operate purely in DST register space with three register indices.

5. **Source**: `tt_metal/hw/inc/api/compute/tile_move_copy.h`
   **Reason**: Exact signature of `copy_tile_to_dst_init_short_with_dt`
   **Key Information**: `copy_tile_to_dst_init_short_with_dt(uint32_t old_cbid, uint32_t new_cbid, uint32_t transpose = 0)` -- reconfigures SRCA data format from old CB to new CB, needed when switching between CBs with different data formats.
