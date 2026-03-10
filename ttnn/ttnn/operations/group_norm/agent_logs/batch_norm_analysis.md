# Batch Norm Implementation Analysis

## Overview

Batch normalization computes `y = gamma * (x - mean) / sqrt(var + eps) + beta` on tiled tensors. The operation receives pre-computed batch mean and batch variance as separate input tensors (not computed inline). Weight (gamma) and bias (beta) are optional affine parameters.

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

**Compute focus**: This analysis emphasizes the compute kernel structure, CB layout for intermediates, multi-pass data reuse patterns, scalar/constant CB setup, and binary op broadcast patterns. Reader/writer details are summarized at interface level only.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32) |
| **Unit size** | 1 tile |
| **Total units** | `output.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Outer loop over "complete channel iterations", inner loop over HtWt tiles within each channel |

## Tensor Format and Layout

### Input Tensors

| Property | Input (x) | batch_mean | batch_var | weight (gamma) | bias (beta) |
|----------|-----------|------------|-----------|----------------|-------------|
| **Logical shape** | [N, C, H, W] | [N, C, 1, 1] | [N, C, 1, 1] | [1, C, 1, 1] | [1, C, 1, 1] |
| **Dimension convention** | NCHW | NCHW | NCHW | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | same as input | same as input | same as input | same as input |

### Output Tensor

| Property | Output (y) |
|----------|-----------|
| **Logical shape** | [N, C, H, W] |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | configurable (defaults to input dtype) |

### Layout Transformations

No tilize/untilize or reshard conversions occur. All tensors are expected in TILE_LAYOUT. The per-channel parameters (mean, var, weight, bias) are scalar values stored in the first element of a tile; the writer kernel reads them from DRAM and uses `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast the scalar across all 1024 elements of the tile before pushing to the CB. This is a **software-level scalar broadcast** -- not a hardware broadcast instruction.

## Data Flow Pattern

### High-Level Flow

```
Reader kernel (RISCV_0):
  - Reads input tiles from DRAM -> cb_input (c_0)
  - Fills eps tile once -> cb_eps (c_4)

Writer kernel (RISCV_1):
  - For each channel group:
    1. Reads batch_mean tile -> fills with first element -> cb_batch_mean (c_1)
    2. Reads batch_var tile -> fills with first element -> cb_batch_var (c_3)
    3. Reads weight tile -> fills with first element -> cb_weight (c_5) [optional]
    4. Reads bias tile -> fills with first element -> cb_bias (c_6) [optional]
    5. For each spatial tile in channel: reads cb_output (c_2) and writes to DRAM

Compute kernel (RISCV_2):
  - Waits for eps tile once (persists for entire program)
  - For each channel group:
    Phase 1: compute den = 1/sqrt(var + eps) [one tile, persists for spatial loop]
    Phase 2: for each spatial tile in channel:
      a. normalized = (input - mean) * den
      b. scaled = normalized * weight   [optional]
      c. output = scaled + bias         [optional]
```

### Detailed Data Flow Table

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (input) | cb_input (c_0) | reserve_back, push_back (per tile) |
| 2 | Reader | runtime arg (eps) | cb_eps (c_4) | reserve_back, FILL_WITH_VALUE, push_back (once) |
| 3 | Writer | DRAM (batch_mean) | cb_batch_mean (c_1) | reserve_back, noc_async_read, FILL_TILE_WITH_FIRST_ELEMENT, push_back (per channel) |
| 4 | Writer | DRAM (batch_var) | cb_batch_var (c_3) | reserve_back, noc_async_read, FILL_TILE_WITH_FIRST_ELEMENT, push_back (per channel) |
| 5 | Writer | DRAM (weight) | cb_weight (c_5) | same as above (per channel, optional) |
| 6 | Writer | DRAM (bias) | cb_bias (c_6) | same as above (per channel, optional) |
| 7 | Compute | cb_batch_var (c_3), cb_eps (c_4) | cb_den (c_7) | wait_front, add, rsqrt, pack (per channel) |
| 8 | Compute | cb_input (c_0), cb_batch_mean (c_1), cb_den (c_7) | cb_tmp_1 or cb_output (c_8/c_2) | wait_front, sub, mul_dest_reuse, pack (per spatial tile) |
| 9 | Compute | cb_tmp_1 (c_8), cb_weight (c_5) | cb_tmp_1 or cb_output (c_8/c_2) | mul, pack (per spatial tile, optional) |
| 10 | Compute | cb_tmp_1 (c_8), cb_bias (c_6) | cb_output (c_2) | add, pack (per spatial tile, optional) |
| 11 | Writer | cb_output (c_2) | DRAM (output) | wait_front, noc_async_write, pop_front (per tile) |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input x tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_1 | cb_batch_mean | Broadcast mean tile | 2 tiles | 1 tile | Double | Writer | Compute | Channel (per channel group) |
| c_2 | cb_output | Final output staging | 2 tiles | 1 tile | Double | Compute | Writer | Block (per tile) |
| c_3 | cb_batch_var | Broadcast variance tile | 2 tiles | 1 tile | Double | Writer | Compute | Channel (consumed per channel) |
| c_4 | cb_eps | Epsilon constant tile | 2 tiles | 1 tile | Double | Reader | Compute | Program (entire kernel) |
| c_5 | cb_weight | Broadcast gamma tile | 2 tiles | 1 tile | Double | Writer | Compute | Channel (per channel group) |
| c_6 | cb_bias | Broadcast beta tile | 2 tiles | 1 tile | Double | Writer | Compute | Channel (per channel group) |
| c_7 | cb_den | 1/sqrt(var+eps) intermediate | 2 tiles | 1 tile | Double | Compute | Compute | Channel (per channel group) |
| c_8 | cb_tmp_1 | Intermediate results | 2 tiles | 1 tile | Double | Compute | Compute | Block (per tile) |

### CB Lifetime Details and Multi-Pass Data Reuse

**Program-lifetime CBs (persist for entire kernel execution):**
- **cb_eps (c_4)**: Filled once by reader at program start. The compute kernel does `cb_wait_front(cb_eps, 1)` at the beginning and `cb_pop_front(cb_eps, 1)` at the very end. The eps tile remains in the CB for the entire kernel, used in every channel's `var + eps` computation. This is the key constant-reuse pattern.

**Channel-lifetime CBs (persist across spatial tiles within one channel):**
- **cb_batch_mean (c_1)**: Pushed once per channel by the writer. The compute kernel does `cb_wait_front(cb_bcast, 1)` at the start of `batchnorm_bcast_tiles()` and `cb_pop_front(cb_bcast, 1)` at the end. It is consumed by every spatial tile's subtraction within that channel.
- **cb_den (c_7)**: Computed once per channel by the compute kernel itself (`var + eps` then `rsqrt`). Used in every spatial tile's normalization multiply. Popped at end of `batchnorm_bcast_tiles()`.
- **cb_weight (c_5)**: Same pattern as cb_batch_mean. Pushed once per channel, consumed by every spatial tile's gamma multiply.
- **cb_bias (c_6)**: Same pattern as cb_batch_mean. Pushed once per channel, consumed by every spatial tile's beta addition.
- **cb_batch_var (c_3)**: Pushed once per channel, consumed once at the start of `batchnorm_bcast_tiles()` to compute cb_den. Does NOT persist across spatial tiles (popped immediately after den computation).

**Block-lifetime CBs (produced and consumed per spatial tile):**
- **cb_input (c_0)**: One tile produced by reader, consumed by compute per spatial tile.
- **cb_output (c_2)**: One tile produced by compute, consumed by writer per spatial tile.
- **cb_tmp_1 (c_8)**: Intermediate scratchpad. When both weight and bias are present, normalization result goes to c_8, then gamma multiply goes to c_8, then bias add goes to c_2. When only weight: normalization goes to c_8, gamma multiply goes to c_2. When neither: normalization goes directly to c_2.

### CB Output Routing Logic

The compute kernel uses conditional CB assignment based on whether weight/bias are present:

```
cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0
cb_scaled_output = (bias_has_value)                     ? cb_tmp_1 : cb_output_0
```

| weight | bias | Normalization output -> | Gamma multiply output -> | Beta add output -> |
|--------|------|------------------------|--------------------------|---------------------|
| no  | no  | cb_output (c_2) | N/A | N/A |
| yes | no  | cb_tmp_1 (c_8) | cb_output (c_2) | N/A |
| no  | yes | cb_tmp_1 (c_8) | N/A | cb_output (c_2) |
| yes | yes | cb_tmp_1 (c_8) | cb_tmp_1 (c_8) | cb_output (c_2) |

## Pipeline Pattern Summary

All CBs have capacity = 2 tiles and block size = 1 tile, giving **double-buffering** throughout. This allows reader/writer data movement to overlap with compute operations.

## Compute Kernel Structure -- Two Variants

The program factory selects between two compute kernel files based on `fp32_dest_acc_en`:

| Condition | Kernel File | Init Call |
|-----------|------------|-----------|
| `fp32_dest_acc_en = false` | `batch_norm_kernel.cpp` | `binary_op_init_common(cb_other, cb_bcast, cb_output_0)` |
| `fp32_dest_acc_en = true` | `batch_norm_sfpu_kernel.cpp` | `unary_op_init_common(cb_other, cb_output_0)` |

### Why Two Variants

- **FPU path** (`batch_norm_kernel.cpp`): Uses the matrix engine (FPU) for binary operations. Calls `add_tiles()`, `sub_tiles()`, `mul_tiles()` and the specialized `binary_dest_reuse_tiles()`. These unpack from two source CBs into SRCA/SRCB.
- **SFPU path** (`batch_norm_sfpu_kernel.cpp`): Uses the vector engine (SFPU) for binary operations. Calls `copy_tile()` into DST, then `add_binary_tile()`, `sub_binary_tile()`, `mul_binary_tile()`. These load both operands into DST register slots (using `i*2` and `i*2+1` indexing), then operate on DST directly. Required when `fp32_dest_acc_en = true` to maintain full 32-bit precision throughout the pipeline, as SFPU operations keep data in the 32-bit DST registers without truncation through source registers.

When `fp32_dest_acc_en = true`, the program factory also sets `UnpackToDestMode::UnpackToDestFp32` for all relevant CBs (c_0 through c_8 except c_2 output) to ensure data is unpacked into DST as 32-bit floats.

### Compute Kernel main() Structure (Both Variants Share This)

```cpp
void kernel_main() {
    // Runtime args
    uint32_t num_tiles = get_arg_val<uint32_t>(0);    // total tiles this core processes
    uint32_t tile_freq = get_arg_val<uint32_t>(1);    // HtWt = spatial tiles per channel
    uint32_t tile_start = get_arg_val<uint32_t>(2);   // offset into first channel

    // Compile-time args: CB IDs and flags
    constexpr uint32_t weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bias_has_value = get_compile_time_arg_val(1) == 1;
    constexpr auto cb_input      = get_compile_time_arg_val(2);   // c_0
    constexpr auto cb_batch_mean = get_compile_time_arg_val(3);   // c_1
    constexpr auto cb_output_0   = get_compile_time_arg_val(4);   // c_2
    constexpr auto cb_batch_var  = get_compile_time_arg_val(5);   // c_3
    constexpr auto cb_eps        = get_compile_time_arg_val(6);   // c_4
    constexpr auto cb_den        = get_compile_time_arg_val(7);   // c_7
    constexpr auto cb_weight     = get_compile_time_arg_val(8);   // c_5
    constexpr auto cb_tmp_1      = get_compile_time_arg_val(9);   // c_8
    constexpr auto cb_bias       = get_compile_time_arg_val(10);  // c_6

    if (num_tiles == 0) return;

    // Init
    binary_op_init_common(cb_input, cb_batch_mean, cb_output_0);  // FPU variant
    // OR: unary_op_init_common(cb_input, cb_output_0);           // SFPU variant

    // Split work into complete channel iterations + remainder
    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    // Wait for eps tile ONCE (program lifetime)
    cb_wait_front(cb_eps, 1);

    // Process complete channel groups
    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles(
            cb_batch_mean, cb_input, tile_freq, tile_start,
            cb_batch_var, cb_eps, cb_den, cb_weight, cb_bias,
            cb_tmp_1, cb_output_0, weight_has_value, bias_has_value);
    }
    // Process remaining tiles in partial channel
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles(
            cb_batch_mean, cb_input, remaining_iterations, tile_start,
            cb_batch_var, cb_eps, cb_den, cb_weight, cb_bias,
            cb_tmp_1, cb_output_0, weight_has_value, bias_has_value);
    }

    // Pop eps at program end
    cb_pop_front(cb_eps, 1);
}
```

### Channel Iteration Logic (freq/counter)

The compute kernel needs to know when a new channel begins, so it can wait for fresh mean/var/weight/bias tiles. This is controlled by two runtime args:

- **`tile_freq`** = `cHt * cWt` = number of spatial tiles per channel. This is the "frequency" at which new per-channel parameters appear.
- **`tile_start`** = `start_tile_id % cHtWt` = offset into the first channel for this core's starting position.

The kernel computes:
- `complete_iterations = (num_tiles + tile_start) / tile_freq` -- number of full channels processed
- `remaining_iterations = (num_tiles + tile_start) % tile_freq` -- leftover spatial tiles in a partial channel

On the first call to `batchnorm_bcast_tiles()`, processing starts at `tile_start` within the inner loop (`for j = tile_start; j < freq`). On subsequent calls, `tile_start = 0` (reset in the for-loop increment).

### batchnorm_bcast_tiles() -- FPU Variant (Detailed)

```
function batchnorm_bcast_tiles(cb_bcast=mean, cb_other=input, freq, tile_start,
                                cb_batch_var, cb_eps, cb_den, cb_weight, cb_bias,
                                cb_tmp_1, cb_output_0, weight_has, bias_has):

  // --- Phase 1: Compute denominator (once per channel) ---
  cb_reserve_back(cb_den, 1)
  cb_wait_front(cb_batch_var, 1)

  tile_regs_acquire()
  add_tiles_init_with_dt(cb_batch_var, cb_eps)    // var + eps
  add_tiles(cb_batch_var, cb_eps, 0, 0, dst0)
  rsqrt_tile_init()                                // 1/sqrt(result)
  rsqrt_tile(dst0)
  tile_regs_commit()

  tile_regs_wait()
  pack_tile_with_dt(dst0, cb_den)                  // pack to cb_den
  tile_regs_release()

  cb_pop_front(cb_batch_var, 1)                    // var consumed
  cb_push_back(cb_den, 1)                          // den available

  // --- Phase 2: Wait for channel-persistent tiles ---
  cb_wait_front(cb_bcast, 1)     // mean tile (persists for all spatial tiles)
  cb_wait_front(cb_den, 1)       // den tile (persists for all spatial tiles)
  if (weight_has) cb_wait_front(cb_weight, 1)
  if (bias_has) cb_wait_front(cb_bias, 1)

  // --- Phase 3: Process each spatial tile ---
  for j = tile_start to freq-1:

    // Step A: normalized = (input - mean)
    cb_wait_front(cb_other, 1)
    cb_reserve_back(cb_affine_or_out, 1)

    tile_regs_acquire()
    sub_tiles_init(cb_other, cb_bcast)
    sub_tiles(cb_other, cb_bcast, 0, 0, dst0)      // dst0 = input - mean

    // Step B: normalized *= den  (dest reuse: dst0 = dst0 * den)
    binary_dest_reuse_tiles_init<ELWMUL, DEST_TO_SRCA>(cb_den)
    binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_den, 0, 0)
    tile_regs_commit()

    tile_regs_wait()
    pack_tile_with_dt(0, cb_affine_or_out)
    tile_regs_release()

    cb_push_back(cb_affine_or_out, 1)
    cb_pop_front(cb_other, 1)                       // input tile consumed

    // Step C: result *= weight (optional)
    if (weight_has):
      cb_reserve_back(cb_scaled_output, 1)
      cb_wait_front(cb_affine_or_out, 1)

      tile_regs_acquire()
      mul_tiles_init_with_dt(cb_affine_or_out, cb_weight)
      mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0)
      tile_regs_commit()

      tile_regs_wait()
      pack_tile_with_dt(dst0, cb_scaled_output)
      tile_regs_release()

      cb_pop_front(cb_affine_or_out, 1)
      cb_push_back(cb_scaled_output, 1)

    // Step D: result += bias (optional)
    if (bias_has):
      cb_reserve_back(cb_output_0, 1)
      cb_wait_front(cb_tmp_1, 1)

      tile_regs_acquire()
      add_tiles_init_with_dt(cb_tmp_1, cb_bias)
      add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0)
      tile_regs_commit()

      tile_regs_wait()
      pack_tile_with_dt(dst0, cb_output_0)
      tile_regs_release()

      cb_pop_front(cb_tmp_1, 1)
      cb_push_back(cb_output_0, 1)

  // --- Phase 4: Release channel-persistent tiles ---
  cb_pop_front(cb_bcast, 1)    // mean
  cb_pop_front(cb_den, 1)      // den
  if (weight_has) cb_pop_front(cb_weight, 1)
  if (bias_has) cb_pop_front(cb_bias, 1)
```

### Key Compute API Calls -- Exact Signatures

**FPU variant (`batch_norm_kernel.cpp`):**

| Call | Signature | Purpose |
|------|-----------|---------|
| `binary_op_init_common` | `binary_op_init_common(cb_other, cb_bcast, cb_output_0)` | One-time init for binary FPU ops |
| `add_tiles_init_with_dt` | `add_tiles_init_with_dt(cb_batch_var, cb_eps)` | Init add with data format reconfig (FP32_DEST_ACC_EN aware) |
| `add_tiles` | `add_tiles(cb_batch_var, cb_eps, 0, 0, dst0)` | var + eps elementwise |
| `rsqrt_tile_init` | `rsqrt_tile_init()` | Init SFPU rsqrt |
| `rsqrt_tile` | `rsqrt_tile(dst0)` | 1/sqrt(dst0) in-place on DST |
| `sub_tiles_init` | `sub_tiles_init(cb_other, cb_bcast)` | Init subtraction (NO _with_dt -- note this) |
| `sub_tiles` | `sub_tiles(cb_other, cb_bcast, 0, 0, 0)` | input - mean |
| `binary_dest_reuse_tiles_init<ELWMUL, DEST_TO_SRCA>` | `binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den)` | Init dest-reuse multiply |
| `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>` | `binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0)` | DST[0] = DST[0] * den[0] |
| `mul_tiles_init_with_dt` | `mul_tiles_init_with_dt(cb_affine_or_out, cb_weight)` | Init multiply with data format reconfig |
| `mul_tiles` | `mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0)` | normalized * gamma |
| `add_tiles_init_with_dt` | `add_tiles_init_with_dt(cb_tmp_1, cb_bias)` | Init add with data format reconfig |
| `add_tiles` | `add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0)` | scaled + beta |
| `pack_tile_with_dt` | `pack_tile_with_dt(dst0, ocb)` | Pack DST to CB with data format reconfig |

**SFPU variant (`batch_norm_sfpu_kernel.cpp`):**

| Call | Signature | Purpose |
|------|-----------|---------|
| `unary_op_init_common` | `unary_op_init_common(cb_other, cb_output_0)` | One-time init for SFPU ops |
| `copy_tile_to_dst_init_short_with_dt` | `copy_tile_to_dst_init_short_with_dt(cb_eps, cb_batch_var)` | Init copy with data format reconfig |
| `copy_tile` | `copy_tile(cb_batch_var, i, i*2)` | Copy to DST slot i*2 |
| `add_binary_tile_init` | `add_binary_tile_init()` | Init SFPU add |
| `add_binary_tile` | `add_binary_tile(i*2, i*2+1, i*2)` | DST[i*2] = DST[i*2] + DST[i*2+1] |
| `rsqrt_tile_init` | `rsqrt_tile_init()` | Same as FPU variant |
| `rsqrt_tile` | `rsqrt_tile(i*2)` | Same as FPU variant |
| `pack_tile` | `pack_tile(i*2, cb_den)` | Pack to CB (no _with_dt in SFPU path) |
| `sub_binary_tile_init` | `sub_binary_tile_init()` | Init SFPU subtract |
| `sub_binary_tile` | `sub_binary_tile(i*2, i*2+1, i*2)` | DST[i*2] = DST[i*2] - DST[i*2+1] |
| `mul_binary_tile_init` | `mul_binary_tile_init()` | Init SFPU multiply |
| `mul_binary_tile` | `mul_binary_tile(i*2, i*2+1, i*2)` | DST[i*2] = DST[i*2] * DST[i*2+1] |

### Key Difference: binary_dest_reuse_tiles (FPU) vs Copy-then-Op (SFPU)

In the FPU path, the normalization step chains two operations within a single acquire/commit block:
1. `sub_tiles()` puts `input - mean` into `DST[0]`
2. `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>()` reuses DST[0] (moves it to SRCA) and unpacks `cb_den` into SRCB, then stores `SRCA * SRCB` back in DST[0]

This avoids an intermediate pack/unpack cycle. The SFPU path achieves the same by using the two-slot DST pattern (`i*2` for accumulator, `i*2+1` for new operand).

## Scalar/Constant CB Setup

### Epsilon (cb_eps, c_4)

**Setup by reader kernel** at program start (before any tile processing):

```cpp
// Reader kernel
union { float f; uint32_t u; } scalar;
scalar.u = eps;  // eps passed as packed uint32_t runtime arg
cb_reserve_back(cb_id_eps, 1);
#ifdef FILL_WITH_VALUE_FLOAT
    FILL_WITH_VALUE_FLOAT(cb_id_eps, scalar.f);  // float32 path: fill 1024 floats
#endif
#ifdef FILL_WITH_VALUE
    FILL_WITH_VALUE(cb_id_eps, eps);              // bfloat16 path: fill 512 packed uint32s
#endif
cb_push_back(cb_id_eps, 1);
```

The eps is passed as runtime arg 0 from the host, packed appropriately:
- FLOAT32: `std::bit_cast<uint32_t>(eps)`
- BFLOAT16: `pack_two_bfloat16_into_uint32({eps, eps})`

### Per-Channel Parameters (mean, var, weight, bias)

**Setup by writer kernel** once per channel. The writer reads a single tile from DRAM using TensorAccessor, then broadcasts the first element:

```cpp
// For each of mean, var, weight, bias:
cb_reserve_back(cb_id, 1);
noc_async_read_tile(tile_offset, accessor, l1_write_addr);
noc_async_read_barrier();
FILL_TILE_WITH_FIRST_ELEMENT(cb_id);  // Reads element [0], fills all 1024 positions
cb_push_back(cb_id, 1);
```

`FILL_TILE_WITH_FIRST_ELEMENT` is defined as a preprocessor macro that resolves to either `fill_tile_with_first_element_bfloat16()` or `fill_tile_with_first_element<float>()` depending on input dtype. These functions read the first element from the tile buffer in L1 and replicate it across all 1024 elements of the tile (or 512 packed uint32 pairs for bfloat16).

## Index Calculations

### Reader Input Tile Indexing

The reader iterates in N, C, spatial-tile order:

```
tile_offset = start_n * n_stride + start_c * c_stride + start_t
```

Where:
- `n_stride = aHt * aWt * aC * (aN > 1)` -- stride between batch elements (0 if N=1)
- `c_stride = aHt * aWt * (aC > 1)` -- stride between channels (0 if C=1)
- `HtWt = cHt * cWt` -- spatial tiles per channel

The reader advances `tile_offset++` for each spatial tile, then adds `next_channel_shift = c_stride - HtWt` when moving to the next channel, and `next_batch_shift = n_stride - c_stride * C` when moving to the next batch.

### Writer Parameter Tile Indexing

The writer indexes mean/var/weight/bias using the same N,C structure but only one tile per (N,C) pair:

```
tile_offset = start_n * n_stride + start_c * c_stride
// advances by c_stride per channel, then next_batch_shift per batch
```

### Compute Frequency-Based Channel Tracking

The compute kernel does not directly track N,C indices. Instead it uses:
- `tile_freq = HtWt` tiles per channel
- `tile_start = start_tile_id % HtWt` offset into first channel

This cleanly separates compute from the N,C,H,W addressing details. Every `tile_freq` tiles, the compute kernel expects fresh mean/var/weight/bias tiles from the writer.

## Memory Access Patterns

### Read Pattern
- **Input (reader)**: Sequential tile reads within each spatial block (HtWt contiguous tiles), with stride jumps between channels and batches. One tile at a time.
- **Parameters (writer)**: One tile per channel, read via TensorAccessor. The tile_offset increments by `c_stride` per channel.
- **Eps (reader)**: Single tile read, written once, persists for program lifetime.

### Write Pattern
- **Output (writer)**: Sequential tile writes using `noc_async_write_tile` with output tile ID `start_tile_id + num_tiles_written`. Contiguous within spatial blocks, jumps between channels/batches implicit from tile ID ordering.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (flattened to 1D row-major) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | `num_cores_x * num_cores_y` (device-dependent) |
| **Work per core** | `num_output_tiles / num_cores` tiles (+1 for remainder cores) |
| **Load balancing** | Two-group split: `core_group_1` gets `ceil(tiles/cores)`, `core_group_2` gets `floor(tiles/cores)` |
| **Remainder handling** | Unused cores receive zero-args and exit early (`num_tiles == 0`) |

Work is split using `tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major=true)`. Each core receives a contiguous range of output tiles starting at `start_tile_id` and processing `num_tiles_per_core` tiles. The `start_tile_id` is accumulated across cores.

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t (bool) | Whether gamma is provided (controls branch elimination) |
| 1 | bias_has_value | uint32_t (bool) | Whether beta is provided (controls branch elimination) |
| 2 | cb_input | uint32_t | CB index for input tensor (c_0) |
| 3 | cb_batch_mean | uint32_t | CB index for batch mean (c_1) |
| 4 | cb_output_0 | uint32_t | CB index for output tensor (c_2) |
| 5 | cb_batch_var | uint32_t | CB index for batch variance (c_3) |
| 6 | cb_eps | uint32_t | CB index for epsilon constant (c_4) |
| 7 | cb_den | uint32_t | CB index for denominator intermediate (c_7) |
| 8 | cb_weight | uint32_t | CB index for weight/gamma (c_5) |
| 9 | cb_tmp_1 | uint32_t | CB index for intermediate scratch (c_8) |
| 10 | cb_bias | uint32_t | CB index for bias/beta (c_6) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total output tiles for this core |
| 1 | tile_freq | uint32_t | HtWt: spatial tiles per channel (channel boundary frequency) |
| 2 | tile_start | uint32_t | `start_tile_id % HtWt`: offset into first channel |

### Compile-Time Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_src | uint32_t | CB index for input (c_0) |
| 1 | cb_id_eps | uint32_t | CB index for eps (c_4) |
| 2+ | TensorAccessorArgs | ... | Input tensor accessor metadata |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar_eps | uint32_t | Epsilon value (packed as float32 or dual bfloat16) |
| 1 | src_addr | uint32_t | Input tensor buffer address |
| 2 | start_tile_id | uint32_t | First output tile ID for this core |
| 3 | num_tiles | uint32_t | Tiles to process on this core |
| 4 | HtWt | uint32_t | Spatial tiles per channel |
| 5 | n_stride | uint32_t | Tile stride between batch elements (input) |
| 6 | c_stride | uint32_t | Tile stride between channels (input) |
| 7 | N | uint32_t | Number of batches (output dims) |
| 8 | C | uint32_t | Number of channels (output dims) |
| 9 | Ht | uint32_t | Height in tiles (output dims) |
| 10 | Wt | uint32_t | Width in tiles (output dims) |

### Compile-Time Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t (bool) | Whether gamma is provided |
| 1 | bias_has_value | uint32_t (bool) | Whether beta is provided |
| 2 | cb_id_src | uint32_t | CB index for batch_mean (c_1) |
| 3 | cb_id_dst | uint32_t | CB index for output (c_2) |
| 4 | cb_id_batch_var | uint32_t | CB index for batch_var (c_3) |
| 5 | cb_id_weight | uint32_t | CB index for weight (c_5) |
| 6 | cb_id_bias | uint32_t | CB index for bias (c_6) |
| 7+ | TensorAccessorArgs | ... | Accessor args for mean, output, var, weight, bias (chained) |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | batch_mean_addr | uint32_t | Batch mean buffer address |
| 1 | batch_var_addr | uint32_t | Batch variance buffer address |
| 2 | weight_addr | uint32_t | Weight buffer address (0 if absent) |
| 3 | bias_addr | uint32_t | Bias buffer address (0 if absent) |
| 4 | output_addr | uint32_t | Output buffer address |
| 5 | start_tile_id | uint32_t | First output tile for this core |
| 6 | num_tiles | uint32_t | Tiles to process |
| 7 | HtWt | uint32_t | Spatial tiles per channel |
| 8 | n_stride | uint32_t | Tile stride between batches (param tensors) |
| 9 | c_stride | uint32_t | Tile stride between channels (param tensors) |
| 10 | N | uint32_t | Number of batches |
| 11 | C | uint32_t | Number of channels |
| 12 | Ht | uint32_t | Height in tiles |
| 13 | Wt | uint32_t | Width in tiles |

## Kernel Implementations

| Kernel | Core | NOC | Input CBs | Output CBs | Operations |
|--------|------|-----|-----------|------------|------------|
| reader_batch_norm | RISCV_0 | NOC0 | DRAM (input) | c_0 (input), c_4 (eps) | Read input tiles + fill eps constant tile |
| batch_norm_kernel / batch_norm_sfpu_kernel | RISCV_2 | N/A | c_0, c_1, c_3, c_4, c_5, c_6 | c_2, c_7, c_8 | sub, rsqrt, mul (dest_reuse or SFPU), add |
| writer_batch_norm | RISCV_1 | NOC1 | DRAM (mean,var,weight,bias) / c_2 | c_1, c_3, c_5, c_6 / DRAM (output) | Read params + fill_tile_with_first_element, write output tiles |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp`
- **Provides to compute**: cb_input (c_0) with one input tile per spatial position, cb_eps (c_4) with constant eps tile (once)

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp`
- **Provides to compute**: cb_batch_mean (c_1), cb_batch_var (c_3), cb_weight (c_5), cb_bias (c_6) -- all broadcast-filled, once per channel
- **Consumes from compute**: cb_output (c_2), writes to DRAM

### Compute Kernel (FPU variant)
- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`
- **Key Logic**: Uses `binary_dest_reuse_tiles` to chain subtraction and multiplication in a single DST register session, avoiding an intermediate CB round-trip. The `_with_dt` helper variants handle `FP32_DEST_ACC_EN` format reconfiguration automatically.

### Compute Kernel (SFPU variant)
- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp`
- **Key Logic**: Uses the two-slot DST pattern (`copy_tile(A, i, i*2)` and `copy_tile(B, i, i*2+1)` then `op_binary_tile(i*2, i*2+1, i*2)`) for all binary operations. This maintains FP32 precision throughout by keeping data in DST registers.

## Implementation Notes

### Split Reader Pattern
This operation uses a **split reader** pattern where the "reader" kernel reads input data and eps, while the "writer" kernel reads parameter tensors (mean, var, weight, bias) AND writes output. This is a common pattern in batch_norm-style operations where per-channel parameters need to be synchronized with the compute kernel's channel iteration.

### Why Writer Reads Parameters
The writer kernel is responsible for reading mean/var/weight/bias because these parameters are consumed at channel boundaries, not per-tile. The writer knows when a new channel begins (from its N/C loop structure) and can push the parameter tiles to CBs in sync with when the compute kernel expects them. The reader, meanwhile, continuously streams input tiles without needing to know about channel boundaries.

### Conditional Compilation via Defines
The dataflow kernels use `#ifdef` to select between float32 and bfloat16 fill functions:
- `FILL_TILE_WITH_FIRST_ELEMENT`: Macro defined to `fill_tile_with_first_element<float>` (float32) or `fill_tile_with_first_element_bfloat16` (bfloat16)
- `FILL_WITH_VALUE_FLOAT` / `FILL_WITH_VALUE`: Used for eps fill; mutually exclusive based on dtype

### CB-Based Parameter Broadcast
Rather than using hardware broadcast instructions (bcast_rows, bcast_cols), the batch_norm operation broadcasts per-channel scalars by filling an entire tile with the same value in L1 (via `FILL_TILE_WITH_FIRST_ELEMENT`). This means the compute kernel can use standard element-wise binary ops (`sub_tiles`, `mul_tiles`, `add_tiles`) instead of broadcast variants. This is a design choice that simplifies the compute kernel at the cost of filling tiles in the writer.

### FP32 Precision Handling
When `fp32_dest_acc_en = true`:
1. The SFPU kernel variant is selected (explicit DST register management)
2. `UnpackToDestMode::UnpackToDestFp32` is set for CBs c_0, c_1, c_3, c_4, c_5, c_6, c_7, c_8
3. Note: c_2 (output) is NOT in the UnpackToDestMode list because it is the final output CB -- it is packed to, not unpacked from

### moreh_common.hpp Helpers Used
From `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`:
- `pack_tile_with_dt(dst_idx, cb)`: Calls `pack_reconfig_data_format(cb)` when `FP32_DEST_ACC_EN` is defined, then `pack_tile()`
- `add_tiles_init_with_dt(cb0, cb1)`: Calls `reconfig_data_format(cb0, cb1)` when `FP32_DEST_ACC_EN`, then `add_tiles_init()`
- `sub_tiles_init_with_dt(cb0, cb1)`: Same pattern for sub
- `mul_tiles_init_with_dt(cb0, cb1)`: Same pattern for mul

These `_with_dt` wrappers are essential when mixing CBs of different data formats under FP32 dest accumulation.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the difference between the FPU kernel path and the SFPU kernel path in compute kernels? When is fp32_dest_acc_en used?"
   **Reason**: The program factory selects between two kernel files based on `fp32_dest_acc_en`. Needed to understand why two variants exist.
   **Key Findings**: FPU path uses matrix engine with SRCA/SRCB register unpacking. SFPU path uses vector engine with explicit DST register management. FP32 dest acc changes DST capacity from 16 tiles (16-bit) to 8 tiles (32-bit). SFPU variant required for FP32 to avoid truncation through source registers.

2. **Query**: "What does binary_dest_reuse_tiles do? What are ELWMUL and DEST_TO_SRCA? How does it differ from mul_tiles?"
   **Reason**: The FPU compute kernel uses this specialized API to chain subtraction and multiplication without intermediate packing.
   **Key Findings**: `binary_dest_reuse_tiles` takes one operand from DST (already computed) and one from a CB. `DEST_TO_SRCA` moves DST[idst] to SRCA, unpacks CB tile to SRCB, computes SRCA op SRCB. This eliminates a pack-to-CB + unpack-from-CB round trip vs using mul_tiles.

3. **Query**: "How does UnpackToDestMode::UnpackToDestFp32 work? Why per circular buffer?"
   **Reason**: The program factory sets this per-CB. Needed to understand the precision pipeline.
   **Key Findings**: While `fp32_dest_acc_en` enables 32-bit DST capacity, `UnpackToDestFp32` tells the unpacker to preserve full FP32 when moving from a specific CB to DST. Without it, data may be truncated to TF32. Applied per-CB because not all CBs need FP32 unpacking.

4. **Query**: "How does the tile_regs_acquire/commit/wait/release pattern work?"
   **Reason**: Every compute step uses this pattern. Needed to understand the pipeline between math and pack stages.
   **Key Findings**: acquire() gives DST to math core (zeroes DST), commit() releases to packer, wait() blocks packer until math finishes, release() frees for next acquire. This enables math/pack overlap.

5. **Query**: "How does split_work_to_cores work? What is core_group_1 vs core_group_2?"
   **Reason**: The program factory uses this for core distribution.
   **Key Findings**: Returns two core groups: group_1 gets ceil(tiles/cores) tiles, group_2 gets floor(tiles/cores). When tiles divide evenly, group_2 is empty. Unused cores receive zero-args.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Compute kernels include this header for `_with_dt` helper wrappers
   **Key Information**: Provides `pack_tile_with_dt`, `add_tiles_init_with_dt`, `mul_tiles_init_with_dt`, `sub_tiles_init_with_dt` that handle data format reconfiguration under FP32_DEST_ACC_EN

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Writer kernel uses `FILL_TILE_WITH_FIRST_ELEMENT` macro that resolves to functions in this file
   **Key Information**: `fill_tile_with_first_element_bfloat16` reads first uint16, packs to uint32, fills 512 positions. `fill_tile_with_first_element<float>` reads first float, fills 1024 positions. `fill_with_val_bfloat16` fills entire tile with a packed scalar.

3. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary.h` (lines 205-257)
   **Reason**: Needed exact semantics of `binary_dest_reuse_tiles` and `binary_dest_reuse_tiles_init`
   **Key Information**: Template params control binary operation type and which register DST tile goes to (SRCA or SRCB). Uses `llk_unpack_A` with `binary_reuse_dest` flag.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Program factory uses `create_cb()` helper
   **Key Information**: Wrapper around `CircularBufferConfig` that sets page size and creates the CB in one call. Returns `(cb_id, cb_handle)`.
