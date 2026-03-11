# POWER (Binary SFPU) Implementation Analysis

## Overview

The POWER operation computes element-wise exponentiation: `output = base ** exponent`, where `base` is input tensor A and `exponent` is input tensor B. It is classified as a **binary SFPU operation** because it takes two tensor inputs and executes on the SFPU (Scalar FPU / vector unit) rather than the matrix FPU.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

The implementation uses two distinct algorithms depending on the data type:
- **BFloat16 mode** (`is_fp32_dest_acc_en = false`): Uses `_sfpu_binary_power_21f_`, a fast polynomial approximation based on Moroz et al. 2022 ("Simple Multiple Precision Algorithms for Exponential Functions").
- **Float32 mode** (`is_fp32_dest_acc_en = true`): Uses `_sfpu_binary_power_f32_`, a higher-precision path with Newton-Raphson reciprocal and Cody-Waite + Taylor exp for sub-1-ULP accuracy.

Both compute `base^pow = 2^(pow * log2(base))` by first computing log2(base), then raising 2 to the resulting product.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | `block_size` tiles (1 for interleaved, `find_max_block_size(num_tiles_per_shard)` for sharded) |
| **Total units** | `num_tiles = physical_volume / TILE_HW` |
| **Loop structure** | Outer loop: `per_core_block_cnt` blocks; inner loop: `per_core_block_size` tiles per block |

For the **interleaved (non-sharded) case**, `block_size = 1` and `block_cnt = num_tiles_per_core`, so the compute kernel processes one tile at a time. For the **sharded case**, `block_size` is the largest power-of-2 that evenly divides `num_tiles_per_shard`.

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A (base) | Input Tensor B (exponent) |
|----------|----------------------|--------------------------|
| **Logical shape** | Arbitrary (must match B) | Arbitrary (must match A) |
| **Dimension convention** | NHWC or any | NHWC or any |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as inputs |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as configured (BFLOAT16, FLOAT32, INT32, UINT32) |

### Layout Transformations

No explicit tilize/untilize or format conversions occur within the program factory. Both inputs must already be in TILE_LAYOUT. The SFPU kernel operates directly on tiles in DST registers.

**Special POWER-specific behavior**: Unlike other binary SFPU operations that always set `UnpackToDestMode::UnpackToDestFp32`, POWER conditionally sets unpack mode based on each input's data type (lines 174-188 of program factory). If input A is FLOAT32, its unpack mode is `UnpackToDestFp32`; otherwise it uses `Default`. This matters because the SFPU algorithm dispatches to different precision paths based on `is_fp32_dest_acc_en`.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src0_buffer) | CB c_0 | `cb_reserve_back`, `noc_async_read_tile`, `cb_push_back` |
| 2 | Reader | DRAM/L1 (src1_buffer) | CB c_1 | `cb_reserve_back`, `noc_async_read_tile`, `cb_push_back` |
| 3 | Compute | CB c_0, CB c_1 | DST registers | `cb_wait_front` on c_0 and c_1 |
| 4 | Compute | DST registers | DST registers | `copy_tile` loads A to DST[i*2], B to DST[i*2+1]; `power_binary_tile(i*2, i*2+1, i*2)` computes result in-place at DST[i*2] |
| 5 | Compute | DST registers | CB c_2 | `pack_tile(i*2, cb_out0)`, `cb_pop_front` both inputs, `cb_push_back` output |
| 6 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front`, `noc_async_write_page`, `cb_pop_front` |

**Note on sharded path**: When inputs are sharded, the reader simply does `cb_reserve_back` + `cb_push_back` for the full shard (the data is already in L1 at the CB's globally-allocated address). When output is sharded, the writer just does `cb_wait_front` and the data remains in L1.

**No pre-scaling for POWER**: The POWER operation does not define `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0`, so the intermediate CBs (c_3, c_4) are not created and the pre-scaling code paths in the compute kernel are skipped. Input tiles flow directly from c_0/c_1 to the main compute loop.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (interleaved) | Capacity (sharded) | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------------------|-------------------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A (base) staging | 2 tiles | num_tiles_per_shard tiles | 1 tile | Double | Reader | Compute | Program |
| c_1 | cb_src1 | Input B (exponent) staging | 2 tiles | num_tiles_per_shard tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles | num_tiles_per_shard tiles | 1 tile | Double | Compute | Writer | Program |

**Notes**:
- CB c_3 and c_4 (interim buffers for pre-scaling) are **not created** for POWER because the operation does not define `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0`.
- For interleaved mode, capacity = `2 * max_block_size` = 2 tiles (since `max_block_size = 1` when no shard spec).
- For sharded inputs/outputs, the CB is globally allocated at the tensor's buffer address with capacity equal to the full shard.

## Pipeline Pattern Summary

**Interleaved mode**: All three CBs (c_0, c_1, c_2) have capacity = 2 tiles and block size = 1 tile, which is **double-buffered**. This allows the reader to fill one slot while compute processes the other, enabling overlap between reader and compute stages, and between compute and writer stages.

**Sharded mode**: CBs are sized to the full shard. The reader publishes all tiles at once, then compute processes them in blocks. This is effectively **single-pass** since the entire shard is available in L1.

## Index Calculations

### Reader Kernel (Interleaved Path)
- Iterates `tile_id` from `start_id` to `start_id + num_tiles`
- Uses `TensorAccessor` with `noc_async_read_tile(tile_id, accessor, l1_addr)` which maps linear tile IDs to physical DRAM bank addresses
- Each tile ID corresponds to a 32x32 element tile in row-major tile order

### Reader Kernel (Block/Width-Sharded Path)
- Iterates with nested loops: `h` over `block_height`, `w` over `block_width`
- `row_start_tile_id` advances by `num_cores_y * block_width` per height iteration (strided access across shards)
- `tile_id = row_start_tile_id + w` within each row

### Compute Kernel
- Two tiles loaded per iteration: input A at DST index `i*2`, input B at DST index `i*2+1`
- Result written to DST index `i*2` (overwrites input A position)
- Packed from DST index `i*2` to output CB

### Writer Kernel
- Linear iteration from `start_id` to `start_id + num_pages`
- Uses `TensorAccessor` with `noc_async_write_page(i, accessor, l1_addr)`

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile-by-tile reads from DRAM. Both inputs read in lockstep (one tile of A, one tile of B, barrier, push both). This serializes the two reads per tile with a `noc_async_read_barrier()` between each pair.
- **Block/Width-Sharded**: Strided access pattern where rows within a block are read sequentially but successive rows skip by `num_cores_y * block_width` tiles.
- **Height-Sharded / Both-Sharded**: No DRAM reads; data is already in L1.

### Write Pattern
- **Interleaved**: Sequential tile-by-tile writes to DRAM with `noc_async_writes_flushed()` between each tile (not a full barrier, just a flush for write ordering).
- **Sharded output**: No writes; result stays in L1 at the globally-allocated CB address.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D or 2D (determined by `operation_attributes.worker_grid`) |
| **Grid dimensions** | Depends on device and tensor size |
| **Total cores** | `all_device_cores.num_cores()` or `grid_x * grid_y` for zero-start grids |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_tiles / num_cores)` tiles, group 2 gets `floor(num_tiles / num_cores)` tiles |

The `split_work_to_cores` function divides total tiles across available cores. When tiles don't divide evenly, some cores (group 1) get one extra tile. Non-working cores (beyond what's needed) receive zero tiles and skip execution.

For sharded tensors, the core grid is determined by the shard spec rather than computed from the total tile count.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block/width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | varies | Accessor parameters for input A buffer (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | varies | Accessor parameters for input B buffer (omitted if IN1_SHARDED) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | varies | Accessor parameters for output buffer |

#### Compute Kernel

Compile-time arguments are passed via `defines`:

| Define | Value for POWER | Description |
|--------|----------------|-------------|
| `BINOP_INIT` | `power_binary_tile_init();` | Initializes SFPU programmable constants for power computation |
| `BINARY_SFPU_OP` | `power_binary_tile(i*2, i*2+1, i*2);` | The SFPU binary power operation call |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | DRAM address of input A buffer |
| 1 | src1_addr | uint32_t | DRAM address of input B buffer |
| 2 | num_tiles | uint32_t | Total tiles this core processes |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if not sharded) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if not sharded) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension (0 if not sharded) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

#### Writer Kernel (Interleaved Output)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | DRAM address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output |

#### Writer Kernel (Block/Width-Sharded to Interleaved Output)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | DRAM address of output buffer |
| 1 | block_height | uint32_t | Block height in tiles |
| 2 | block_width | uint32_t | Block width in tiles |
| 3 | unpadded_block_height | uint32_t | Actual (unpadded) block height |
| 4 | unpadded_block_width | uint32_t | Actual (unpadded) block width |
| 5 | output_width | uint32_t | Output tensor width in tiles |
| 6 | block_size | uint32_t | Total tiles per block (height * width) |
| 7 | start_id | uint32_t | Starting tile ID |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_binary_interleaved_start_id | BRISC (RISCV_0) | NOC0 | DRAM (src0, src1) | CB c_0, CB c_1 | Read tiles for both inputs |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Key Logic**: Reads both input tensors in the same kernel (not split reader). For each tile, reserves space in both c_0 and c_1, issues async reads, waits for barrier, then pushes both. Supports three modes via preprocessor defines: both interleaved, one sharded + one interleaved, or both sharded. The block/width-sharded path uses a 2D nested loop with strided tile access.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| eltwise_binary_sfpu_kernel | Tensix compute (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | SFPU power computation |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`
- **Key Logic**:
  1. Waits for `per_core_block_size` tiles in both input CBs
  2. Acquires DST tile registers
  3. Copies input A tiles to DST at even indices (`i*2`) and input B tiles to odd indices (`i*2+1`) using `copy_tile` with data type switching via `copy_tile_to_dst_init_short_with_dt`
  4. For each tile pair, calls `power_binary_tile_init()` then `power_binary_tile(i*2, i*2+1, i*2)` which executes on the SFPU
  5. Packs result from DST[i*2] to output CB
  6. Releases DST registers and pops input CBs

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM (dst) | Write output tiles |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple linear writer. For each tile, waits for data in output CB, reads the L1 address, issues an async write, flushes, then pops. For sharded output, just does a single `cb_wait_front` for all pages (data stays in L1).

### SFPU Implementation

- **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h`
- **Algorithm**: `base^pow = 2^(pow * log2(base))`
  - **BFloat16 path** (`_sfpu_binary_power_21f_`):
    1. Extracts mantissa and exponent from base
    2. Computes log2 via 3rd-order polynomial approximation (coefficients from rminimax over [1,2])
    3. Multiplies by exponent, clamps to [-127, ...] to prevent overflow
    4. Computes 2^z using the Moroz et al. exp_21f algorithm with `addexp` (single-cycle SFPDIVP2 instruction)
    5. Explicit BFloat16 round-to-nearest-even conversion to avoid truncation error
  - **Float32 path** (`_sfpu_binary_power_f32_`):
    1. Range-reduces mantissa to [sqrt(2)/2, sqrt(2)]
    2. Computes ln(m) via z = (m-1)/(m+1) with Newton-Raphson reciprocal (2 iterations) and 6th-order odd-power polynomial
    3. Uses `_sfpu_exp_f32_accurate_` (Cody-Waite + Taylor) for sub-1-ULP accuracy
  - **Special cases** (both paths): 0^(negative) returns NaN; negative base with non-integer exponent returns NaN; negative base with odd integer exponent returns negative result.
  - **Init**: Sets programmable constants: `vConstFloatPrgm0 = 1.442695` (1/ln2), `vConstFloatPrgm1 = -127.0` (clamp threshold), `vConstFloatPrgm2 = NaN` (for special cases).
  - Each `calculate_sfpu_binary_pow` call processes 8 iterations (ITERATIONS=8), corresponding to the 8 sub-blocks of a 32x32 tile (each sub-block = 4 rows of 32 elements = 128 elements processed by the 32-wide SFPU vector unit in 4 passes).

## Implementation Notes

1. **POWER is the only binary SFPU op with conditional unpack mode**: Lines 174-188 of the program factory show that POWER does NOT force `UnpackToDestFp32` on all CBs. Instead, it matches the unpack mode to each input's actual data type. This is because the SFPU algorithm has separate BFloat16 and Float32 precision paths, and using FP32 accumulation when inputs are BFloat16 would change which algorithm path is selected.

2. **No pre-scaling**: Unlike some other binary SFPU ops (e.g., HYPOT which pre-scales inputs with SQUARE), POWER does not define pre-scaling macros. Both inputs pass directly through to the main compute loop.

3. **DST register layout**: Input A occupies even DST slots (0, 2, 4, ...) and input B occupies odd slots (1, 3, 5, ...). The result overwrites the even slot (input A position). This interleaving allows `per_core_block_size` tile pairs to be processed in a single DST acquire/release cycle.

4. **Reader reads both inputs**: This is a single-reader design (not split reader). Both input tensors are read by the same BRISC kernel using NOC0, with a `noc_async_read_barrier()` after each tile pair to ensure both are available before pushing to CBs.

5. **Cached program reuse**: The `override_runtime_arguments` method updates buffer addresses and tile counts without recreating the program, enabling efficient re-execution when only tensor data (not shapes/types) changes.

6. **Writer kernel reuse**: The writer kernel is shared with unary operations (`writer_unary_interleaved_start_id.cpp`), since writing output tiles is identical regardless of whether the operation is unary or binary.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary SFPU element-wise program factory work? What is the structure of element_wise_multi_core_sfpu_pgm_factory.cpp for binary operations like power?"
   **Reason**: Needed initial architectural context before diving into source code.
   **Key Findings**: Confirmed that POWER uses `ElementWiseMultiCoreSfpu` factory, identified the three-kernel structure (reader/compute/writer), learned that POWER has special unpack-to-dest mode handling distinct from other binary SFPU ops, and understood the difference from the standard `ElementWiseMultiCore` factory.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`
   **Reason**: Primary program factory under analysis.
   **Key Information**: CB setup, kernel creation, POWER-specific unpack mode logic, runtime args delegation.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Contains `set_eltwise_binary_runtime_args` which handles core distribution and all runtime argument setup.
   **Key Information**: Work splitting via `split_work_to_cores`, two-group load balancing, sharded vs interleaved argument patterns.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`
   **Reason**: Contains `get_defines_fp32` which generates the preprocessor defines for POWER.
   **Key Information**: POWER maps to `BINOP_INIT = "power_binary_tile_init();"` and `BINARY_SFPU_OP = "power_binary_tile(i*2, i*2+1, i*2);"`.

4. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h`
   **Reason**: Contains the actual SFPU implementation of the power function.
   **Key Information**: Two algorithm paths (21f approximation for BFloat16, accurate f32 path), polynomial log2 approximation, Moroz et al. exp algorithm, special case handling for negative bases and zero bases.

5. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Compute API header that maps `power_binary_tile` to the LLK function.
   **Key Information**: `power_binary_tile` calls `llk_math_eltwise_binary_sfpu_binary_pow<APPROX, DST_ACCUM_MODE>`.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binary_pow.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `power_binary_tile_init()` (macro `BINOP_INIT`) which invokes `llk_math_eltwise_binary_sfpu_binary_pow_init<APPROX>()`. This calls `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX>()` which configures SFPU registers, address modes, and counters, then invokes `sfpu_binary_pow_init<APPROX>()` to load programmable constants into `vConstFloatPrgm0/1/2`.

2. The compute kernel then calls `power_binary_tile(i*2, i*2+1, i*2)` (macro `BINARY_SFPU_OP`) which invokes `llk_math_eltwise_binary_sfpu_binary_pow<APPROX, DST_ACCUM_MODE>(i*2, i*2+1, i*2)`.

3. This calls `_llk_math_eltwise_binary_sfpu_params_<APPROX>(calculate_sfpu_binary_pow<APPROX, 8, is_fp32_dest_acc_en>, dst_index_in0, dst_index_in1, dst_index_out, VectorMode::RC)`.

4. The params function stalls until the SFPU is available (`TTI_STALLWAIT`), then iterates over 4 tile faces in RC mode. For each face, it calls `calculate_sfpu_binary_pow(dst_index_in0, dst_index_in1, dst_index_out)` and advances the DEST read/write counter by 16 rows (`TTI_SETRWC` twice with stride 8).

5. Inside `calculate_sfpu_binary_pow`, the function loops 8 times (ITERATIONS=8), reading base and exponent from `dst_reg[]` at the appropriate offsets, calling `_sfpu_binary_power_<is_fp32_dest_acc_en>(base, pow)`, storing the result, and incrementing `dst_reg++`.

6. `_sfpu_binary_power_<false>` dispatches to `_sfpu_binary_power_21f_` (BFloat16 path); `_sfpu_binary_power_<true>` dispatches to `_sfpu_binary_power_f32_` (Float32 path).

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h
// (Wormhole B0 version is identical)

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Implementation notes, see the original file for more details
template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_binary_power_21f_(sfpi::vFloat base, sfpi::vFloat pow) {
    // Step 1: Compute log2(base)
    sfpi::vFloat absbase = setsgn(base, 0);       // SFPSETSGN: clear sign bit (absolute value)
    sfpi::vFloat x = sfpi::setexp(absbase, 127);   // SFPSETEXP: force exponent to 127 (range [1,2))

    // 3rd order polynomial approx (rminimax coefficients over [1,2])
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Convert exponent to float
    sfpi::vInt exp = sfpi::exexp(base);             // SFPEXEXP: extract biased exponent - 127
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); } // negate via ones complement+1, set sign bit
    v_endif;
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(exp, 0); // SFPCAST: sign-magnitude int32 -> fp32

    // De-normalize to original range
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;           // 1/ln(2) = 1.442695
    sfpi::vFloat log2_result = exp_f32 + series_result * vConst1Ln2;

    // Step 2: Compute base**pow = 2**(pow * log2(base))
    sfpi::vFloat z_f32 = pow * log2_result;
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;         // -127.0
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }            // clamp to prevent overflow
    v_endif;

    // Implementation notes, see the original file for more details
    z_f32 = addexp(z_f32, 23);  // SFPDIVP2: multiply by 2^23 (single-cycle exponent add)
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias);

    sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));         // SFPEXEXP: extract exponent part
    sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // SFPEXMAN: extract 9-bit mantissa

    // Horner form polynomial for 2^frac
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + zif, 0);

    d2 = d1 * d2;
    zif = _float_to_int32_positive_(d2 * d3);

    // Restore exponent
    zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

    // Post-processing: handle special cases
    sfpi::vInt pow_int = sfpi::float_to_int16(pow, 0); // SFPSTOCHRND: fp32 -> int16 (round to nearest)
    sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);

    // 0^(negative) => NaN
    v_if((absbase == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;  // NaN
    }
    v_endif;

    v_if(base < 0.0f) {  // negative base
        y = setsgn(y, pow_int << 31); // SFPSHFT: shift LSB of pow to sign position
        v_if(pow_rounded != pow) {     // non-integer power => NaN
            y = sfpi::vConstFloatPrgm2;
        }
        v_endif;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        // Explicit bf16 round-to-nearest-even to avoid SFPSTORE truncation error
        y = reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFPSTOCHRND: fp32 -> bf16
    }

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_binary_power_f32_(sfpi::vFloat base, sfpi::vFloat pow) {
    // Step 1: Compute log2(base) using improved log with Newton-Raphson reciprocal
    sfpi::vFloat abs_base = sfpi::abs(base);
    sfpi::vFloat m = sfpi::setexp(abs_base, 127);      // SFPSETEXP: normalize mantissa to [1,2)
    sfpi::vInt exp = sfpi::exexp(abs_base);             // SFPEXEXP: extract exponent

    // Range reduction to [sqrt(2)/2, sqrt(2)]
    constexpr float SQRT2 = 1.4142135381698608f;
    v_if(m >= SQRT2) {
        m = m * 0.5f;
        exp = exp + 1;
    }
    v_endif;

    // Transform to z = (m - 1) / (m + 1) via Newton-Raphson reciprocal
    sfpi::vFloat m_plus_1 = m + sfpi::vConst1;
    sfpi::vFloat m_minus_1 = m - sfpi::vConst1;
    sfpi::vFloat recip = sfpi::vConst1 - 0.2426406871192851f * m_plus_1; // initial guess
    recip = recip * (2.0f - m_plus_1 * recip);  // 1st Newton-Raphson iteration
    recip = recip * (2.0f - m_plus_1 * recip);  // 2nd Newton-Raphson iteration
    sfpi::vFloat z = m_minus_1 * recip;

    // Odd-power polynomial: ln(1+z)/(1-z) = 2z * P(z^2)
    sfpi::vFloat z2 = z * z;
    sfpi::vFloat p = PolynomialEvaluator::eval(
        z2, sfpi::vConst1, 0.3333333333333333f, 0.2f, 0.14285714285714285f, 0.1111111111111111f, 0.09090909090909091f);
    sfpi::vFloat ln_m = 2.0f * (z * p);

    // Convert exponent to float (handles negative exponents via two's complement)
    sfpi::vInt sign_bit = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(exp) >> 31);
    sfpi::vInt exp_sign = sfpi::vInt(0) - sign_bit;
    sfpi::vInt exp_abs = (exp ^ exp_sign) - exp_sign;
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(sfpi::setsgn(exp_abs, exp_sign), 0);

    // log2(base) = exp + ln_m * (1/ln2)
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0; // 1/ln(2) = 1.442695
    sfpi::vFloat log2_result = exp_f32 + ln_m * vConst1Ln2;

    // Step 2: base**pow = 2**(pow * log2(base))
    sfpi::vFloat z_f32 = pow * log2_result;
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1; // -127.0
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }
    v_endif;

    // Cody-Waite + Taylor series exp for sub-1-ULP fp32 accuracy
    constexpr float LN2 = 0.693147180559945309f;
    sfpi::vFloat y = _sfpu_exp_f32_accurate_(z_f32 * LN2);

    // 0^(negative) => NaN
    v_if((abs_base == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;
    }
    v_endif;

    v_if(base < 0.0f) {
        sfpi::vInt pow_int = sfpi::float_to_int16(pow, 0);
        sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);
        y = sfpi::setsgn(y, pow_int << 31);            // odd integer power => negative result
        v_if(pow_rounded != pow) {
            y = sfpi::vConstFloatPrgm2;                 // non-integer power => NaN
        }
        v_endif;
    }
    v_endif;

    return y;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_power_(sfpi::vFloat base, sfpi::vFloat pow);

template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<false>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_21f_<false>(base, pow);
}

template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<true>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_f32_(base, pow);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_pow(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE=true (from APPROX define), ITERATIONS=8, is_fp32_dest_acc_en from DST_ACCUM_MODE
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size_sfpi = 32; // 64/SFP_DESTREG_STRIDE = 32 rows per tile in SFPI addressing
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from DEST
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from DEST

        sfpi::vFloat result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE to DEST
        sfpi::dst_reg++; // TTI_INCRWC: advance DEST pointer by SFP_DESTREG_STRIDE (4 rows)
    }
}

template <bool APPROXIMATION_MODE>
inline void sfpu_binary_pow_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;                           // SFPLOADI to CREG: 1/ln(2)
    sfpi::vConstFloatPrgm1 = -127.0f;                             // SFPLOADI to CREG: clamp threshold
    sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN(); // SFPLOADI to CREG: NaN sentinel
}
```

### SFPU Instructions Used

| Instruction | SFPI Intrinsic | Description |
|-------------|---------------|-------------|
| **SFPSETSGN** | `setsgn(v, sgn)` | Sets or clears the sign bit of a floating-point value. Used to compute absolute value (`setsgn(base, 0)`) and to apply sign from the exponent's parity (`setsgn(y, pow_int << 31)`). |
| **SFPSETEXP** | `setexp(v, exp)` | Replaces the exponent field of a float with an immediate or register value. Used to normalize mantissa to [1,2) range (`setexp(abs_base, 127)`) and to reconstruct the final result exponent (`setexp(frac, 127U + zii)`). |
| **SFPEXEXP** | `exexp(v)` | Extracts the biased exponent minus 127 from a float as a signed integer. Core operation for decomposing the base into mantissa and exponent parts. |
| **SFPEXMAN** | `exman8(v)`, `exman9(v)` | Extracts mantissa bits from a float. `exman8` extracts 8-bit mantissa (used in `_float_to_int32_positive_`); `exman9` extracts 9-bit mantissa (used in the exp_21f reconstruction). |
| **SFPDIVP2** | `addexp(v, imm)` | Adds an immediate value to the exponent field, equivalent to multiplying by 2^imm. Used as `addexp(z_f32, 23)` to multiply by 2^23 in a single cycle (replaces SFPLOADI + MAD). |
| **SFPCAST** | `int32_to_float(v, 0)` | Converts a sign-magnitude 32-bit integer to IEEE 754 float. Used to convert extracted exponents and integer power values back to float for arithmetic. |
| **SFPSTOCHRND** | `float_to_int16(v, 0)` | Converts fp32 to a bounded 16-bit signed integer with round-to-nearest. Used to test whether the exponent is an integer (by round-tripping through int16 and comparing). |
| **SFPSTOCHRND** | `float_to_fp16b(v, 0)` | Converts fp32 to bf16 using round-to-nearest-even. Used in the BFloat16 path to avoid truncation errors from SFPSTORE. |
| **SFPSHFT** | `shft(v, shift)` | Bitwise shift (logical or arithmetic). Used in `_float_to_int32_positive_` to shift mantissa by `(23 - exponent)` positions. Also underlies the `pow_int << 31` sign extraction. |
| **SFPLOAD** | `dst_reg[index]` (read) | Loads a 32-bit value from a DEST register row into an SFPU local register (LReg). Each iteration reads base and exponent from their respective DEST tile offsets. |
| **SFPSTORE** | `dst_reg[index] = v` (write) | Stores a 32-bit value from an SFPU local register back to a DEST register row. Writes the computed power result. |
| **SFPLOADI** | `vConstFloatPrgm0 = ...` | Loads an immediate 32-bit value into a programmable constant register (CREG). Used during init to set 1/ln(2), -127.0, and NaN. |
| **SFPMAD** | `a * b + c` (implicit) | Fused multiply-add on LRegs. The core arithmetic workhorse; all polynomial evaluations, Newton-Raphson iterations, and scaling operations compile to sequences of SFPMAD instructions. |
| **SFPABS** | `abs(v)` | Computes absolute value (used in f32 path as `sfpi::abs(base)`). |
| **SFPXCONDI/SFPXCONDB** | `v_if`/`v_endif` | Conditional execution control. Sets and manages the condition code (CC) register to predicate subsequent SFPU instructions. Used extensively for special-case handling (negative base, zero base, overflow clamping). |
| **TTI_SETRWC** | (in params dispatch) | Sets the read/write counter for DEST addressing. Used between face iterations to advance the DEST pointer by 8 rows per call (16 rows total per face = half a 32x32 tile face). |
| **TTI_STALLWAIT** | (in start function) | Stalls until the SFPU is available for new work. Called at the beginning of each binary SFPU operation to synchronize with prior math operations. |
| **TTI_INCRWC** | `dst_reg++` | Increments the DEST write counter by `SFP_DESTREG_STRIDE` (2 rows), advancing to the next 32-element vector within a tile face. |

### SFPU Register Usage

**DEST Registers (DST)**:
- **Input A (base)**: Loaded at DEST offset `dst_index_in0 * 32` (even tile slots: 0, 2, 4, ...). Each tile occupies 32 SFPI rows (= 64 physical DEST rows at stride 2).
- **Input B (exponent)**: Loaded at DEST offset `dst_index_in1 * 32` (odd tile slots: 1, 3, 5, ...).
- **Output**: Written back to DEST offset `dst_index_out * 32`, which equals `dst_index_in0 * 32` (overwrites the base tile in-place).
- The `dst_reg++` after each iteration advances by `SFP_DESTREG_STRIDE` rows, processing 4 rows of 32 elements per SFPU vector operation. Over 8 iterations, this covers one 16x16 face (half a tile face with 128 elements).

**SFPU Local Registers (LRegs)**:
- LReg0-LReg7: Used as temporary storage for intermediate values during the power computation. The compiler allocates these automatically from the SFPI C++ variables (`absbase`, `x`, `series_result`, `exp_f32`, `z_f32`, `y`, etc.). The 21f path uses fewer LRegs than the f32 path due to its simpler polynomial.

**Programmable Constant Registers (CREGs)**:
- `vConstFloatPrgm0` (CREG_IDX_PRGM1): `1.442695f` = 1/ln(2), used in both log2 computation paths.
- `vConstFloatPrgm1` (CREG_IDX_PRGM2): `-127.0f`, the underflow clamp threshold for 2^z computation.
- `vConstFloatPrgm2` (CREG_IDX_PRGM3): `NaN` (quiet), used as the return value for undefined cases (0^negative, negative base with non-integer exponent).

**Built-in Constants**:
- `vConst0`: 0.0f (used for zero comparison in f32 path).
- `vConst1`: 1.0f (used in Newton-Raphson and polynomial evaluation).

### Address Mode Configuration

The binary SFPU init function `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` configures ADDR_MOD_7 for the power operation. Since the `SfpuType` is `unused` (not one of the special-cased types like `mul_int32`, `max`, `min`, etc.), only ADDR_MOD_7 is configured:

**ADDR_MOD_7** (configured for all binary SFPU ops):
| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment for SRC A addressing |
| `srcb.incr` | 0 | No auto-increment for SRC B addressing |
| `dest.incr` | 0 | No auto-increment for DEST addressing |

ADDR_MOD_6 is NOT configured for POWER (only used for `mul_int32`, `mul_uint16`, `max`, `min`, and their int32/uint32 variants which need `dest.incr = 2`).

The DEST register advancement is handled explicitly by `TTI_SETRWC` calls in the params dispatch function (advancing by 8 rows per call, twice per face) and by `dst_reg++` inside the SFPU kernel (advancing by `SFP_DESTREG_STRIDE` per iteration).

**Wormhole vs Blackhole differences**: The `_llk_math_eltwise_binary_sfpu_init_` function is identical across both architectures for address mode configuration. However, the `_llk_math_eltwise_binary_sfpu_start_` function has a minor difference: Wormhole calls `math::set_addr_mod_base()` at start and `math::clear_addr_mod_base()` at done (plus an extra `TTI_STALLWAIT` for SFPU completion), while Blackhole omits these calls. The SFPU kernel code itself (`ckernel_sfpu_binary_pow.h`) is identical across both architectures.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the POWER binary SFPU operation work? What defines does it set? Where is the ckernel SFPU power implementation?"
   **Reason**: Needed to locate the SFPU kernel files and understand the define/macro system for POWER.
   **Key Findings**: Confirmed POWER uses `BINOP_INIT`/`BINARY_SFPU_OP` defines, identified `ckernel_sfpu_binary_pow.h` as the core implementation, learned about the two precision paths (21f vs f32).

2. **Query**: "What do the SFPU instructions setsgn, setexp, exexp, exman8, exman9, addexp (SFPDIVP2), shft, int32_to_float, float_to_int16, float_to_fp16b do?"
   **Reason**: Needed precise instruction-level documentation for each SFPU intrinsic used in the power kernel.
   **Key Findings**: Confirmed SFPSETSGN manipulates sign bits, SFPSETEXP replaces exponent fields, SFPEXEXP extracts biased exponent minus 127, SFPDIVP2 adds immediate to exponent (single-cycle multiply by power of 2), SFPSTOCHRND handles both float-to-int16 and float-to-bf16 conversions.

3. **Query**: "How do dst_reg[], v_if/v_endif, vConstFloatPrgm0/1/2, and reinterpret work in SFPI?"
   **Reason**: Needed to understand the SFPI programming model abstractions that the power kernel uses extensively.
   **Key Findings**: `dst_reg[]` maps to SFPLOAD/SFPSTORE, `dst_reg++` compiles to TTI_INCRWC with SFP_DESTREG_STRIDE, `v_if/v_endif` uses SFPXCONDI/SFPXCONDB for lane-wise predication, programmable constants map to CREG indices loaded via SFPLOADI.

### Confluence References
No Confluence references were needed for this analysis. The DeepWiki documentation for tt-isa-documentation and sfpi provided sufficient detail on all SFPU instructions used.

### Glean References
No Glean references were needed for this analysis.
