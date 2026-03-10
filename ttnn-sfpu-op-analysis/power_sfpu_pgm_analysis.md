# POWER (Binary SFPU) Implementation Analysis

## Overview

The POWER binary SFPU operation computes element-wise exponentiation: `output = base ** exponent`, where both `base` and `exponent` are tensors. It is implemented as `BinaryOpType::POWER` within the `ElementWiseMultiCoreSfpu` program factory. The operation uses polynomial approximation algorithms to compute `base^pow = 2^(pow * log2(base))`, with separate code paths for bfloat16 (21-float approximation) and float32 (improved log2 + Cody-Waite exp) data types.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Work Unit Definition

The fundamental work unit is a **tile** (32x32 elements). Tiles are grouped into **blocks** for processing efficiency. For non-sharded tensors, the block size is 1 tile (i.e., `max_block_size = 1`). For sharded tensors, the block size is the largest power of 2 that evenly divides `num_tiles_per_shard`. The compute kernel processes `per_core_block_cnt` blocks of `per_core_block_size` tiles each.

## Tensor Format and Layout

### Input Tensor(s)

| Property | Input A (base) | Input B (exponent) |
|---|---|---|
| Dimension Convention | NHWC (last dim is innermost) | NHWC (last dim is innermost) |
| Tensor Layout | TILE (32x32) | TILE (32x32) |
| Memory Layout | Interleaved or Sharded | Interleaved or Sharded |
| Buffer Type | DRAM or L1 | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32 | BFLOAT16, FLOAT32 |

### Output Tensor(s)

| Property | Output |
|---|---|
| Dimension Convention | NHWC (last dim is innermost) |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded |
| Buffer Type | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32, INT32, UINT32 |

### Layout Transformations

No explicit tilize/untilize or format conversions are performed within the program factory. Both inputs and the output are expected to be in tiled layout. If the output data format is Float32, Int32, or UInt32, the `fp32_dest_acc_en` flag is set, which selects the float32 SFPU kernel path and enables 32-bit destination accumulation.

**Special POWER behavior for UnpackToDestMode**: Unlike other binary SFPU ops (which always use `UnpackToDestFp32`), POWER only sets `UnpackToDestFp32` when the corresponding input dtype is `FLOAT32`. For bfloat16 inputs, it uses `UnpackToDestMode::Default`. This is because the bf16 kernel path (`_sfpu_binary_power_21f_`) is specifically designed for that precision and explicitly rounds back to bf16 at the end.

## Data Flow Pattern

1. **Reader kernel** reads tiles for both input tensors from DRAM/L1 into circular buffers `cb_in0` (CB c_0) and `cb_in1` (CB c_1). For sharded inputs, data is already in L1, so the reader simply marks tiles as available.
2. **Compute kernel** waits for tiles in `cb_in0` and `cb_in1` (aliased as `cb_inp0` and `cb_inp1` since no pre-scaling is active for POWER).
3. For each tile in the block:
   - Tiles from `cb_inp0` (base) are copied to DST at even indices (`i*2`).
   - Tiles from `cb_inp1` (exponent) are copied to DST at odd indices (`i*2+1`).
   - The SFPU `BINOP_INIT` macro calls `power_binary_tile_init()` to set programmable constants.
   - The `BINARY_SFPU_OP` macro calls `power_binary_tile(i*2, i*2+1, i*2)`, which reads base from DST[i*2] and exponent from DST[i*2+1], computes the power, and writes the result back to DST[i*2].
   - The result at DST[i*2] is packed into `cb_out0`.
4. **Writer kernel** reads tiles from `cb_out0` and writes them back to DRAM/L1. For sharded outputs, data stays in L1.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|---|
| c_0 | cb_src0 | Input A (base) | 2 * max_block_size (non-sharded) or num_tiles_per_shard (sharded) | 1 (non-sharded reader pushes 1 at a time) | Double-buffered (non-sharded) | Reader | Compute | Per-block |
| c_1 | cb_src1 | Input B (exponent) | 2 * max_block_size (non-sharded) or num_tiles_per_shard (sharded) | 1 (non-sharded reader pushes 1 at a time) | Double-buffered (non-sharded) | Reader | Compute | Per-block |
| c_2 | cb_out0 | Output | 2 * max_block_size (non-sharded) or num_tiles_per_shard (sharded/block-width-sharded) | per_core_block_size | Double-buffered (non-sharded) | Compute | Writer | Per-block |

**Note on interim CBs**: The program factory conditionally creates CB c_3 and c_4 for pre-scaling operations (when `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines are present). For POWER, these pre-scaling defines are not set by default, so `cb_inp0` aliases to `cb_in0` (c_0) and `cb_inp1` aliases to `cb_in1` (c_1). However, if the user provides an `input_tensor_a_activation` (e.g., applying abs to the base before power), then CB c_3 would be created.

## Pipeline Pattern Summary

For non-sharded interleaved tensors with `max_block_size = 1`:
- **cb_src0 / cb_src1**: Capacity = 2 tiles, block size = 1 tile. This is **double-buffered**, allowing the reader to write the next tile while compute processes the current one.
- **cb_out0**: Capacity = 2 tiles, block size = 1 tile. **Double-buffered**, allowing compute to produce the next result while the writer drains the current one.

For sharded tensors, all tiles for the shard are available at once (single-buffered with full shard capacity).

## Index Calculations

The reader kernel uses `TensorAccessor` for DRAM address translation. For non-sharded interleaved layout, `noc_async_read_tile(tile_id, accessor, l1_addr)` maps a linear tile ID to the physical DRAM bank and offset. For block/width-sharded layouts, the reader uses a nested loop over `block_height` and `block_width` with stride `num_cores_y * block_width` to compute tile IDs.

In the compute kernel, tiles are placed in DST using interleaved indexing: base tile `i` goes to DST slot `i*2`, exponent tile `i` goes to DST slot `i*2+1`. The SFPU output overwrites DST slot `i*2` (the base position), which is then packed to the output CB.

## Memory Access Patterns

### Read Pattern

For non-sharded interleaved tensors, tiles are read sequentially from a contiguous tile ID range (`start_id` to `start_id + num_tiles`). Each tile read issues a NoC async read followed by a barrier, resulting in a tile-at-a-time sequential pattern. Both inputs are read in lockstep (same tile ID for both src0 and src1).

For block/width-sharded tensors, tiles are accessed in row-major order within the shard, with stride-based addressing across rows (`row_start_tile_id += num_cores_y * block_width`).

### Write Pattern

For non-sharded interleaved output, tiles are written sequentially one at a time from `start_id` to `start_id + num_pages`, using `noc_async_write_page`. For sharded output, the writer simply waits on the CB (data is already in L1 at the correct location).

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Rectangular grid from `operation_attributes.worker_grid` |
| Work Splitting | `tt::tt_metal::split_work_to_cores` divides total tiles across cores |
| Core Groups | Group 1: cores with `ceil(num_tiles / num_cores)` tiles; Group 2: cores with `floor(num_tiles / num_cores)` tiles |
| Remainder Handling | Extra tiles distributed to group 1 cores; unused cores receive zero-length work |
| Traversal Order | Row-major for non-sharded; follows shard orientation for sharded |
| Zero-start Grid Optimization | If the grid is a single rectangle starting at (0,0), faster work-splitting algorithms are used |

For sharded tensors, each core processes exactly `num_tiles_per_shard` tiles from its local shard, and all cores are in group 1 (uniform work distribution).

## Arguments

### Compile-Time Arguments

**Reader Kernel** (`reader_binary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | block_or_width_sharded | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | src0_args (TensorAccessorArgs) | varies | Tensor accessor parameters for input A (only if not IN0_SHARDED) |
| varies | src1_args (TensorAccessorArgs) | varies | Tensor accessor parameters for input B (only if not IN1_SHARDED) |

**Writer Kernel** (`writer_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_out | uint32_t | Output CB index (c_2) |
| 1+ | dst_args (TensorAccessorArgs) | varies | Tensor accessor parameters for output buffer |

**Compute Kernel** (`eltwise_binary_sfpu_kernel.cpp`):

No indexed compile-time args. Configuration is via preprocessor defines:

| Define | Value for POWER | Description |
|---|---|---|
| `BINOP_INIT` | `power_binary_tile_init();` | Initializes SFPU programmable constants for power |
| `BINARY_SFPU_OP` | `power_binary_tile(i*2, i*2+1, i*2);` | Dispatches the binary power SFPU operation |
| `APPROX` | from ComputeConfig | Whether approximate mode is enabled |
| `DST_ACCUM_MODE` | from ComputeConfig | Destination accumulation mode (fp32 or default) |

### Runtime Arguments

**Reader Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | DRAM address of input A buffer |
| 1 | src1_addr | uint32_t | DRAM address of input B buffer |
| 2 | num_tiles | uint32_t | Total tiles this core must read |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if not sharded) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if not sharded) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension (0 if not sharded) |

**Compute Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of blocks this core processes |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

**Writer Kernel** (non-sharded interleaved):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | DRAM address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output |

## Kernel Implementations

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Key Logic**: Reads tiles for both inputs in lockstep. For non-sharded inputs, reads one tile at a time using `noc_async_read_tile` with a TensorAccessor, followed by `noc_async_read_barrier`. For sharded inputs, simply marks the entire shard as available via `cb_reserve_back` / `cb_push_back`. Supports block/width-sharded access pattern with nested height/width loops.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Writes one tile at a time using `noc_async_write_page`. For sharded output, simply calls `cb_wait_front` to ensure compute is done. Uses `noc_async_writes_flushed` after each tile write for flow control.

### Compute Kernel

#### Compute Kernel File

`ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"  // includes power_binary_tile and power_binary_tile_init
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/remainder_int32.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/xlogy.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/binary_comp.h"

// PRE_SCALE is true if either input has a pre-scaling activation defined.
// For POWER, this is false by default (no pre-scaling).
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // number of blocks to process
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);   // tiles per block

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input A (base) circular buffer
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input B (exponent) circular buffer

// For POWER, SFPU_OP_INIT_PRE_IN0_0 is not defined, so cb_inp0 = cb_in0
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;  // interim CB for pre-scaled input A
#else
    constexpr auto cb_inp0 = cb_in0;             // no pre-scaling, use input directly
#endif

// For POWER, SFPU_OP_INIT_PRE_IN1_0 is not defined, so cb_inp1 = cb_in1
#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;  // interim CB for pre-scaled input B
#else
    constexpr auto cb_inp1 = cb_in1;             // no pre-scaling, use input directly
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;  // output circular buffer

    unary_op_init_common(cb_in0, cb_out0);  // initialize unpack/pack pipeline for these CBs

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));  // not used for POWER
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

// PRE_SCALE section: skipped for POWER since no pre-scaling defines are set.
// If input_tensor_a_activation were set (e.g., abs), the SFPU_OP_INIT_PRE_IN0_0
// block would copy tiles from cb_in0 to DST, apply the activation via SFPU,
// then pack results to cb_inp0 (c_3).
#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);
#endif

#ifdef SFPU_OP_INIT_PRE_IN0_0
        // ... pre-scaling for input A (not active for POWER) ...
        cb_wait_front(cb_in0, per_core_block_size);
        cb_reserve_back(cb_inp0, per_core_block_size);
        tile_regs_acquire();
        SFPU_OP_INIT_PRE_IN0_0
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in0, i, i);
            SFPU_OP_FUNC_PRE_IN0_0
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_inp0);
        }
        tile_regs_release();
        cb_pop_front(cb_in0, per_core_block_size);
        cb_push_back(cb_inp0, per_core_block_size);
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
        // ... pre-scaling for input B (not active for POWER) ...
        cb_wait_front(cb_in1, per_core_block_size);
        cb_reserve_back(cb_inp1, per_core_block_size);
        tile_regs_acquire();
        SFPU_OP_INIT_PRE_IN1_0
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in1, i, i);
            SFPU_OP_FUNC_PRE_IN1_0
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_inp1);
        }
        tile_regs_release();
        cb_pop_front(cb_in1, per_core_block_size);
        cb_push_back(cb_inp1, per_core_block_size);
#endif

        // Wait for both inputs to be ready
        cb_wait_front(cb_inp0, per_core_block_size);  // blocks until base tiles are available
        cb_wait_front(cb_inp1, per_core_block_size);  // blocks until exponent tiles are available
        cb_reserve_back(cb_out0, per_core_block_size); // reserve space in output CB

        tile_regs_acquire();  // acquire DST register file for writing
        tile_regs_wait();     // wait for DST to be ready (no pending operations)

        // Copy base tiles from cb_inp0 to even DST slots (i*2)
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);  // init unpack for cb_inp0 data type
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);  // copy base tile i from CB to DST[i*2]
        }

        // Copy exponent tiles from cb_inp1 to odd DST slots (i*2+1)
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);  // init unpack for cb_inp1 data type
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1);  // copy exponent tile i from CB to DST[i*2+1]

            // For POWER, BINOP_INIT is defined as: power_binary_tile_init();
            // This sets programmable constants: vConstFloatPrgm0=1/ln2, Prgm1=-127, Prgm2=NaN
#ifdef BINOP_INIT
            BINOP_INIT
#endif
            // ... other init macros for other op types (not active for POWER) ...
#ifdef ADD_INT_INIT
            ADD_INT_INIT
#endif
#ifdef SUB_INT_INIT
            SUB_INT_INIT
#endif
#ifdef MUL_INT_INIT
            MUL_INT_INIT
#endif
#ifdef LT_INT32_INIT
            LT_INT32_INIT
#endif
#ifdef GT_INT32_INIT
            GT_INT32_INIT
#endif
#ifdef GE_INT32_INIT
            GE_INT32_INIT
#endif
#ifdef LE_INT32_INIT
            LE_INT32_INIT
#endif
#ifdef BITWISE_INIT
            BITWISE_INIT
#endif
#ifdef BITWISE_UINT16_INIT
            BITWISE_UINT16_INIT
#endif
#ifdef SHIFT_INIT
            SHIFT_INIT
#endif

            // For POWER, BINARY_SFPU_OP is defined as:
            // power_binary_tile(i*2, i*2+1, i*2);
            // This calls the SFPU to compute base^exponent:
            //   - reads base from DST[i*2]
            //   - reads exponent from DST[i*2+1]
            //   - writes result to DST[i*2]
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP
#endif
#ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            // Pack result from DST[i*2] to output circular buffer
            pack_tile(i * 2, cb_out0);
        }

        tile_regs_commit();   // signal that DST writes are complete
        tile_regs_release();  // release DST registers for next block

        cb_pop_front(cb_inp0, per_core_block_size);   // free consumed base tiles
        cb_pop_front(cb_inp1, per_core_block_size);   // free consumed exponent tiles
        cb_push_back(cb_out0, per_core_block_size);   // publish output tiles to writer
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File

`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h`
(Wormhole B0 counterpart: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h` -- identical code)

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"  // provides _float_to_int32_positive_
#include "ckernel_sfpu_exp.h"          // provides _sfpu_exp_f32_accurate_ for float32 path
#include "sfpu/ckernel_sfpu_polyval.h" // provides PolynomialEvaluator for float32 path
#include "sfpi.h"                       // SFPI programming interface: vFloat, vInt, dst_reg, etc.

using namespace sfpi;

namespace ckernel {
namespace sfpu {

/**
 * BFloat16 path: _sfpu_binary_power_21f_
 * Implements base^pow = 2^(pow * log2(base)) using the "exp_21f" algorithm
 * from Moroz et al. 2022. This is a compact ~21-float approximation suitable
 * for bfloat16 precision.
 */
template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_binary_power_21f_(sfpi::vFloat base, sfpi::vFloat pow) {
    // Algorithm: base^pow = 2^(pow * log2(base))
    // Step 1: Compute log2(base)

    // Take absolute value of base for log computation (handle sign separately later)
    sfpi::vFloat absbase = setsgn(base, 0);       // clear sign bit -> |base|
    sfpi::vFloat x = sfpi::setexp(absbase, 127);   // normalize to range [1,2) by setting exponent to bias (127)

    // 3rd-order polynomial approximation for ln(x) over [1,2], coefficients from rminimax
    // Evaluates: p(x) = x*(x*(x*0x2.44734p-4 - 0xd.e712ap-4) + 0x2.4f5388p+0) - 0x1.952992p+0
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Extract the biased exponent of the original base as an integer
    sfpi::vInt exp = sfpi::exexp(base);  // exexp extracts biased exponent field
    // Handle negative exponents: compute absolute value with sign
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }  // two's complement negation, set sign bit
    v_endif;
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(exp, 0);  // convert exponent integer to float

    // De-normalize: log2(base) = exponent + ln(mantissa) * (1/ln(2))
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;           // 1/ln(2) = 1.442695f (set during init)
    sfpi::vFloat log2_result = exp_f32 + series_result * vConst1Ln2;

    // Step 2: Compute 2^(pow * log2(base))
    sfpi::vFloat z_f32 = pow * log2_result;
    // Clamp to -127 to prevent overflow in intermediate computation
    // (e.g., 0^(+inf) or N^(-inf) should approach 0, not overflow)
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;  // -127.0f (set during init)
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }
    v_endif;

    // exp_21f algorithm from Moroz et al.:
    // z = (bias + z_f32) * 2^23, reinterpreted as integer, gives approximate 2^z_f32
    // Optimized: use addexp to multiply by 2^23 (single SFPDIVP2 instruction)
    z_f32 = addexp(z_f32, 23);  // z_f32 *= 2^23 (SFPDIVP2 with immediate +23)
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);  // IEEE 754 representation of 1.0f
    sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias);  // convert to int (the paper's formula)

    // Split z into exponent and mantissa parts for Horner polynomial evaluation
    sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));         // extract exponent bits of z
    sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // extract mantissa (9-bit) of z

    // Horner-form polynomial to refine the 2^x approximation
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif, 0);  // constant + mantissa fraction
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + zif, 0);    // constant + mantissa fraction

    d2 = d1 * d2;
    zif = _float_to_int32_positive_(d2 * d3);  // evaluate polynomial

    // Restore the exponent: combine polynomial result mantissa with computed exponent
    zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);  // final 2^(pow*log2(base)) result

    // Post-processing: handle special cases
    sfpi::vInt pow_int =
        sfpi::float_to_int16(pow, 0);  // truncate exponent to int16 for sign/parity checks
    sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);  // round-trip to detect non-integer

    // Special case: 0^(negative) = NaN (division by zero)
    v_if((absbase == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;  // NaN (set during init)
    }
    v_endif;

    // Handle negative base
    v_if(base < 0.0f) {
        // Set sign based on parity of exponent: odd power -> negative, even -> positive
        // Shift LSB of pow_int to sign bit position (bit 31)
        y = setsgn(y, pow_int << 31);

        // If exponent is not an integer, result is complex -> NaN
        v_if(pow_rounded != pow) {
            y = sfpi::vConstFloatPrgm2;  // NaN for non-integer powers of negative base
        }
        v_endif;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        // When DST is bfloat16, explicitly round to bf16 using round-to-nearest-even
        // to avoid truncation errors (e.g., 9^2 = 80.8 truncated to 80.5 instead of 81)
        y = reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

/**
 * Float32 path: _sfpu_binary_power_f32_
 * Uses improved log2 with range reduction to [sqrt(2)/2, sqrt(2)] and
 * Newton-Raphson reciprocal, followed by Cody-Waite + Taylor exp for <1 ULP accuracy.
 */
sfpi_inline sfpi::vFloat _sfpu_binary_power_f32_(sfpi::vFloat base, sfpi::vFloat pow) {
    // Step 1: Compute log2(base) with higher accuracy for float32

    sfpi::vFloat abs_base = sfpi::abs(base);
    sfpi::vFloat m = sfpi::setexp(abs_base, 127);  // normalize mantissa to [1,2)
    sfpi::vInt exp = sfpi::exexp(abs_base);          // extract biased exponent

    // Range reduction: bring m into [sqrt(2)/2, sqrt(2)] for better polynomial accuracy
    constexpr float SQRT2 = 1.4142135381698608f;
    v_if(m >= SQRT2) {
        m = m * 0.5f;    // divide by 2 (SFPMUL)
        exp = exp + 1;    // compensate exponent
    }
    v_endif;

    // Compute z = (m-1)/(m+1) using Newton-Raphson reciprocal for 1/(m+1)
    sfpi::vFloat m_plus_1 = m + sfpi::vConst1;
    sfpi::vFloat m_minus_1 = m - sfpi::vConst1;
    // Linear initial guess for 1/(m+1), valid over [1.707, 2.414]
    sfpi::vFloat recip = sfpi::vConst1 - 0.2426406871192851f * m_plus_1;
    recip = recip * (2.0f - m_plus_1 * recip);  // 1st Newton-Raphson refinement
    recip = recip * (2.0f - m_plus_1 * recip);  // 2nd Newton-Raphson for float32 accuracy
    sfpi::vFloat z = m_minus_1 * recip;           // z = (m-1)/(m+1)

    // Polynomial approximation for atanh(z) = z + z^3/3 + z^5/5 + ...
    // ln(m) = 2*atanh(z) since m = (1+z)/(1-z)
    sfpi::vFloat z2 = z * z;
    sfpi::vFloat p = PolynomialEvaluator::eval(
        z2, sfpi::vConst1, 0.3333333333333333f, 0.2f, 0.14285714285714285f,
        0.1111111111111111f, 0.09090909090909091f);  // coefficients: 1, 1/3, 1/5, 1/7, 1/9, 1/11
    sfpi::vFloat ln_m = 2.0f * (z * p);  // ln(m) = 2 * z * p(z^2)

    // Convert exponent to float with proper sign handling
    sfpi::vInt sign_bit = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(exp) >> 31);
    sfpi::vInt exp_sign = sfpi::vInt(0) - sign_bit;      // 0 or 0xFFFFFFFF (all-ones mask)
    sfpi::vInt exp_abs = (exp ^ exp_sign) - exp_sign;     // absolute value via XOR trick
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(sfpi::setsgn(exp_abs, exp_sign), 0);

    // log2(base) = exponent + ln(m)/ln(2) = exponent + ln_m * (1/ln(2))
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;  // 1/ln(2) = 1.442695f
    sfpi::vFloat log2_result = exp_f32 + ln_m * vConst1Ln2;

    // Step 2: Compute 2^(pow * log2(base))
    sfpi::vFloat z_f32 = pow * log2_result;
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;  // -127.0f
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }
    v_endif;

    // Convert 2^z to exp(z*ln2) and use accurate Cody-Waite + Taylor expansion
    constexpr float LN2 = 0.693147180559945309f;
    sfpi::vFloat y = _sfpu_exp_f32_accurate_(z_f32 * LN2);  // <1 ULP float32 exp

    // Special case: 0^(negative) = NaN
    v_if((abs_base == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;  // NaN
    }
    v_endif;

    // Handle negative base (same logic as bf16 path)
    v_if(base < 0.0f) {
        sfpi::vInt pow_int = sfpi::float_to_int16(pow, 0);
        sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);

        y = sfpi::setsgn(y, pow_int << 31);  // set sign from parity of integer exponent

        v_if(pow_rounded != pow) {
            y = sfpi::vConstFloatPrgm2;  // NaN for non-integer powers of negative base
        }
        v_endif;
    }
    v_endif;

    return y;
}

// Template dispatch: selects bf16 or f32 path based on DST accumulation mode
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_power_(sfpi::vFloat base, sfpi::vFloat pow);

template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<false>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_21f_<false>(base, pow);  // bf16 path
}

template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<true>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_f32_(base, pow);  // f32 path
}

/**
 * Top-level SFPU kernel: iterates over tile faces and computes binary power.
 * Called by _llk_math_eltwise_binary_sfpu_params_ which handles face iteration
 * and DST pointer management. However, within each face, this function iterates
 * 8 times (ITERATIONS=8) to process all 8 rows of the face (4 elements per row
 * in the SFPU's SIMD width).
 *
 * The dst_tile_size_sfpi=32 accounts for the fact that SFPI sees 32 "rows" per
 * tile (64 rows in DST / SFP_DESTREG_STRIDE=2).
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_pow(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size_sfpi = 32;  // 64 rows per tile / stride 2

        // Load one SIMD-width vector (4 elements) of base from DST
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        // Load one SIMD-width vector (4 elements) of exponent from DST
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // Compute power: dispatches to _21f_ (bf16) or _f32_ based on is_fp32_dest_acc_en
        sfpi::vFloat result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1);

        // Store result back to DST at the output tile position
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;  // advance to next row within the face
    }
}

/**
 * Initialization function: sets programmable constants used by the power kernel.
 * Called once per tile (via BINOP_INIT -> power_binary_tile_init()).
 */
template <bool APPROXIMATION_MODE>
inline void sfpu_binary_pow_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;                            // 1/ln(2) for log2 computation
    sfpi::vConstFloatPrgm1 = -127.0f;                              // clamping threshold
    sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN(); // NaN for special cases
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|---|---|
| `sfpi::setsgn(val, sign)` | Sets the sign bit of a float vector. Used to compute absolute value and set result sign. |
| `sfpi::setexp(val, exp)` | Sets the exponent field of a float vector. Used to normalize mantissa to [1,2). |
| `sfpi::exexp(val)` | Extracts the biased exponent field as an integer vector. |
| `sfpi::exman9(val)` | Extracts the 9 most significant mantissa bits (used in Horner polynomial). |
| `sfpi::addexp(val, imm)` | Adds an immediate to the exponent field (multiplies by 2^imm). Maps to SFPDIVP2. |
| `sfpi::int32_to_float(val, mode)` | Converts integer vector to float vector. |
| `sfpi::float_to_int16(val, mode)` | Converts float to 16-bit integer (truncation). Used for parity check. |
| `sfpi::float_to_fp16b(val, mode)` | Converts float32 to bfloat16 with round-to-nearest-even. |
| `sfpi::abs(val)` | Computes absolute value (clears sign bit). |
| `sfpi::reinterpret<T>(val)` | Bit-cast between vFloat/vInt/vUInt without conversion. |
| `sfpi::dst_reg[idx]` | Load/store from DEST register file (SFPLOAD/SFPSTORE). |
| `sfpi::dst_reg++` | Advance DST pointer by one row (SFP_DESTREG_STRIDE). |
| `v_if / v_elseif / v_else / v_endif` | SFPU conditional execution (predicated lanes). Maps to SFPSETCC/SFPENCC. |
| `sfpi::vConstFloatPrgm0/1/2` | Programmable constant registers (SFPCONFIG). |
| `sfpi::vConst1` | Hardware constant register (1.0f). |
| `_float_to_int32_positive_(val)` | Helper: converts positive float to int32 using exponent/mantissa manipulation. |
| `_sfpu_exp_f32_accurate_(val)` | Cody-Waite + Taylor expansion for accurate float32 exp. Used in f32 path. |
| `PolynomialEvaluator::eval(...)` | Horner-form polynomial evaluator for atanh approximation (f32 path). |

#### SFPU Register Usage

- **DEST registers**: Two input tiles occupy even (`dst_index_in0 * 32`) and odd (`dst_index_in1 * 32`) positions. The output overwrites the even position (`dst_index_out * 32`, which equals `dst_index_in0 * 32` since `odst = idst0`).
- **SFPU Local Registers (LRegs)**: Used implicitly by vFloat/vInt variables. The SFPU has 4 local registers (LReg0-LReg3) for intermediate computation. The compiler maps vFloat temporaries to these registers.
- **Programmable Constants**: `vConstFloatPrgm0` = 1/ln(2), `vConstFloatPrgm1` = -127.0, `vConstFloatPrgm2` = NaN. Set once during `sfpu_binary_pow_init()`.
- **Hardware Constants**: `vConst0` (0.0f), `vConst1` (1.0f) used as needed.

#### SFPU Execution Flow

1. **Initialization** (`power_binary_tile_init()` via `BINOP_INIT`):
   - Calls `llk_math_eltwise_binary_sfpu_binary_pow_init()` which invokes `_llk_math_eltwise_binary_sfpu_init_()` (configures SFPU pipeline) then `sfpu_binary_pow_init()` (sets programmable constants).

2. **Tile Processing** (`power_binary_tile(i*2, i*2+1, i*2)` via `BINARY_SFPU_OP`):
   - Calls `llk_math_eltwise_binary_sfpu_binary_pow<APPROX, DST_ACCUM_MODE>(i*2, i*2+1, i*2)`.
   - This delegates to `_llk_math_eltwise_binary_sfpu_params_` which:
     a. Calls `_llk_math_eltwise_binary_sfpu_start_(0)` to set DST write address.
     b. In `VectorMode::RC` (default), iterates over all 4 faces of the tile.
     c. For each face, calls `calculate_sfpu_binary_pow` with 8 iterations (one per row of the face).
     d. Each iteration loads base and exponent from DST, computes `_sfpu_binary_power_`, stores result back.
     e. `dst_reg++` advances to the next SIMD row.
     f. Between faces, `TTI_SETRWC` advances the DST read/write pointer.
     g. Calls `_llk_math_eltwise_binary_sfpu_done_()` to finalize.

3. **Per-element computation** (within `_sfpu_binary_power_`):
   - **BFloat16 path** (`_sfpu_binary_power_21f_`): log2 via 3rd-order polynomial on [1,2] + exponent extraction, then 2^x via exp_21f (addexp + Horner polynomial), then special case handling, then explicit bf16 rounding.
   - **Float32 path** (`_sfpu_binary_power_f32_`): log2 via range reduction to [sqrt(2)/2, sqrt(2)], Newton-Raphson reciprocal, 6th-order atanh polynomial, then 2^x via Cody-Waite + Taylor exp, then special case handling.

4. **Packing**: After the SFPU completes, `pack_tile(i*2, cb_out0)` packs the result from DST[i*2] into the output CB.

#### SFPU Configuration

- **`fp32_dest_acc_en`**: Set to `true` when output dtype is Float32/Int32/UInt32. Controls whether `_sfpu_binary_power_21f_` (bf16) or `_sfpu_binary_power_f32_` (f32) is used.
- **`UnpackToDestMode`**: For POWER specifically, `UnpackToDestFp32` is only set when the input dtype is FLOAT32 (unlike other binary SFPU ops which always use Fp32 unpack). This ensures proper data handling for both bf16 and f32 inputs.
- **`APPROX`**: Template parameter passed through. The POWER kernel does not change behavior based on APPROX (both paths use the same code regardless).
- **`DST_ACCUM_MODE`**: Passed as `is_fp32_dest_acc_en` template parameter to select the computation path.

#### Hardware Compatibility Notes

- The Blackhole and Wormhole B0 implementations of `ckernel_sfpu_binary_pow.h` are **identical** in this codebase. Both use the same SFPI intrinsics and algorithms.
- The `_sfpu_exp_f32_accurate_` function (used in the f32 path) relies on `_sfpu_round_nearest_int32_` for Cody-Waite range reduction, which is available on both architectures.
- The `addexp` intrinsic maps to `SFPDIVP2` (divide/multiply by power of 2) which is a single-cycle instruction on both architectures.
- The `exman9` intrinsic extracts 9 mantissa bits and is used in the Horner polynomial for the bf16 path.

## Implementation Notes

1. **Two distinct accuracy levels**: The bf16 path uses a compact 21-float approximation (Moroz et al. 2022) prioritizing speed, while the f32 path uses range reduction + Newton-Raphson + 6th-order polynomial + Cody-Waite exp for sub-ULP accuracy.

2. **Special case handling**: Both paths properly handle negative bases (sign from parity of integer exponent), non-integer powers of negative bases (NaN), and zero raised to negative power (NaN). These are computed using SFPU predicated execution (`v_if`/`v_endif`).

3. **BFloat16 rounding**: The bf16 path explicitly rounds to bf16 using `float_to_fp16b` before storing to DST. This avoids truncation artifacts (e.g., `9^2 = 80.8` truncating to 80.5 instead of rounding to 81).

4. **BINOP_INIT called per tile**: The `power_binary_tile_init()` is called inside the per-tile loop (after each `copy_tile` of the exponent). This means programmable constants are re-set for every tile. While seemingly redundant, this is the standard pattern for binary SFPU ops and ensures correctness if other SFPU operations ran between blocks.

5. **DST interleaving**: Base tiles occupy even DST slots (0, 2, 4, ...) and exponent tiles occupy odd slots (1, 3, 5, ...). The SFPU reads from both and writes the result back to the base's slot. This interleaving allows processing multiple tiles in a single DST acquire/release cycle.

6. **No pre-scaling for POWER by default**: The `SFPU_OP_INIT_PRE_IN0_0` and `SFPU_OP_INIT_PRE_IN1_0` macros are not defined for bare POWER. However, if the user specifies `input_tensor_a_activation` (e.g., abs), a pre-scaling pass would be inserted via CB c_3.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the element_wise_multi_core_sfpu program factory work for binary operations?"
   **Reason**: Needed to understand the overall program factory architecture, kernel selection, and CB configuration patterns.
   **Key Findings**: Confirmed three kernels (reader, compute, writer), CB configuration with double-buffering for non-sharded, sharded globally-allocated CB support, and SPMD work distribution via `split_work_to_cores`.

2. **Query**: "How does the POWER binary SFPU operation work in ttnn?"
   **Reason**: Needed to understand the SFPU kernel implementation, the dispatch path from compute kernel to LLK, and the distinction between bf16/f32 paths.
   **Key Findings**: Identified the two-path architecture (_21f_ for bf16, _f32_ for float32), the Moroz et al. algorithm reference, and the full call chain from `power_binary_tile` through `llk_math_eltwise_binary_sfpu_binary_pow` to `calculate_sfpu_binary_pow`.

3. **Query**: "What does _llk_math_eltwise_binary_sfpu_params_ do?"
   **Reason**: Needed to understand how the LLK layer manages DST register addressing and face iteration for binary SFPU operations.
   **Key Findings**: The function handles VectorMode (RC processes all 4 faces), manages DST pointer via TTI_SETRWC between faces, and passes dst_index parameters through to the SFPU function. In RC mode, it iterates 4 times (one per face), and within each face the SFPU function iterates 8 times (one per row).

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`
   **Reason**: Needed to trace how `BinaryOpType::POWER` maps to specific defines.
   **Key Information**: POWER generates `BINOP_INIT = "power_binary_tile_init();"` and `BINARY_SFPU_OP = "power_binary_tile(i*2, i*2+1, i*2);"`. The DST indices are `idst1 = "i*2"` (base/output), `idst2 = "i*2+1"` (exponent).

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Needed to verify the API-level function signatures for `power_binary_tile` and `power_binary_tile_init`.
   **Key Information**: `power_binary_tile(idst0, idst1, odst)` calls `llk_math_eltwise_binary_sfpu_binary_pow<APPROX, DST_ACCUM_MODE>`. `power_binary_tile_init()` calls `llk_math_eltwise_binary_sfpu_binary_pow_init<APPROX>()`.

3. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
   **Reason**: Needed to understand the `_sfpu_exp_f32_accurate_` function used in the f32 POWER path.
   **Key Information**: Uses Cody-Waite range reduction followed by Taylor expansion for <1 ULP float32 accuracy. Handles overflow/underflow and NaN/infinity inputs.
