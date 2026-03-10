# POWER (binary_ng) Implementation Analysis

## Overview

The POWER operation computes element-wise exponentiation: `output = base ** exponent`, where `base` is input tensor A and `exponent` is input tensor B (or a scalar). It is implemented as an SFPU-only binary operation within the `binary_ng` framework. The operation supports full broadcasting semantics across all dimensions, scalar exponents, and handles special cases such as negative bases, zero bases with negative exponents, and non-integer exponents with negative bases.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Work Unit Definition

One work unit is a single 32x32 tile. The compute kernel processes `num_tiles_per_cycle = 1` output tile per read-compute-write cycle. Each cycle loads one LHS tile and one RHS tile into DEST registers, applies the SFPU power kernel across all 4 faces of the tile (8 iterations of 4 rows each = 32 rows total per face, times 4 faces), packs the result, and writes it to the output circular buffer.

## Tensor Format and Layout

### Input Tensor A (base)

| Property | Value |
|---|---|
| Dimension Convention | Up to N-D; internally decomposed to (nD, D, N, C, Ht, Wt) |
| Tensor Layout | TILE (32x32) |
| Memory Layout | INTERLEAVED or SHARDED (HEIGHT, WIDTH, BLOCK) |
| Buffer Type | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32 (determines algorithm path) |

### Input Tensor B (exponent)

| Property | Value |
|---|---|
| Dimension Convention | Same as A, or scalar |
| Tensor Layout | TILE (32x32) or scalar (single value broadcast to all elements) |
| Memory Layout | INTERLEAVED or SHARDED |
| Buffer Type | DRAM or L1 |
| Data Type | Matches A dtype for SFPU path; BFLOAT16 or FLOAT32 |

### Output Tensor C

| Property | Value |
|---|---|
| Dimension Convention | Broadcast-expanded shape of A and B |
| Tensor Layout | TILE (32x32) |
| Memory Layout | INTERLEAVED or SHARDED |
| Buffer Type | DRAM or L1 |
| Data Type | Configurable; typically matches input dtype |

### Layout Transformations

No explicit tilize/untilize is performed within the program factory. All tensors must already be in tiled format. The reader kernels handle broadcasting by using stride values that collapse dimensions where the input size is 1 (stride set to 0 when a dimension has size 1), effectively repeating data along broadcast dimensions.

## Data Flow Pattern

The data flow depends on whether B is a tensor or a scalar, and on the `SubtileBroadcastType`.

### Two-Tensor Path (B is a tensor, no broadcast or with broadcast)

1. **Reader kernel** reads tiles from both tensor A (into CB c_0) and tensor B (into CB c_1) from DRAM/L1, one tile at a time per tensor. For sharded inputs, the reader simply marks tiles as available without NoC reads.
2. **Compute kernel** waits for tiles in CB c_0 and CB c_1, copies them to DEST register slots 0 and 1 respectively, executes `power_binary_tile(0, 1, 0)` which runs the SFPU binary power kernel, then packs the result from DEST[0] into CB c_2.
3. **Writer kernel** reads completed tiles from CB c_2 and writes them to the output tensor in DRAM/L1 via NoC.

### Scalar Path (B is a scalar)

1. **Writer kernel** (acting as scalar data provider) fills a single tile in CB c_1 with the scalar value. This tile remains in the CB for the entire kernel execution.
2. **Reader kernel** reads tiles from tensor A into CB c_0 one at a time.
3. **Compute kernel** waits for each LHS tile, copies both LHS and the persistent scalar RHS tile to DEST, runs the SFPU power operation, packs, and outputs.
4. **Writer kernel** also handles writing output tiles from CB c_2 to DRAM/L1.

### Broadcast Path (e.g., SCALAR_A, COL_A, COL_B)

For broadcast subtypes, the compute kernel (`eltwise_binary_sfpu.cpp`) uses a frequency-based iteration pattern. The broadcast operand is loaded once and held while the non-broadcast operand cycles through `freq` tiles before the broadcast operand is refreshed.

## Circular Buffer Configuration

| CB ID | Purpose | Capacity (tiles) | Data Format | Producer | Consumer | Buffering |
|---|---|---|---|---|---|---|
| c_0 | Input A (base) | 2 (interleaved) or shard_volume (sharded) | A data format | Reader | Compute | Double-buffered (interleaved) |
| c_1 | Input B (exponent) | 1 (scalar) or 2 (tensor interleaved) or shard_volume (sharded) | B data format | Reader or Writer (scalar path) | Compute | Single (scalar) or Double (tensor) |
| c_2 | Output C (result) | 2 (interleaved) or shard_volume (sharded) | C data format | Compute | Writer | Double-buffered (interleaved) |
| c_3 | LHS intermediate (activations) | 1 | A data format | Compute (preprocess) | Compute | Single-buffered, only if LHS activations defined |
| c_4 | RHS intermediate (activations) | 1 | B data format | Compute (preprocess) | Compute | Single-buffered, only if RHS activations defined |
| c_5 | Row broadcast A scratch | 2 | A data format | Reader | Compute | Double-buffered, only for ROW_A or ROW_A_COL_B |
| c_6 | Row broadcast B scratch | 2 | B data format | Reader | Compute | Double-buffered, only for ROW_B or ROW_B_COL_A |

For POWER specifically, there are no LHS or RHS pre-activations (the `OpConfig` for POWER sets no `process_lhs` or `process_rhs`), so CBs c_3 and c_4 are not created. CBs c_5 and c_6 are only created for row-broadcast subtypes.

## Pipeline Pattern Summary

- **Interleaved mode**: CB c_0 and c_1 have capacity 2, enabling double-buffering where the reader can fill the next tile while the compute processes the current one. CB c_2 similarly allows compute to fill while writer drains.
- **Sharded mode**: CBs are sized to the full shard volume, so all tiles are available at once (no streaming overlap needed).
- **Scalar mode**: CB c_1 has capacity 1 (single tile), filled once and consumed repeatedly.

## Index Calculations

The program factory decomposes the output tensor shape into 5D+ dimensions: `(nD, D, N, C, Ht, Wt)` where `nD` collapses all dimensions beyond rank 5. The `start_tile_id` for each core is a linear offset into the flattened output tile space.

Within the reader/writer kernels, the linear `start_tile_id` is decomposed back into per-dimension indices:
```
tiles_per_n = C * Ht * Wt
tiles_per_d = N * tiles_per_n
tiles_per_nd = D * tiles_per_d
```
Each dimension index is computed via successive division/modulo operations.

For **broadcasting**, the reader kernel uses stride values per dimension. When an input dimension has size 1, the stride is set to 0 by the host code (`aHt * aWt * (aC > 1)` evaluates to 0 when `aC == 1`), causing the reader to repeat the same tiles along that dimension.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads with broadcast-aware strides. Each tile is read individually via `noc_async_read_page` with an immediate barrier (no pipelining of NoC reads within the innermost loop).
- **Sharded**: No NoC reads; data is already in L1. The reader kernel just calls `cb_reserve_back` + `cb_push_back` to expose the shard to the compute kernel.

### Write Pattern
- **Interleaved**: Sequential tile writes via `noc_async_write_page` with immediate barrier.
- **Sharded**: No NoC writes; output is already in L1 via the sharded CB.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Rectangular grid from `operation_attributes.worker_grid` |
| Work Splitting | `split_work_to_cores` distributes output tiles across cores |
| Load Balancing | Two core groups: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets the remainder |
| Remainder Handling | Cores outside both groups receive zero-tile runtime args and exit immediately |
| Sharded Mode | Grid is determined by the shard spec; each core processes its local shard |

For the zero-start rectangular grid optimization, `grid_to_cores` provides a fast row-major or column-major core enumeration. Non-zero-start or multi-range grids fall back to `corerange_to_cores`.

## Arguments

### Compile-Time Arguments

**Compute Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles_per_cycle | uint32_t | Always 1; number of output tiles per compute cycle |

The compute kernel also receives preprocessor defines:
- `BINARY_SFPU_INIT` = `power_binary_tile_init();`
- `BINARY_SFPU_OP` = `power_binary_tile`
- `BCAST_INPUT` = `""` or `"0"` or `"1"` depending on broadcast type
- `PROCESS_LHS_ACTIVATIONS(i)` = empty (no LHS preprocess for POWER)
- `PROCESS_RHS_ACTIVATIONS(i)` = empty (no RHS preprocess for POWER)
- `PROCESS_POST_ACTIVATIONS(i)` = empty (unless user specifies post-activations)

**Reader Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0..N | TensorAccessorArgs for A | uint32_t[] | Compile-time accessor config for tensor A |
| N+1..M | TensorAccessorArgs for B | uint32_t[] | Compile-time accessor config for tensor B |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

Defines: `SRC_SHARDED`, `SRC_SHARDED_B` (0 or 1).

**Writer Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0..N | TensorAccessorArgs for C | uint32_t[] | Compile-time accessor config for output tensor |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

Defines: `DST_SHARDED` (0 or 1).

### Runtime Arguments

**Reader Kernel** (21 args for two-tensor path):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | DRAM/L1 address of tensor A |
| 1 | start_tile_id | uint32_t | Starting output tile index for this core |
| 2 | src_num_tiles | uint32_t | Number of A tiles in shard (sharded only) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles this core processes |
| 4 | dst_shard_width | uint32_t | Width of shard in tiles (sharded only) |
| 5 | nD_stride | uint32_t | Tile stride for collapsed dims >5 (0 if dim==1) |
| 6 | d_stride | uint32_t | Tile stride for D dimension |
| 7 | n_stride | uint32_t | Tile stride for N dimension |
| 8 | c_stride | uint32_t | Tile stride for C dimension |
| 9 | D | uint32_t | Size of D dimension |
| 10 | N | uint32_t | Size of N dimension |
| 11 | C | uint32_t | Size of C dimension |
| 12 | Ht | uint32_t | Height in tiles |
| 13 | Wt | uint32_t | Width in tiles |
| 14 | cND | uint32_t | Number of collapsed dimensions >5 |
| 15 | src_addr_b | uint32_t | DRAM/L1 address of tensor B |
| 16 | nD_stride_b | uint32_t | B tile stride for collapsed dims >5 |
| 17 | d_stride_b | uint32_t | B tile stride for D dimension |
| 18 | n_stride_b | uint32_t | B tile stride for N dimension |
| 19 | c_stride_b | uint32_t | B tile stride for C dimension |
| 20 | src_num_tiles_b | uint32_t | Number of B tiles in shard (sharded only) |

**Writer Kernel** (11 args for two-tensor path):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | DRAM/L1 address of output tensor |
| 1 | start_tile_id | uint32_t | Starting output tile index |
| 2 | dst_num_tiles | uint32_t | Number of output tiles |
| 3 | dst_shard_width | uint32_t | Shard width in tiles |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Collapsed dims >5 |
| 10 | (unused) | uint32_t | Reserved (set to 0) |

**Writer Kernel** (11 args for scalar path):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | packed_scalar | uint32_t | Scalar exponent value packed into uint32 |
| 1 | dst_addr | uint32_t | DRAM/L1 address of output tensor |
| 2 | start_tile_id | uint32_t | Starting output tile index |
| 3 | dst_num_tiles | uint32_t | Number of output tiles |
| 4 | dst_shard_width | uint32_t | Shard width in tiles |
| 5 | D | uint32_t | Output D dimension |
| 6 | N | uint32_t | Output N dimension |
| 7 | C | uint32_t | Output C dimension |
| 8 | Ht | uint32_t | Output height in tiles |
| 9 | Wt | uint32_t | Output width in tiles |
| 10 | cND | uint32_t | Collapsed dims >5 |

**Compute Kernel** (4 args):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | tile_freq | uint32_t | Broadcast frequency (for bcast kernel variants) |
| 2 | tile_start | uint32_t | Starting offset within broadcast cycle |
| 3 | compute_scalar_value | uint32_t | Unused for POWER (set to 0) |

## Kernel Implementations

### Reader Kernel

**Two-tensor path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`

Reads tiles from both tensor A and tensor B into CB c_0 and CB c_1 respectively. Uses nested loops over the 5D+ decomposition to handle broadcasting through stride values. For sharded inputs, simply marks shard tiles as available.

**Scalar path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp`

Reads only tensor A tiles into CB c_0. The scalar value is handled by the writer kernel.

### Writer Kernel

**Two-tensor path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`

Reads completed tiles from CB c_2 and writes to output tensor. For sharded outputs, no NoC writes needed.

**Scalar path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`

Fills a single tile in CB c_1 with the packed scalar value, then writes output tiles from CB c_2 to DRAM/L1.

### Compute Kernel

The specific compute kernel file depends on the `SubtileBroadcastType`:

- **NONE (no broadcast)**: `eltwise_binary_sfpu_no_bcast.cpp` -- simple per-tile loop
- **SCALAR_A/SCALAR_B, COL_A/COL_B, ROW_B_COL_A/ROW_A_COL_B**: `eltwise_binary_sfpu.cpp` -- frequency-based broadcast iteration
- **Scalar B (no tensor B)**: `eltwise_binary_sfpu_scalar.cpp` -- RHS loaded once, LHS iterates

For the primary no-broadcast case, the compute kernel is:

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "api/compute/eltwise_binary_sfpu.h"       // provides power_binary_tile, power_binary_tile_init
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/remainder_int32.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"

#include "eltwise_utils_common.hpp"    // macro infrastructure: HAS_ACTIVATIONS, PROCESS_ACTIVATIONS, etc.
#include "eltwise_utils_sfpu.hpp"      // PREPROCESS macro for activation preprocessing

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);  // runtime arg 0: total tiles for this core

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // always 1

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;   // CB for input A (base)
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;   // CB for input B (exponent)
    constexpr auto cb_out = tt::CBIndex::c_2;        // CB for output C (result)

    // If LHS/RHS activations are defined, use intermediate CBs; otherwise alias to pre-CBs
    // For POWER, no activations are defined, so cb_post_lhs == cb_pre_lhs, cb_post_rhs == cb_pre_rhs
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    unary_op_init_common(cb_post_lhs, cb_out);  // initialize unpack/pack pipeline for LHS->output path
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));  // not used for POWER
#endif

    // For POWER with no activations, BINARY_SFPU_INIT is called once here
    // This expands to: power_binary_tile_init();
    // which calls llk_math_eltwise_binary_sfpu_binary_pow_init<APPROX>()
    // which sets vConstFloatPrgm0=1.442695, vConstFloatPrgm1=-127.0, vConstFloatPrgm2=NaN
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS is a no-op for POWER (no LHS activations)
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);  // wait for reader to produce 1 LHS tile

        // PREPROCESS is a no-op for POWER (no RHS activations)
        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);  // wait for reader to produce 1 RHS tile

        cb_reserve_back(cb_out, num_tiles_per_cycle);  // reserve space for 1 output tile

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT  // not reached for POWER
#endif
        tile_regs_acquire();  // acquire DEST register file (blocks until math RISC has exclusive access)

        // Copy LHS tile from CB c_0 to DEST[0] (even slot)
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);  // configure unpack for LHS format
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_lhs, i, i * 2);  // unpack tile i from CB to DEST[0] (base)
        }

        // Copy RHS tile from CB c_1 to DEST[1] (odd slot)
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);  // reconfigure unpack for RHS format
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_rhs, i, i * 2 + 1);  // unpack tile i from CB to DEST[1] (exponent)

#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT  // not reached for POWER without post-activations
#endif
            // Execute SFPU power: DEST[0] = DEST[0] ** DEST[1]
            // Expands to: power_binary_tile(0, 1, 0)
            // which calls llk_math_eltwise_binary_sfpu_binary_pow<APPROX, DST_ACCUM_MODE>(0, 1, 0)
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);
            PROCESS_POST_ACTIVATIONS(i * 2);  // no-op for POWER without post-activations
        }
        tile_regs_commit();  // signal that DEST writes are complete, hand off to pack

        tile_regs_wait();  // wait for pack RISC to be ready

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);  // pack DEST[0] result into output CB c_2
        }
        tile_regs_release();  // release DEST registers

        cb_push_back(cb_out, num_tiles_per_cycle);      // notify writer that 1 output tile is ready
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);  // free LHS tile slot for reader
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);  // free RHS tile slot for reader
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h`
(Wormhole variant at: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h`)

Both architecture variants are identical in implementation.

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"
#include "ckernel_sfpu_exp.h"           // provides _sfpu_exp_f32_accurate_ used by f32 path
#include "sfpu/ckernel_sfpu_polyval.h"  // provides PolynomialEvaluator used by f32 path
#include "sfpi.h"                       // SFPI programming interface for vector operations

using namespace sfpi;

namespace ckernel {
namespace sfpu {

/**
 * BF16 path: _sfpu_binary_power_21f_
 * Computes base**pow using: 2^(pow * log2(base))
 * Uses 3rd-order polynomial approximation for log2 and Moroz et al. exp_21f algorithm for 2^x
 * This is the lower-precision, faster path used when is_fp32_dest_acc_en == false
 */
template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_binary_power_21f_(sfpi::vFloat base, sfpi::vFloat pow) {
    // === Step 1: Compute log2(base) ===

    sfpi::vFloat absbase = setsgn(base, 0);       // take absolute value by clearing sign bit
    sfpi::vFloat x = sfpi::setexp(absbase, 127);   // normalize mantissa to range [1, 2) by setting exponent to bias

    // 3rd order polynomial approximation of ln(x) for x in [1,2], coefficients from rminimax
    // This computes ln(mantissa) using Horner's method
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Extract the biased exponent and convert to float
    sfpi::vInt exp = sfpi::exexp(base);  // SFPU instruction: extract exponent field as signed int
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }  // manual abs + sign for negative exponents
    v_endif;
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(exp, 0);  // convert integer exponent to float

    // Combine: log2(base) = exponent + ln(mantissa) * (1/ln(2))
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;  // 1.442695 = 1/ln(2), loaded from programmable constant
    sfpi::vFloat log2_result = exp_f32 + series_result * vConst1Ln2;

    // === Step 2: Compute 2^(pow * log2(base)) using Moroz et al. exp_21f ===

    sfpi::vFloat z_f32 = pow * log2_result;
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;  // -127.0
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }  // clamp to prevent overflow for large negative exponents
    v_endif;

    // Moroz exp_21f: compute 2^z by bit manipulation
    // Multiply z by 2^23 (mantissa bits of float32) using addexp which is a single SFPDIVP2 instruction
    z_f32 = addexp(z_f32, 23);  // z_f32 *= 2^23; single-cycle SFPDIVP2 instruction
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);  // IEEE 754 representation of 1.0f = float bias * 2^23
    sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias);  // convert to integer for bit manipulation

    // Split z into integer exponent part and fractional mantissa part
    sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));    // extract exponent bits (z & 0x7f800000)
    sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // extract mantissa bits (z & 0x007fffff)

    // Compute 2^fraction using Horner-form polynomial with pre-computed constants
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + zif, 0);

    d2 = d1 * d2;
    zif = _float_to_int32_positive_(d2 * d3);  // compute fractional 2^x result as integer bits

    // Restore the integer exponent to get final 2^z result
    zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

    // === Post-processing: handle special cases ===

    sfpi::vInt pow_int = sfpi::float_to_int16(pow, 0);  // truncate exponent to int16 for sign/parity checks
    sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);  // round-trip to check if pow is integer

    // 0^(negative) = NaN
    v_if((absbase == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;  // NaN from programmable constant
    }
    v_endif;

    // Negative base handling
    v_if(base < 0.0f) {
        // Sign of result depends on parity of exponent: odd -> negative, even -> positive
        y = setsgn(y, pow_int << 31);  // shift LSB of pow_int to sign bit position

        // Non-integer power of negative base -> NaN (complex result)
        v_if(pow_rounded != pow) {
            y = sfpi::vConstFloatPrgm2;  // NaN
        }
        v_endif;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        // When DEST is bfloat16, explicitly round to bf16 using round-to-nearest-even
        // to avoid truncation artifacts (e.g., 9^2 = 80.8 truncates to 80.5 instead of 81)
        y = reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

/**
 * FP32 path: _sfpu_binary_power_f32_
 * Higher-precision implementation using improved log2 and Cody-Waite + Taylor exp
 * Used when is_fp32_dest_acc_en == true (FLOAT32 input/output)
 */
sfpi_inline sfpi::vFloat _sfpu_binary_power_f32_(sfpi::vFloat base, sfpi::vFloat pow) {
    // === Step 1: Compute log2(base) using improved log ===

    sfpi::vFloat abs_base = sfpi::abs(base);
    sfpi::vFloat m = sfpi::setexp(abs_base, 127);  // normalize mantissa to [1, 2)
    sfpi::vInt exp = sfpi::exexp(abs_base);          // extract exponent

    // Range reduction: ensure m in [sqrt(2)/2, sqrt(2)] for better polynomial convergence
    constexpr float SQRT2 = 1.4142135381698608f;
    v_if(m >= SQRT2) {
        m = m * 0.5f;    // divide by 2
        exp = exp + 1;   // compensate exponent
    }
    v_endif;

    // Transform to z = (m-1)/(m+1) for log series
    sfpi::vFloat m_plus_1 = m + sfpi::vConst1;
    sfpi::vFloat m_minus_1 = m - sfpi::vConst1;

    // Compute reciprocal of (m+1) using linear initial guess + 2 Newton-Raphson iterations
    sfpi::vFloat recip = sfpi::vConst1 - 0.2426406871192851f * m_plus_1;  // linear approximation on [1.7, 2.4]
    recip = recip * (2.0f - m_plus_1 * recip);  // 1st Newton-Raphson refinement
    recip = recip * (2.0f - m_plus_1 * recip);  // 2nd Newton-Raphson for float32 precision
    sfpi::vFloat z = m_minus_1 * recip;

    // Polynomial approximation of atanh-like series: ln((1+z)/(1-z)) = 2*z*(1 + z^2/3 + z^4/5 + ...)
    sfpi::vFloat z2 = z * z;
    sfpi::vFloat p = PolynomialEvaluator::eval(
        z2, sfpi::vConst1, 0.3333333333333333f, 0.2f, 0.14285714285714285f,
        0.1111111111111111f, 0.09090909090909091f);  // coefficients: 1, 1/3, 1/5, 1/7, 1/9, 1/11
    sfpi::vFloat ln_m = 2.0f * (z * p);  // ln(m) result

    // Convert exponent to float with proper sign handling
    sfpi::vInt sign_bit = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(exp) >> 31);
    sfpi::vInt exp_sign = sfpi::vInt(0) - sign_bit;     // 0 or 0xFFFFFFFF
    sfpi::vInt exp_abs = (exp ^ exp_sign) - exp_sign;    // absolute value via two's complement
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(sfpi::setsgn(exp_abs, exp_sign), 0);

    // log2(base) = exponent + ln(mantissa) / ln(2)
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;  // 1/ln(2)
    sfpi::vFloat log2_result = exp_f32 + ln_m * vConst1Ln2;

    // === Step 2: Compute 2^(pow * log2(base)) ===

    sfpi::vFloat z_f32 = pow * log2_result;
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;  // -127.0
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }  // clamp for underflow
    v_endif;

    // Use Cody-Waite + Taylor exp for <1 ULP float32 accuracy
    constexpr float LN2 = 0.693147180559945309f;
    sfpi::vFloat y = _sfpu_exp_f32_accurate_(z_f32 * LN2);  // 2^z = exp(z * ln(2))

    // === Special case handling (same as bf16 path) ===

    v_if((abs_base == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;  // 0^(negative) = NaN
    }
    v_endif;

    v_if(base < 0.0f) {
        sfpi::vInt pow_int = sfpi::float_to_int16(pow, 0);
        sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);
        y = sfpi::setsgn(y, pow_int << 31);  // set sign based on parity of integer exponent
        v_if(pow_rounded != pow) {
            y = sfpi::vConstFloatPrgm2;  // non-integer power of negative base = NaN
        }
        v_endif;
    }
    v_endif;

    return y;
}

// Template dispatch: select algorithm based on DEST accumulation mode
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_power_(sfpi::vFloat base, sfpi::vFloat pow);

template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<false>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_21f_<false>(base, pow);  // BF16 path
}

template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<true>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_f32_(base, pow);  // FP32 high-precision path
}

/**
 * Top-level SFPU kernel entry point for binary power
 * Called once per face (4 times per tile in RC vector mode)
 * Each call processes 8 iterations (rows) within the face
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_pow(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        // Each tile face has 32 rows in SFPI addressing (64 / SFP_DESTREG_STRIDE = 32)
        constexpr uint dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];  // load base from DEST
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];  // load exponent from DEST

        sfpi::vFloat result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1);  // compute base^exponent

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;  // store result back to DEST
        sfpi::dst_reg++;  // advance to next row within the face
    }
}

/**
 * Initialization: set programmable constants used by the power algorithm
 */
template <bool APPROXIMATION_MODE>
inline void sfpu_binary_pow_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;                            // 1/ln(2) for log2 computation
    sfpi::vConstFloatPrgm1 = -127.0f;                              // underflow clamp threshold
    sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN(); // NaN for invalid cases
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|---|---|
| `sfpi::setsgn(val, sign)` | Sets the sign bit of a float value; maps to SFPSETSGN instruction |
| `sfpi::setexp(val, exp)` | Sets the exponent field of a float; maps to SFPSETEXP instruction |
| `sfpi::exexp(val)` | Extracts the exponent field as a signed integer; maps to SFPEXEXP instruction |
| `sfpi::exman9(val)` | Extracts the 9-bit mantissa + lower bits; maps to SFPEXMAN instruction |
| `sfpi::abs(val)` | Computes absolute value (clears sign bit) |
| `sfpi::addexp(val, n)` | Adds `n` to the exponent field (multiply by 2^n); maps to SFPDIVP2 instruction |
| `sfpi::int32_to_float(val, mode)` | Converts integer to float; maps to SFPCAST instruction |
| `sfpi::float_to_int16(val, mode)` | Converts float to 16-bit integer (truncation); maps to SFPCAST instruction |
| `sfpi::float_to_fp16b(val, mode)` | Converts float32 to bfloat16 with rounding; maps to SFPCAST instruction |
| `_float_to_int32_positive_(val)` | Converts positive float to int32 via bit manipulation |
| `_sfpu_exp_f32_accurate_(val)` | Computes exp(x) using Cody-Waite + Taylor series for FP32 accuracy |
| `sfpi::reinterpret<T>(val)` | Reinterprets bits between vFloat/vInt without conversion |
| `v_if(...) { } v_endif` | Predicated execution (SFPSETCC + conditional stores); vectorized branch |
| `sfpi::dst_reg[idx]` | Load/store from DEST register file; maps to SFPLOAD/SFPSTORE |
| `sfpi::vConstFloatPrgm0/1/2` | Programmable SFPU constants; set via SFPLOADI |
| `PolynomialEvaluator::eval(...)` | Horner-form polynomial evaluation using MAD instructions |

#### SFPU Register Usage

- **DEST registers**: Two input tiles occupy DEST[0] (base, at `dst_index_in0 * 32`) and DEST[1] (exponent, at `dst_index_in1 * 32`). The output overwrites DEST[0] (at `dst_index_out * 32`).
- **SFPU LRegs (vFloat/vInt)**: The kernel uses multiple local vector registers for intermediate calculations (absbase, x, series_result, exp_f32, log2_result, z_f32, z, zii, zif, d1, d2, d3, y, pow_int, pow_rounded). These map to SFPU local registers (LRegs 0-7).
- **Programmable constants**: `vConstFloatPrgm0` = 1/ln(2), `vConstFloatPrgm1` = -127.0, `vConstFloatPrgm2` = NaN. Set during `sfpu_binary_pow_init()`.
- **`dst_reg++`**: Advances the DEST register pointer by `SFP_DESTREG_STRIDE` (2 rows) to process the next row pair within a face.

#### SFPU Execution Flow

1. **Initialization** (`sfpu_binary_pow_init`): Three programmable constants are loaded: 1/ln(2), -127.0, and NaN. These persist across all tiles processed by this core.

2. **Per-tile dispatch** (`_llk_math_eltwise_binary_sfpu_params_`): The LLK params wrapper is called with `VectorMode::RC` (default), which iterates over all 4 faces of the 32x32 tile:
   - For each face, `calculate_sfpu_binary_pow` is called once
   - After each face, `TTI_SETRWC` advances the DEST register write pointer by 16 rows (8+8) to the next face
   - Total: 4 faces x 8 iterations x 4 elements per vector = 1024 elements = 32x32 tile

3. **Per-face computation** (`calculate_sfpu_binary_pow`, 8 iterations per face):
   - Load one row-vector of base values from DEST[in0] and one from DEST[in1]
   - Call `_sfpu_binary_power_<is_fp32_dest_acc_en>` which dispatches to either:
     - `_sfpu_binary_power_21f_` (BF16 path): 3rd-order polynomial log2 + Moroz exp_21f
     - `_sfpu_binary_power_f32_` (FP32 path): improved log2 with Newton-Raphson reciprocal + Cody-Waite Taylor exp
   - Store result back to DEST[out]
   - Advance `dst_reg++`

4. **Algorithm detail** (both paths share the same structure):
   - Compute `log2(|base|)` = exponent + polynomial(mantissa) * (1/ln(2))
   - Compute `z = pow * log2(|base|)`, clamped to [-127, +inf)
   - Compute `2^z` (BF16: bit-manipulation exp_21f; FP32: Cody-Waite accurate exp)
   - Handle special cases: 0^negative -> NaN, negative base with odd power -> negate, negative base with non-integer power -> NaN
   - BF16 path: explicit round-to-nearest-even conversion to avoid truncation artifacts

5. **Pack**: After all 4 faces, `tile_regs_commit()` signals pack RISC, which reads DEST[0] and packs into the output CB.

#### SFPU Configuration

- **`UnpackToDestMode`**: For POWER specifically, the program factory does NOT force `UnpackToDestFp32` for all CBs (unlike most other binary SFPU ops). Instead, it conditionally sets `UnpackToDestFp32` only when the input dtype is FLOAT32. This is because the power algorithm has dedicated BF16 and FP32 paths.
- **`fp32_dest_acc_en`**: Enabled when output, or both inputs, are FLOAT32/INT32/UINT32. This selects the `_sfpu_binary_power_f32_` path.
- **`DST_ACCUM_MODE`**: Passed as template parameter to the LLK function; determines DEST register sizing and accumulation behavior.
- **`APPROX`**: Template parameter passed through from compute config; for POWER both paths are the same regardless of this flag.
- **Math fidelity**: Not directly configurable for SFPU ops (fidelity primarily affects FPU matrix operations).

#### Hardware Compatibility Notes

The Blackhole and Wormhole implementations of `ckernel_sfpu_binary_pow.h` are **identical**. Both architectures use the same SFPI instruction set for this operation. The key SFPU instructions used (SFPLOAD, SFPSTORE, SFPSETSGN, SFPSETEXP, SFPEXEXP, SFPEXMAN, SFPDIVP2, SFPCAST, SFPMAD, SFPSETCC) are available on both Wormhole B0 and Blackhole architectures.

The `_sfpu_exp_f32_accurate_` function used by the FP32 path is also defined in `ckernel_sfpu_exp.h` which has architecture-specific implementations, but the power kernel itself is architecture-agnostic.

## Implementation Notes

1. **Two algorithm paths**: The POWER operation uniquely selects between a fast BF16 polynomial approximation (`_sfpu_binary_power_21f_`) and a high-precision FP32 path (`_sfpu_binary_power_f32_`). This is controlled at compile time by the `is_fp32_dest_acc_en` template parameter, which is set based on whether the input/output types require FP32 DEST accumulation.

2. **No FPU fallback**: Unlike some binary ops (ADD, SUB, MUL) that can use both FPU and SFPU paths, POWER is SFPU-only. Attempting to use it with the FPU path throws `TT_THROW("Unsupported binary op for FPU")`.

3. **Special UnpackToDestMode handling**: POWER is the only binary_ng SFPU operation that does NOT unconditionally force `UnpackToDestFp32` for all circular buffers. Lines 741-755 of the program factory explicitly check `op_type != BinaryOpType::POWER` and conditionally set the unpack mode based on input dtype. This is because the BF16 path (`_sfpu_binary_power_21f_`) works directly with BF16 data in DEST and includes its own explicit `float_to_fp16b` rounding at the end.

4. **Broadcast support**: The operation supports all `SubtileBroadcastType` variants (NONE, SCALAR_A/B, ROW_A/B, COL_A/B, and mixed ROW_COL), allowing flexible broadcasting patterns between the base and exponent tensors.

5. **Moroz et al. algorithm**: The BF16 path implements the `exp_21f` algorithm from "Simple Multiple Precision Algorithms for Exponential Functions" (IEEE Signal Processing Magazine, 2022). Key optimization: `addexp(z, 23)` compiles to a single-cycle `SFPDIVP2` instruction instead of a multiply-by-constant, saving 2 cycles.

6. **Accuracy consideration for BF16**: The explicit `float_to_fp16b` conversion at line 147 uses round-to-nearest-even instead of truncation. Without this, results like 9^2 would yield 80.5 instead of 81 due to the intermediate FP32->BF16 truncation in SFPSTORE.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng program factory work? What are the different subtypes (scalar, bcast, etc.) and how does it select kernels?"
   **Reason**: Needed to understand the overall binary_ng framework architecture, kernel selection logic, and broadcast handling.
   **Key Findings**: The `BinaryNgKernelConfig` maps `SubtileBroadcastType` to specific kernel names. The program factory creates reader/compute/writer kernels with defines that configure the operation type. Circular buffers are double-buffered (2 tiles) for interleaved mode and shard-sized for sharded mode.

2. **Query**: "How does the SFPU power operation work in binary_ng? What compute kernels are used for eltwise_power in the binary_ng framework?"
   **Reason**: Needed to understand the SFPU power kernel call chain and algorithm details.
   **Key Findings**: POWER maps to `SfpuBinaryOp::POWER`, uses `power_binary_tile` / `power_binary_tile_init`, dispatches to `calculate_sfpu_binary_pow` which has two algorithm paths based on `is_fp32_dest_acc_en`. The BF16 path uses polynomial log2 + exp_21f, the FP32 path uses improved log2 with Newton-Raphson + Cody-Waite Taylor exp.

### Documentation References

1. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h`
   **Reason**: Primary SFPU kernel implementation for the power operation.
   **Key Information**: Complete implementation of both BF16 and FP32 paths, special case handling, programmable constant initialization.

2. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
   **Reason**: Understanding the LLK dispatch wrapper that iterates over tile faces.
   **Key Information**: `_llk_math_eltwise_binary_sfpu_params_` dispatches the SFPU function 4 times (one per face) in RC vector mode, using `TTI_SETRWC` to advance the DEST register pointer between faces.
