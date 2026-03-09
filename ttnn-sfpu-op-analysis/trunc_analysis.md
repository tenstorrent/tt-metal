# Trunc Implementation Analysis

## Overview

The `trunc` operation performs element-wise truncation on a tensor, removing the fractional part of each floating-point value and returning the integer part (rounded toward zero). For example, `trunc(3.7) = 3.0` and `trunc(-2.3) = -2.0`. This is mathematically equivalent to `sign(x) * floor(abs(x))`.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

The trunc operation uses the shared `UnaryProgramFactory` infrastructure. It does not have a dedicated program factory; instead, the generic unary SFPU program factory dispatches to the trunc-specific SFPU kernel through compile-time macro defines.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles per block (always 1 for this factory) |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | Flat tile iteration (no dimension semantics in kernel) |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, or UINT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or specified output dtype) |

### Layout Transformations

No layout transformations (tilize/untilize, reshard, or format conversions) are performed. Input and output are both in tile layout. The unpacker converts from the source data format to the DEST register format internally, and the packer converts back.

## Data Flow Pattern

1. **Reader kernel** reads one tile at a time from DRAM/L1 via NoC into CB c_in0 (input staging).
2. **Compute kernel** waits for a tile in CB c_in0, copies it to DEST registers, executes the SFPU trunc operation on the tile, packs the result into CB c_out (output staging).
3. **Writer kernel** waits for a tile in CB c_out, writes it back to DRAM/L1 via NoC.

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 buffer | CB c_in0 | cb_reserve_back, cb_push_back |
| 2 | Compute | CB c_in0 | CB c_out | cb_wait_front, cb_pop_front, cb_reserve_back, cb_push_back |
| 3 | Writer | CB c_out | DRAM/L1 buffer | cb_wait_front, cb_pop_front |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0_cb | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | output_cb | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Notes:
- CB c_1 (tmp0_cb) is **not** created for trunc. It is only created for HARDSHRINK, CBRT, or LOGIT operations.
- Both input and output CBs have capacity for 2 tiles with a block size of 1 tile, enabling double-buffered operation.
- Page size matches the tile size of the respective data format.

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 are double-buffered (capacity = 2x block size). This allows the reader to fill one tile slot while the compute kernel processes from another, and similarly for compute/writer overlap. The three-stage pipeline (reader -> compute -> writer) can overlap across tiles.

## Index Calculations

Tile indices are linear: each core processes a contiguous range of tile IDs from `start_id` to `start_id + num_pages_per_core`. The `TensorAccessor` utility handles the mapping from linear tile ID to physical memory address, accounting for interleaved bank distribution. No special index transformations are needed since the operation is element-wise and dimension-agnostic.

## Memory Access Patterns

### Read Pattern

Sequential linear tile access. The reader iterates from `start_id` to `start_id + num_pages` in order, reading one tile per iteration via `noc_async_read_page`. Each read is followed by `noc_async_read_barrier` (blocking until complete before pushing to CB).

### Write Pattern

Sequential linear tile access. The writer iterates from `start_id` to `start_id + num_pages` in order, writing one tile per iteration via `noc_async_write_page`. Writes are flushed per tile with `noc_async_writes_flushed`, and a final `noc_async_write_barrier` ensures all writes complete.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (flattened to 1D iteration) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size` (device-dependent) |
| **Total cores** | Determined by `split_work_to_cores` based on total tile count |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two core groups: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles |

Core iteration order is column-major: `core = {i / num_cores_y, i % num_cores_y}`. The `split_work_to_cores` utility divides tiles as evenly as possible across available cores, with at most a 1-tile difference between the two groups.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for source buffer (bank mapping, page size, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for destination buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core |
| 1 | per_core_block_dim | uint32_t | Number of tiles per block (always 1 for this factory) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of tiles this core processes |
| 2 | start_id | uint32_t | Starting tile index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of tiles this core processes |
| 2 | start_id | uint32_t | Starting tile index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for trunc (set to 0) |
| 1 | packed_scalar2 | uint32_t | Unused for trunc (set to 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 | CB c_0 | Read tiles sequentially |
| compute | RISCV_2 (math) | N/A | CB c_0 | CB c_2 | copy_tile, trunc SFPU op, pack_tile |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 | Write tiles sequentially |

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential tile reader. Creates a `TensorAccessor` from compile-time args and the runtime source address. Iterates from `start_id` to `start_id + num_pages`, reading one tile per iteration into CB c_0. Each read blocks on `noc_async_read_barrier` before pushing.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential tile writer. Creates a `TensorAccessor` from compile-time args and the runtime destination address. Iterates from `start_id` to `start_id + num_pages`, writing one tile per iteration from CB c_2. Flushes writes per tile and barriers at the end.

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File

`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // conditionally includes rounding.h when SFPU_OP_ROUND_FAMILY_INCLUDE is defined
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // number of tile blocks this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block (always 1 for trunc via UnaryProgramFactory)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // initializes unpack/pack pipelines for input CB c_0 and output CB c_2
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve space in output CB for one block of tiles
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DEST registers for math

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // wait until reader has produced 1 tile in input CB

            copy_tile(tt::CBIndex::c_0, 0, 0);  // unpack tile 0 from CB c_0 into DEST register 0

// For trunc, SFPU_OP_CHAIN_0 expands to:
//   rounding_op_tile_init();   -- SFPU_OP_CHAIN_0_INIT_0: initializes SFPU for rounding operations
//   trunc_tile(0);             -- SFPU_OP_CHAIN_0_FUNC_0: executes trunc on tile in DEST[0]
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();  // signal that DEST registers are ready for packing

            tile_regs_wait();  // wait for pack stage to be ready

            pack_tile(0, tt::CBIndex::c_2);  // pack DEST[0] result into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // free the consumed input tile from CB c_0

            tile_regs_release();  // release DEST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // publish the block of output tiles to writer
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File

- **Blackhole**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_rounding_ops.h`
- **Wormhole B0**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_rounding_ops.h`

Both architectures share identical `_trunc_body_()` logic. The only difference is the address modifier used in `_calculate_trunc_()`: Blackhole uses `ADDR_MOD_7`, while Wormhole B0 uses `ADDR_MOD_3`.

#### Annotated SFPU Kernel Source (Blackhole version)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <climits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_isinf_isnan.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// computes L1=trunc(L0).
// This is the core truncation primitive. It zeroes out the fractional bits of a
// floating-point number by constructing and applying a bitmask.
// Input: LREG0 contains the float value.
// Output: LREG1 contains the truncated result.
// Clobbers: LREG2 (exponent), LREG3 (constant 23).
inline void _trunc_body_()
{
    // Load the constant 23 into LREG3. This is the number of mantissa bits in IEEE 754 float32.
    // Used later to compute the shift amount for the mantissa mask.
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_SHORT, 23);

    // Load 0x8000 in FLOATB format into LREG1, which produces 0x8000_0000 (negative zero / sign-only mask).
    // This initializes the mask to preserve only the sign bit for values with |x| < 1.
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);

    // Extract the unbiased exponent from LREG0 into LREG2.
    // Simultaneously sets condition codes: lanes where the exponent is negative (|x| < 1)
    // are disabled for subsequent conditional instructions. The SGN_EXP flag checks sign of
    // the exponent, and COMP_EXP compares it, effectively disabling lanes where exp < 0.
    // For those disabled lanes, the mask stays as 0x8000_0000 (sign bit only), so trunc(x) = +/-0.
    TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);

    // Overwrite LREG1 with 0xFFFF as a short integer, which represents 0x0000_FFFF.
    // However, since this is loaded as a 16-bit short sign-extended, it becomes 0xFFFF_FFFF (all ones).
    // This is the starting mask for lanes where |x| >= 1.
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);

    // Compute LREG2 = LREG3 - LREG2 = 23 - exponent.
    // This gives the number of fractional mantissa bits to zero out.
    // The ARG_2SCOMP_LREG_DST flag means: negate LREG2 (dst) before adding LREG3 (src), store in LREG2.
    // CC_GTE0: only execute for lanes where the result (23 - exp) >= 0.
    // If exp > 23, the number is already an integer (no fractional bits), so no masking is needed.
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);

    // Left-shift LREG1 (0xFFFF_FFFF) by the amount in LREG2 (23 - exp).
    // This creates a mask like 0xFFFF_FC00 that has 1s in the sign, exponent, and integer mantissa bits,
    // and 0s in the fractional mantissa bits.
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SHFT_LREG);

    // Re-enable all lanes by clearing condition codes.
    TTI_SFPENCC(0, 0, 0, 0);

    // Bitwise AND: LREG1 = LREG0 & LREG1.
    // This zeroes out the fractional mantissa bits, producing the truncated result.
    TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
}

// Template function that iterates over tile faces, loading from DEST, applying
// _trunc_body_, and storing back.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_trunc_()
{
    for (int d = 0; d < ITERATIONS; d++)  // iterate over 8 tile faces (4 faces x 16 rows = 64 rows per face group; 8 iterations covers 32x32 tile)
    {
        // Load one face-row (32 elements) from DEST register into LREG0.
        // ADDR_MOD_7 auto-increments the DEST row pointer after load.
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);

        _trunc_body_();  // compute trunc in LREG1

        // Store the truncated result from LREG1 back to DEST register.
        // ADDR_MOD_7 auto-increments the DEST row pointer after store.
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);

        sfpi::dst_reg++;  // advance SFPI's DEST register pointer to the next face-row
    }
}

} // namespace sfpu
} // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `TTI_SFPLOADI` | Loads an immediate value into a local register (LREG). Supports different modes: `SFPLOADI_MOD0_SHORT` loads a 16-bit sign-extended integer; `SFPLOADI_MOD0_FLOATB` loads a 16-bit value into the upper 16 bits of the 32-bit register (bfloat16 format). |
| `TTI_SFPEXEXP` | Extracts the unbiased exponent from a floating-point value in a local register. Can set condition codes based on the sign and magnitude of the exponent, enabling per-lane conditional execution. |
| `TTI_SFPIADD` | Performs integer addition on local registers. With `ARG_2SCOMP_LREG_DST`, it negates the destination operand before adding (effectively computing `src - dst`). Can set condition codes (e.g., `CC_GTE0` enables only lanes where the result is >= 0). |
| `TTI_SFPSHFT2` | Performs a bitwise left shift of one local register by the amount stored in another local register. `SFPSHFT2_MOD1_SHFT_LREG` specifies that the shift amount comes from a register rather than an immediate. |
| `TTI_SFPENCC` | Resets (clears) the condition codes, re-enabling all SIMD lanes for subsequent instructions. |
| `TTI_SFPAND` | Performs bitwise AND between two local registers, storing the result in the destination. |
| `TTI_SFPLOAD` | Loads a 32-element vector from a DEST register row into a local register (LREG). The address modifier controls auto-increment behavior. |
| `TTI_SFPSTORE` | Stores a 32-element vector from a local register back to a DEST register row. The address modifier controls auto-increment behavior. |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Input: holds the floating-point value loaded from DEST. Read by `_trunc_body_` but not modified. |
| **LREG1** | Output: initially loaded with a mask value (0x8000_0000 for small values, then 0xFFFF_FFFF for larger values), shifted to create the truncation mask, and finally holds the AND result (truncated value). |
| **LREG2** | Scratch: holds the extracted exponent, then `23 - exponent` (the shift amount). |
| **LREG3** | Scratch: holds the constant 23 (number of mantissa bits in float32). |
| **DEST[row]** | The tile data register. One row (32 elements) is loaded/stored per SFPU iteration. 8 iterations cover a full tile face group. |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `cb_wait_front(c_0, 1)` to wait for a tile from the reader, then `copy_tile(c_0, 0, 0)` to unpack the tile from CB c_0 into DEST register 0.

2. **SFPU initialization**: `rounding_op_tile_init()` is called, which expands to `SFPU_UNARY_KERNEL_INIT(unused, APPROX)` -> `llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROX>()`. This sets up address modifiers and SFPU counters.

3. **SFPU dispatch**: `trunc_tile(0)` is called, which expands via the `SFPU_TWO_PARAM_KERNEL` macro to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_trunc_<APPROX, 8>, 0, (int)VectorMode::RC)`. This sets the DEST write address to tile index 0 and calls `_calculate_trunc_`.

4. **SFPU iteration loop**: `_calculate_trunc_` iterates 8 times (ITERATIONS=8). Each iteration:
   - `TTI_SFPLOAD` loads 32 elements (one face-row) from DEST into LREG0.
   - `_trunc_body_()` computes the truncation:
     a. Loads constant 23 into LREG3.
     b. Initializes mask in LREG1 to 0x8000_0000 (sign-bit only).
     c. Extracts exponent from LREG0 into LREG2; disables lanes with exp < 0 (|x| < 1).
     d. For enabled lanes (|x| >= 1), loads 0xFFFF_FFFF into LREG1.
     e. Computes shift amount = 23 - exp; further disables lanes where exp > 23 (already integers).
     f. Left-shifts the mask by the shift amount, zeroing out fractional bit positions.
     g. Re-enables all lanes.
     h. ANDs LREG0 (input) with LREG1 (mask), storing result in LREG1.
   - `TTI_SFPSTORE` writes the result from LREG1 back to DEST.
   - `dst_reg++` advances to the next face-row.

5. **Result packing**: After the SFPU completes, `tile_regs_commit()` signals readiness, `tile_regs_wait()` synchronizes with the packer, and `pack_tile(0, c_2)` packs DEST[0] into the output circular buffer.

6. **Cleanup**: `cb_pop_front(c_0, 1)` frees the input tile, `tile_regs_release()` releases DEST registers.

#### SFPU Configuration

- **Math fidelity**: `MathFidelity::HiFi4` (highest fidelity, set in program factory).
- **Approximation mode**: `false` (trunc's `get_op_approx_mode` returns false by default).
- **fp32_dest_acc_en**: Configurable per-call via `UnaryParams::fp32_dest_acc_en`.
- **unpack_to_dest_mode**: Default (`UnpackToDestMode::Default`), unless `preserve_fp32_precision` is set, in which case `UnpackToDestFp32` is used.
- **Compile-time defines**: `SFPU_OP_ROUND_FAMILY_INCLUDE=1` (triggers inclusion of `rounding.h`), `SFPU_OP_CHAIN_0` = `rounding_op_tile_init(); trunc_tile(0);`, plus input dtype flag (`INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, or `INP_FLOAT`).

#### Hardware Compatibility Notes

The `_trunc_body_()` implementation is **identical** between Blackhole and Wormhole B0. Both use the same sequence of TTI instructions (SFPLOADI, SFPEXEXP, SFPIADD, SFPSHFT2, SFPENCC, SFPAND).

The only difference is in `_calculate_trunc_()`:
- **Blackhole** uses `ADDR_MOD_7` for SFPLOAD/SFPSTORE addressing.
- **Wormhole B0** uses `ADDR_MOD_3` for SFPLOAD/SFPSTORE addressing.

These address modifiers control auto-increment behavior of the DEST register pointer and are architecture-specific configurations. The mathematical logic is the same on both platforms.

Note: The Wormhole B0 version of `_floor_body_()` and `_ceil_body_()` (which build on `_trunc_body_()`) use `SFPSETCC`+`SFPIADD` instead of `SFPGT`, since Wormhole does not have the `SFPGT` instruction. However, this difference does not affect `_trunc_body_()` itself, which is identical across architectures.

## Implementation Notes

1. **Bit manipulation approach**: The trunc operation is implemented entirely through integer bit manipulation of the IEEE 754 floating-point representation, rather than using floating-point arithmetic. This makes it exact (no rounding errors) and efficient.

2. **Three-case handling via condition codes**:
   - **|x| < 1** (exponent < 0): The mask stays as 0x8000_0000 (initialized before SFPEXEXP disables these lanes). The AND produces +/-0.0, which is correct since trunc of any value in (-1, 1) is 0.
   - **1 <= |x| < 2^23** (0 <= exponent < 23): The mask is constructed to zero out fractional mantissa bits. The shift amount `23 - exp` determines how many low bits to clear.
   - **|x| >= 2^23** (exponent >= 23): The SFPIADD result (23 - exp) is negative, so CC_GTE0 disables these lanes. The mask stays 0xFFFF_FFFF, and AND preserves the value unchanged. This is correct since such large floats have no fractional part.

3. **LREG clobbering**: `_trunc_body_()` clobbers LREG2 and LREG3. Code that calls `_trunc_body_()` (such as `calculate_rdiv` and `_sfpu_binary_fmod_`) must be aware of this and read these registers afterward to inform the SFPI register allocator.

4. **Reuse as a primitive**: `_trunc_body_()` serves as a building block for `floor`, `ceil`, `frac`, `rdiv`, and `fmod` operations, making it one of the most widely-reused SFPU primitives.

5. **No temporary CB needed**: Unlike operations such as HARDSHRINK or CBRT, trunc does not require the temporary circular buffer (CB c_1).

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the unary operation program factory structured in ttnn? Specifically, how does unary_program_factory.cpp work for SFPU operations like trunc?"
   **Reason**: Needed to understand the overall program factory infrastructure and how SFPU operations are dispatched.
   **Key Findings**: The factory uses a generic `eltwise_sfpu.cpp` compute kernel configured with compile-time macro defines. Reader/writer kernels use `reader_unary_interleaved_start_id.cpp` and `writer_unary_interleaved_start_id.cpp`. Circular buffers are double-buffered with 2 tiles capacity.

2. **Query**: "Where is the SFPU trunc kernel implementation located in tt-metal?"
   **Reason**: Needed to locate the actual SFPU kernel source files.
   **Key Findings**: The compute API is in `rounding.h` (`trunc_tile` function). The underlying SFPU kernel is `_calculate_trunc_` calling `_trunc_body_()`, found in `ckernel_sfpu_rounding_ops.h` in the tt_llk submodule.

3. **Query**: "How is the trunc SFPU operation implemented in the LLK layer?" (asked to tenstorrent/tt-llk)
   **Reason**: Needed detailed understanding of the SFPU instruction sequence and register usage.
   **Key Findings**: Detailed step-by-step breakdown of `_trunc_body_()` using TTI instructions (SFPLOADI, SFPEXEXP, SFPIADD, SFPSHFT2, SFPENCC, SFPAND) to construct and apply a bitmask that zeroes fractional mantissa bits.

### Confluence References

Not consulted. The DeepWiki and source code provided sufficient detail on all SFPU instructions used.

### Glean References

Not consulted. No confidential hardware specifications were needed beyond what was available in open source.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to verify the macro defines, kernel path selection, and initialization/function strings for the TRUNC operation.
   **Key Information**: TRUNC uses `SFPU_OP_ROUND_FAMILY_INCLUDE`, kernel path defaults to `eltwise_sfpu.cpp`, init is `rounding_op_tile_init()`, function is `trunc_tile(0)`, approx mode is false.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
   **Reason**: Needed to verify how `SFPU_OP_ROUND_FAMILY_INCLUDE` triggers inclusion of `rounding.h`.
   **Key Information**: When `SFPU_OP_ROUND_FAMILY_INCLUDE` is defined to 1, the file conditionally includes `api/compute/eltwise_unary/rounding.h`.

3. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Needed to trace the macro expansion chain from `SFPU_TWO_PARAM_KERNEL` to the actual LLK call.
   **Key Information**: `SFPU_TWO_PARAM_KERNEL(FN, APPROXIMATE, T2, DST_IDX, VECTOR_MODE)` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE, T2>, DST_IDX, VECTOR_MODE)`.
