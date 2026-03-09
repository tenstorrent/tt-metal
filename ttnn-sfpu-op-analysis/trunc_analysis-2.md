# Trunc Implementation Analysis

## Overview

The **trunc** (truncation) operation computes the element-wise truncation of a floating-point tensor toward zero. For each element `x`, `trunc(x)` returns the integer part of `x`, discarding the fractional part. For example, `trunc(3.7) = 3.0` and `trunc(-2.3) = -2.0`.

Trunc is implemented as a standard SFPU unary operation that shares the generic `UnaryProgramFactory` with all other element-wise unary ops. The operation belongs to the "rounding family" of SFPU operations (alongside `floor`, `ceil`, `frac`, and `round`).

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles per block. For the standard factory, `per_core_block_dim = 1`, so the outer loop processes one tile per iteration. |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | Flattened to pages |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | BFLOAT16, FLOAT32, INT32, or UINT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Same as input (or specified output dtype) |

### Layout Transformations

No layout transformations are performed. The operation processes data in-place within its tile layout. The input and output share the same logical and physical layout.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile(c_0, 0, 0)`, SFPU trunc, `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer | CB c_2 | DRAM (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

The compute kernel reserves the output CB once per block (before the inner tile loop), processes individual tiles through copy + SFPU + pack, then pushes all tiles at once after the block completes.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Note: CB c_1 (tmp0) is **not** allocated for trunc. It is only allocated for HARDSHRINK, CBRT, and LOGIT operations.

## Pipeline Pattern Summary

Both input (c_0) and output (c_2) circular buffers use double-buffering (capacity = 2x block size). This allows the reader to fill one buffer slot while compute processes the other, and similarly compute can write to one output slot while the writer drains the other. This creates a three-stage overlapped pipeline: read / compute / write.

## Index Calculations

Index mapping is handled by the `TensorAccessor` abstraction. The reader and writer each receive a `start_id` (starting page index) and iterate sequentially through `num_pages` pages. The `TensorAccessor` translates logical page indices to physical DRAM bank addresses using the buffer's interleaved layout metadata. The compile-time arguments encode the `TensorAccessorArgs` which describe the buffer's banking and addressing scheme.

## Memory Access Patterns

### Read Pattern
Sequential page-by-page reads from DRAM. Each iteration reads one page (one tile) via `noc_async_read_page(i, s, l1_write_addr)`, where `i` increments from `start_id` to `start_id + num_pages`. A `noc_async_read_barrier()` follows each read to ensure completion before the CB is pushed.

### Write Pattern
Sequential page-by-page writes to DRAM. Each iteration writes one page via `noc_async_write_page(i, s, l1_read_addr)`, with `noc_async_writes_flushed()` after each write. A final `noc_async_write_barrier()` at the end ensures all writes complete.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (linearized to 1D for work assignment) |
| **Grid dimensions** | Device `compute_with_storage_grid_size` (e.g., 8x8) |
| **Total cores** | Determined by `split_work_to_cores` |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles |

Cores are indexed as `{i / num_cores_y, i % num_cores_y}`, filling column-first. The `split_work_to_cores` utility divides total pages into two core groups to handle remainders. If tiles divide evenly, all cores are in group 1. Otherwise, group 1 has one extra tile per core compared to group 2.

## Arguments

### Compile-Time Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encoded buffer addressing metadata (bank count, page size, alignment, etc.) for the source buffer |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encoded buffer addressing metadata for the destination buffer |

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tiles) this core must process |
| 1 | per_core_block_dim | uint32_t | Number of tiles per block (always 1 for standard factory) |

### Runtime Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_pages | uint32_t | Number of pages (tiles) for this core to read |
| 2 | start_id | uint32_t | Starting page index for this core |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | num_pages | uint32_t | Number of pages (tiles) for this core to write |
| 2 | start_id | uint32_t | Starting page index for this core |

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for trunc (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for trunc (always 0) |

### Preprocessor Defines

The following defines are passed to the compute kernel at compile time:

| Define | Value | Description |
|--------|-------|-------------|
| `SFPU_OP_ROUND_FAMILY_INCLUDE` | `1` | Enables inclusion of `rounding.h` header |
| `SFPU_OP_CHAIN_0` | `rounding_op_tile_init(); trunc_tile(0);` | The actual SFPU operation chain to execute |
| `SFPU_OP_CHAIN_0_INIT_0` | `rounding_op_tile_init();` | Init function for the rounding family |
| `SFPU_OP_CHAIN_0_FUNC_0` | `trunc_tile(0);` | Compute function call |
| `INP_FLOAT` / `INP_FLOAT32` / `INP_INT32` / `INP_UINT32` | `1` | One of these, based on input dtype |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 (BRISC) | NOC0 | DRAM src_buffer | CB c_0 | Sequential page reads via TensorAccessor |
| compute | RISCV_2 (MATH/TRISC) | N/A | CB c_0 | CB c_2 | copy_tile + SFPU trunc + pack_tile |
| writer | RISCV_1 (NCRISC) | NOC1 | CB c_2 | DRAM dst_buffer | Sequential page writes via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Uses `TensorAccessor` to abstract DRAM bank addressing. Reads one page at a time with a read barrier after each page. Supports a `BACKWARDS` define for reverse iteration (not used by trunc).

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Mirrors the reader structure. Writes one page at a time, calls `noc_async_writes_flushed()` after each page, and issues a final `noc_async_write_barrier()`. Supports an `OUT_SHARDED` define for sharded output (not used by standard factory).

### Compute Kernel

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
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // conditionally includes rounding.h when SFPU_OP_ROUND_FAMILY_INCLUDE=1
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // number of tile blocks to process (equals num_pages_per_core for trunc)
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block (always 1 for standard unary factory)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // initialize SFPU pipeline: configure unpack from c_0 and pack to c_2
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve space in output CB for per_core_block_dim tiles (1 tile)
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DST register file for SFPU computation

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // wait until reader has produced 1 tile in input CB

            copy_tile(tt::CBIndex::c_0, 0, 0);  // unpack tile from c_0 slot 0 into DST register 0

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // expands to: rounding_op_tile_init(); trunc_tile(0);
            // rounding_op_tile_init() -> SFPU_UNARY_KERNEL_INIT(unused, APPROX) -> llk_math_eltwise_unary_sfpu_init<>()
            // trunc_tile(0) -> SFPU_TWO_PARAM_KERNEL(_calculate_trunc_, APPROX, 8, 0, (int)VectorMode::RC)
            //   -> _llk_math_eltwise_unary_sfpu_params_<APPROX>(_calculate_trunc_<APPROX, 8>, 0, VectorMode::RC)
            //   This processes all 4 faces of the tile (32 rows of 32 elements) in DST register 0
#endif

            tile_regs_commit();  // signal that SFPU computation is done, DST is ready for packing

            tile_regs_wait();  // wait for pack pipeline to be ready

            pack_tile(0, tt::CBIndex::c_2);  // pack DST register 0 into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // free the consumed input tile from c_0

            tile_regs_release();  // release DST register file for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // push completed output tiles to writer
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File

**Blackhole**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_rounding_ops.h`
**Wormhole B0**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_rounding_ops.h`

Both architectures share the identical `_trunc_body_()` implementation. The only difference is the `ADDR_MOD` used in the `_calculate_trunc_` wrapper: Blackhole uses `ADDR_MOD_7`, Wormhole uses `ADDR_MOD_3`.

#### Annotated SFPU Kernel Source (Blackhole variant)

```cpp
namespace ckernel
{
namespace sfpu
{

// computes L1=trunc(L0).
// This function truncates a float toward zero by zeroing out the fractional mantissa bits.
// It works by constructing a bitmask that preserves only the integer portion of the IEEE 754
// float and then ANDing the input with that mask.
inline void _trunc_body_()
{
    // Load the constant 23 into LREG3. This is the number of mantissa bits in IEEE 754 single-precision.
    // Will be used to compute how many mantissa bits represent the fractional part.
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_SHORT, 23);

    // Load mask = 0x8000_0000 into LREG1.
    // SFPLOADI_MOD0_FLOATB interprets the 16-bit immediate as BF16, so 0x8000 -> 0x80000000 in FP32,
    // which is negative zero. Treated as a bitmask, this is the sign bit only.
    // This serves as the initial mask value for lanes where the exponent is negative (|x| < 1),
    // ensuring those lanes produce 0 (with sign preserved by the AND with original value).
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);

    // Extract the unbiased exponent from LREG0 (the input value) into LREG2.
    // The modifiers SET_CC_SGN_EXP | SET_CC_COMP_EXP set condition codes such that:
    //   - Lanes where the exponent < 0 (i.e., |x| < 1.0) are disabled.
    // This means subsequent instructions only execute on lanes where |x| >= 1.0.
    // For disabled lanes, LREG1 retains 0x80000000 (sign-bit-only mask).
    TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);

    // For lanes where |x| >= 1.0 (condition met), load mask = 0xFFFF_FFFF into LREG1.
    // SFPLOADI_MOD0_SHORT with 0xFFFF loads 0x0000FFFF, but this is sign-extended to 0xFFFFFFFF
    // as a 32-bit integer (short immediate is treated as signed 16-bit, -1 sign-extends).
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);

    // Compute (23 - exponent) in LREG2. This gives the number of fractional mantissa bits.
    // SFPIADD_MOD1_ARG_2SCOMP_LREG_DST means: LREG2 = LREG3 + (-LREG2) = 23 - exp.
    // SFPIADD_MOD1_CC_GTE0: only executes on lanes where the condition code allows (exp >= 0 lanes).
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);

    // Left-shift the all-ones mask (LREG1=0xFFFFFFFF) by (23 - exp) bits (from LREG2).
    // SFPSHFT2_MOD1_SHFT_LREG: shift LREG1 left by the value in LREG2.
    // Result: a mask like 0xFFFFF000 that has 1s in the sign+exponent+integer mantissa bits
    // and 0s in the fractional mantissa bits.
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SHFT_LREG);

    // Reset condition codes (re-enable all lanes).
    TTI_SFPENCC(0, 0, 0, 0);

    // AND the input (LREG0) with the mask (LREG1), storing the result in LREG1.
    // For |x| >= 1: zeros out fractional mantissa bits, keeping the integer part.
    // For |x| < 1: mask is 0x80000000, so result is +/-0.0 (preserving sign).
    TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
}

// The top-level trunc kernel function. Called once per tile face (4 faces per tile in RC mode).
// ITERATIONS=8 means it processes 8 rows per face (32 elements per row, 8 rows = 256 elements = one 16x16 face).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_trunc_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Load one row (32 elements) from DST register into LREG0.
        // ADDR_MOD_7 configures the address modifier for auto-incrementing DST row pointer.
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);

        // Perform truncation: input in LREG0, result in LREG1.
        _trunc_body_();

        // Store the truncated result from LREG1 back to DST register.
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);

        // Advance the DST register row pointer for the next iteration.
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOADI` | Load an immediate value into an LREG. Used to load constants (23 for mantissa bit count, 0x80000000 sign-bit mask, 0xFFFFFFFF all-ones mask). Supports multiple format modes: `MOD0_SHORT` for 16-bit sign-extended integer, `MOD0_FLOATB` for BF16-to-FP32 conversion. |
| `SFPEXEXP` | Extract the unbiased exponent from a floating-point value in an LREG. With `SET_CC_SGN_EXP | SET_CC_COMP_EXP` modifiers, it also sets per-lane condition codes to disable lanes where the exponent is negative (|x| < 1.0). |
| `SFPIADD` | Integer addition/subtraction on LREGs. With `ARG_2SCOMP_LREG_DST` modifier, computes `LREG3 - LREG2` (i.e., `23 - exp`). The `CC_GTE0` modifier restricts execution to conditionally enabled lanes. |
| `SFPSHFT2` | Bitwise shift. With `MOD1_SHFT_LREG`, performs `LREG1 <<= LREG2`, shifting the all-ones mask left by `(23 - exp)` to create a mask that zeros fractional bits. |
| `SFPENCC` | Enable/disable conditional execution. Called with all-zero arguments to reset condition codes, re-enabling all lanes for subsequent instructions. |
| `SFPAND` | Bitwise AND between two LREGs. Applies the constructed mask to the input, zeroing fractional mantissa bits. |
| `SFPLOAD` | Load data from DST register file into an LREG. Transfers one row of 32 elements from the current DST position. |
| `SFPSTORE` | Store data from an LREG back to the DST register file. Writes the truncated result back to DST. |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Holds the input value loaded from DST. Preserved throughout `_trunc_body_` for the final AND operation. |
| **LREG1** | Holds the bitmask. Initialized to `0x80000000` (for |x| < 1 lanes), then overwritten with `0xFFFFFFFF` (for |x| >= 1 lanes), then shifted left. After `SFPAND`, holds the final truncated result. |
| **LREG2** | Holds the extracted exponent from `SFPEXEXP`, then used for the shift amount `(23 - exp)` after `SFPIADD`. |
| **LREG3** | Holds the constant `23` (number of mantissa bits in IEEE 754 single-precision). |
| **DST** | The destination register file shared between unpack, SFPU, and pack stages. One tile's worth of data resides here; SFPU loads/stores rows from/to it. |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `cb_wait_front(c_0, 1)` to wait for an input tile, then `copy_tile(c_0, 0, 0)` to unpack the tile from CB c_0 into DST register 0.

2. **SFPU initialization**: `rounding_op_tile_init()` calls `SFPU_UNARY_KERNEL_INIT(unused, APPROX)` which invokes `llk_math_eltwise_unary_sfpu_init<>()` to configure the SFPU pipeline.

3. **SFPU dispatch**: `trunc_tile(0)` expands to `SFPU_TWO_PARAM_KERNEL(_calculate_trunc_, APPROX, 8, 0, (int)VectorMode::RC)`, which calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>`. This function:
   - Calls `_llk_math_eltwise_unary_sfpu_start_` to set up the DST address for tile index 0.
   - Loops over all 4 faces of the tile (VectorMode::RC processes rows and columns).
   - For each face, calls `_calculate_trunc_<APPROX, 8>()`.
   - After all faces, calls `_llk_math_eltwise_unary_sfpu_done_()`.

4. **Per-face processing** (`_calculate_trunc_`): Iterates 8 times (8 rows per face, each row containing 32 elements):
   - `SFPLOAD`: Load a row from DST into LREG0.
   - `_trunc_body_()`: Compute truncation via bitmask construction and AND (see annotated source above).
   - `SFPSTORE`: Write truncated result from LREG1 back to DST.
   - `dst_reg++`: Advance to next row.

5. **Result packing**: After the SFPU completes, `tile_regs_commit()` signals readiness, `tile_regs_wait()` waits for the pack engine, and `pack_tile(0, c_2)` moves the result from DST to the output circular buffer.

#### SFPU Configuration

- **Math fidelity**: `MathFidelity::HiFi4` (highest fidelity, though this does not significantly affect SFPU integer/bitwise operations).
- **Math approx mode**: `false` (trunc is not in the list of operations that return `true` from `get_op_approx_mode`). The `APPROXIMATION_MODE` template parameter is `false`.
- **fp32_dest_acc_en**: Configurable via operation parameters; when enabled, DST registers hold full FP32 values.
- **preserve_fp32_precision**: When set, configures `UnpackToDestFp32` mode for CB c_0 to preserve FP32 precision during unpack-to-DST.
- **Unpack mode**: Default (`UnpackToDestMode::Default`) unless `preserve_fp32_precision` is set.

#### Hardware Compatibility Notes

The `_trunc_body_()` implementation is **identical** on both Wormhole B0 and Blackhole. The only difference is in the `_calculate_trunc_` wrapper:
- **Blackhole** uses `ADDR_MOD_7` for SFPLOAD/SFPSTORE address modifier.
- **Wormhole B0** uses `ADDR_MOD_3` for SFPLOAD/SFPSTORE address modifier.

These address modifiers control auto-increment behavior of the DST register pointer; the specific modifier number is architecture-dependent but functionally equivalent.

Note that the `_floor_body_()` and `_ceil_body_()` implementations differ between architectures: Blackhole uses `SFPGT` instructions for comparison, while Wormhole B0 (which lacks `SFPGT`) uses `SFPSETCC` + `SFPIADD` as a workaround. However, `_trunc_body_()` does not require comparison instructions and is therefore identical across both architectures.

## Implementation Notes

1. **Bit-manipulation approach**: The trunc operation is implemented entirely through integer/bitwise SFPU instructions rather than floating-point arithmetic. This avoids rounding issues and is exact for all representable floats.

2. **Algorithm correctness**: For IEEE 754 single-precision floats:
   - If `|x| < 1.0` (exponent < 0): the mask is `0x80000000`, so the result is `+/-0.0`.
   - If `|x| >= 1.0` and `exponent < 23`: a mask with `(23 - exp)` trailing zeros is constructed to zero out fractional mantissa bits.
   - If `exponent >= 23`: the value is already an integer (no fractional bits exist), and the shift amount would be 0 or negative. The `SFPSHFT2` with shift amount 0 produces an all-ones mask, preserving the value unchanged. For very large exponents, the behavior is still correct since there are no fractional bits to remove.

3. **Condition code usage**: The `SFPEXEXP` instruction with `SET_CC_SGN_EXP | SET_CC_COMP_EXP` disables lanes where the exponent is negative. This means for `|x| < 1.0`, the `SFPLOADI` (0xFFFF) and `SFPIADD` and `SFPSHFT2` instructions are skipped, leaving the mask at `0x80000000`. The `SFPENCC` resets all lanes before the final `SFPAND`, so the AND applies to all lanes.

4. **Foundation for other rounding ops**: `_trunc_body_()` is the building block for `floor`, `ceil`, and `frac` operations. Floor adds a subtract-one step for negative non-integer values; ceil adds an add-one step for positive non-integer values; frac computes `x - trunc(x)`.

5. **No scalar parameters**: Unlike some unary ops (e.g., `hardshrink` with a threshold), trunc requires no scalar parameters. The `packed_scalar1` and `packed_scalar2` runtime arguments are always 0.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the unary program factory implemented for SFPU eltwise unary operations?"
   **Reason**: Needed to understand the overall program factory structure, kernel file selection, and core distribution strategy.
   **Key Findings**: The `UnaryProgramFactory` uses `split_work_to_cores` for distributing tiles across cores, creates reader/compute/writer kernels, and uses `get_compute_kernel_path` to select the appropriate compute kernel. For trunc, this returns the generic `eltwise_sfpu.cpp`.

2. **Query**: "How is the trunc (truncation) SFPU operation implemented in tt-metal?"
   **Reason**: Needed to locate the specific SFPU kernel files and understand the LLK function chain.
   **Key Findings**: Trunc uses `_calculate_trunc_` from `ckernel_sfpu_rounding_ops.h`, dispatched via `SFPU_TWO_PARAM_KERNEL` macro. The core logic is in `_trunc_body_()` which constructs a bitmask to zero fractional mantissa bits.

3. **Query**: "What do the SFPU instructions SFPLOAD, SFPSTORE, SFPLOADI, SFPEXEXP, SFPIADD, SFPSHFT2, SFPENCC, SFPAND do?"
   **Reason**: Needed detailed specifications of each SFPU instruction used in the trunc kernel to provide accurate annotations.
   **Key Findings**: Obtained descriptions of all instruction operands, modifier flags, and semantics. SFPEXEXP extracts exponents and can set condition codes; SFPIADD does integer add/subtract with conditional execution; SFPSHFT2 does bitwise shifts; SFPENCC controls lane predication; SFPAND performs bitwise AND.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to trace how `TRUNC` maps to its include macro (`SFPU_OP_ROUND_FAMILY_INCLUDE`), init/func strings (`rounding_op_tile_init()` / `trunc_tile(0)`), compute kernel path (`eltwise_sfpu.cpp`), and approx mode (`false`).
   **Key Information**: TRUNC belongs to the rounding family. Its defines produce the SFPU_OP_CHAIN that expands inside `eltwise_sfpu.cpp`.

2. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Needed to understand the macro expansion chain from `trunc_tile` to the actual SFPU kernel invocation.
   **Key Information**: `SFPU_TWO_PARAM_KERNEL` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE, T2>, DST_IDX, VECTOR_MODE)`.

3. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Needed to understand how the params function dispatches across tile faces.
   **Key Information**: In `VectorMode::RC`, the function iterates over all 4 faces, calling the SFPU kernel function once per face, with `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` between faces.
