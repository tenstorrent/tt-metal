# TYPECAST (Interleaved) - SFPU Operation Analysis

## Operation Overview

**Operation**: `typecast` (interleaved variant)
**Category**: `copy/typecast`
**Namespace**: `ttnn::prim`
**Program Factory**: `TypecastProgramFactory` (default interleaved path)

The typecast operation converts tensor data between different numeric formats on Tenstorrent hardware. It is a unary SFPU operation that processes tiles one at a time, loading data from an input circular buffer, applying format conversion in the SFPU, and packing the result to an output circular buffer.

Unlike most SFPU operations that apply a single mathematical function, typecast is a **polymorphic dispatcher**: the program factory injects compile-time defines (`TYPECAST_LLK_INIT` and `TYPECAST_LLK`) that expand to `typecast_tile_init<IN_DTYPE, OUT_DTYPE>()` and `typecast_tile<IN_DTYPE, OUT_DTYPE>(0)` respectively, with the actual DataFormat enum values baked in. This means the specific SFPU kernel instantiated depends entirely on the input/output dtype pair.

**Supported Conversions**:
- Float16_b <-> Float32, Int32, UInt16, UInt32, UInt8
- Float32 <-> Int32, UInt16, UInt32, UInt8
- Bfp8_b <-> Int32, UInt16, UInt32, UInt8, Float16_b, Float32
- Bfp4_b <-> Int32, UInt16, UInt32, UInt8, Bfp8_b, Float16_b, Float32
- UInt16 <-> UInt32, Int32, UInt8

Some conversions (e.g., Float16_b -> Float32, Bfp8_b -> Float16_b) are handled entirely by the unpacker/packer hardware and require no SFPU kernel at all.

**Key Design Decision**: The operation supports both TILE and ROW_MAJOR layouts. For ROW_MAJOR, pages are individual rows (including padding) rather than 32x32 tiles. The `per_core_block_dim` is always 1 regardless of layout, meaning tiles/rows are processed one at a time.

---

## Program Factory Analysis

### File Location
`ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp`

### Program Factory Variants
The file contains two program factories:
1. **`TypecastProgramFactory`** - Standard interleaved variant (this analysis)
2. **`TypecastSubgridProgramFactory`** - Subgrid variant for custom core grids

### Operation Parameters

```cpp
struct TypecastParams {
    const DataType input_dtype;          // Source data type
    const DataType output_dtype;         // Target data type
    const MemoryConfig output_memory_config;
    const bool fp32_dest_acc_en = false;       // Enable 32-bit dest accumulator
    const bool preserve_fp32_precision = false; // UnpackToDestFp32 mode
    const bool bfp8_pack_precise = false;       // Precise BFP8 packing
    const std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
};
```

### Work Distribution

Work is distributed across all available compute cores using `split_work_to_cores`:
- `num_pages` = total number of tiles (TILE layout) or rows (ROW_MAJOR layout) in the input tensor
- Two core groups handle uneven division: `core_group_1` gets `num_items_per_core_group_1` pages, `core_group_2` gets `num_items_per_core_group_2` pages
- Core ordering is row-wise for ROW_MAJOR layout, column-wise for TILE layout

### Circular Buffer Configuration

| CB Index | Name | Format | Pages | Purpose |
|----------|------|--------|-------|---------|
| `c_0` (0) | `src0_cb` | Input data format | 2 | Input buffer - double buffered |
| `c_2` (2) | `output_cb` | Output data format | 2 | Output buffer - double buffered |

**Design Note**: The CB page size depends on layout:
- TILE layout: `tile_size(cb_data_format)` (derived from format, e.g., 2048 bytes for Float16_b)
- ROW_MAJOR layout: `buffer->page_size()` (actual row size including padding)

Both CBs use only 2 pages (double buffering), which is the minimum needed for overlapping compute with data movement. This is appropriate because typecast processes one tile/row at a time.

### Compile-Time Defines

The program factory injects two critical defines into the compute kernel:

```cpp
unary_defines["TYPECAST_LLK_INIT"] = fmt::format(
    "typecast_tile_init<{0}u, {1}u>",
    static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
    static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));

unary_defines["TYPECAST_LLK"] = fmt::format(
    "typecast_tile<{0}u, {1}u>",
    static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
    static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));
```

This approach bakes the DataFormat enum values (as unsigned integers) into the template parameters at compile time. The LLK layer then uses a large `if constexpr` dispatch table to select the correct SFPU kernel.

### Compute Configuration

```cpp
ComputeConfig{
    .math_fidelity = MathFidelity::HiFi4,
    .fp32_dest_acc_en = args.fp32_dest_acc_en,
    .unpack_to_dest_mode = unpack_to_dest_mode,  // UnpackToDestFp32 if preserve_fp32_precision
    .bfp8_pack_precise = args.bfp8_pack_precise,
    .math_approx_mode = false,                    // Always exact
    .compile_args = compute_kernel_args,
    .defines = unary_defines
}
```

**Key observations**:
- `math_approx_mode` is always `false` -- typecast requires exact conversion, not approximation
- `MathFidelity::HiFi4` is used for maximum precision
- `fp32_dest_acc_en` enables 32-bit destination registers when converting to/from 32-bit types
- `preserve_fp32_precision` enables `UnpackToDestFp32` mode on `src0_cb`, which unpacks directly to destination registers in FP32 format

### Runtime Arguments

**Reader kernel**: `{src_buffer_address, num_items_per_core, start_id}`
**Writer kernel**: `{dst_buffer_address, num_items_per_core, start_id}`

The `start_id` is accumulated across cores, giving each core a contiguous range of pages to process.

### Program Caching

The `override_runtime_arguments` function only updates buffer addresses (arg[0]) for both reader and writer kernels. This means the program can be reused when:
- The same dtype conversion is needed
- Only the buffer addresses change (e.g., different tensor instances with same shape/dtype)

---

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

This is a shared reader kernel used across many unary operations. It reads pages from DRAM using the NoC, one page at a time:

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);   // DRAM buffer address
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of pages this core processes
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting page index

    constexpr auto src_args = TensorAccessorArgs<0>();      // Compile-time accessor args from program factory

    constexpr uint32_t cb_id_in0 = 0;                      // Always reads into CB 0

    // Page size is determined at runtime from the CB config, supporting both TILE and ROW_MAJOR
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    constexpr uint32_t onepage = 1;                        // Process one page at a time

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    // Sequential page read loop with NoC barrier per page
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_id_in0, onepage);               // Wait for space in input CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0); // Get L1 write address
        noc_async_read_page(i, s, l1_write_addr);          // Issue NoC read from DRAM
        noc_async_read_barrier();                           // Wait for read to complete
        cb_push_back(cb_id_in0, onepage);                  // Signal page is ready for compute
    }
}
```

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

Shared writer kernel that writes output pages to DRAM:

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);    // DRAM output buffer address
    const uint32_t num_pages = get_arg_val<uint32_t>(1);    // Number of pages this core writes
    const uint32_t start_id = get_arg_val<uint32_t>(2);     // Starting page index

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0); // Output CB index (c_2 = 2)
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_id_out, onepage);                  // Wait for compute to produce a page
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);    // Get L1 read address
        noc_async_write_page(i, s, l1_read_addr);           // Issue NoC write to DRAM
        noc_async_writes_flushed();                         // Flush write (non-blocking)
        cb_pop_front(cb_id_out, onepage);                   // Free the CB page
    }
    noc_async_write_barrier();                              // Final barrier to ensure all writes complete
}
```

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"                // Provides typecast_tile<> and typecast_tile_init<>

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0); // Number of blocks (tiles/rows) to process
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1); // Always 1 for typecast
    constexpr uint32_t input_cb = get_compile_time_arg_val(2);           // CB index 0 (c_0)
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);          // CB index 2 (c_2)

    // Initialize the SFPU for the specific typecast conversion
    // This configures SFPLOADMACRO instruction templates, store modes, and constants
    init_sfpu(input_cb, output_cb);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve space in the output CB before processing
        // Since per_core_block_dim=1, this reserves 1 page
        cb_reserve_back(output_cb, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to the destination register file
            // This is the shared register space between math (SFPU/FPU) and pack RISC-V cores
            tile_regs_acquire();

            // Wait for one page to be available in the input CB (written by reader kernel)
            cb_wait_front(input_cb, 1);

            // Unpack the tile from input CB into destination registers
            // copy_tile uses the unpacker to move data from L1 (CB) into DEST registers
            // The unpack format is determined by the CB's data format
            copy_tile(input_cb, 0, 0);

            // Initialize the SFPU for this specific typecast direction
            // This is the macro-expanded TYPECAST_LLK_INIT, e.g.:
            //   typecast_tile_init<Float16_b_format, UInt16_format>()
            // It configures SFPLOADMACRO templates, store format, and rounding modes
            TYPECAST_LLK_INIT();

            // Execute the actual typecast SFPU operation on tile at dest index 0
            // This is the macro-expanded TYPECAST_LLK, e.g.:
            //   typecast_tile<Float16_b_format, UInt16_format>(0)
            // It runs the appropriate SFPU kernel on all 1024 elements (32x32) of the tile
            TYPECAST_LLK(0);

            // Release math registers to the packer
            tile_regs_commit();

            // Wait for the packer to signal it is ready
            tile_regs_wait();

            // Pack the result from DEST registers into the output CB
            // The pack format is determined by the output CB's data format
            pack_tile(0, output_cb);

            // Free the input page in the input CB
            cb_pop_front(input_cb, 1);

            // Release the tile register lock so the math core can use it again
            tile_regs_release();
        }
        // Push the processed page(s) to the output CB for the writer kernel
        cb_push_back(output_cb, per_core_block_dim);
    }
}
```

**Why `per_core_block_dim` is always 1**: Typecast processes each tile independently with no cross-tile dependencies. Setting block_dim=1 means each iteration of the outer loop processes exactly one tile, which simplifies the pipeline and allows the double-buffered CBs to overlap compute with data movement effectively.

**Why `TYPECAST_LLK_INIT` is called inside the inner loop**: Unlike most SFPU operations where init is called once, typecast calls init on every tile. This is because `copy_tile` (which calls `copy_tile_to_dst_init_short`) may reconfigure the unpacker state, and the typecast init needs to restore SFPU configuration (SFPLOADMACRO templates, store modes) after each unpack operation. This is a correctness requirement, not a performance optimization.

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to.

#### SFPU Kernel Files

**LLK dispatch layer** (selects kernel based on dtype pair):
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h`

**Architecture-specific wrappers**:
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h`

**Shared SFPU implementations** (in tt_llk submodule):
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h`

#### Annotated SFPU Kernel Source

The shared implementation file contains all the `_calculate_typecast_*` and `_init_typecast_*` functions. Below is the complete annotated source:

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// ===========================================================================
// FP32 -> UInt16
// Uses SFPLOADMACRO for 2-cycle throughput per row.
// Pipeline: Load -> max(v,0) clamp -> SFPSTOCHRND to uint16 -> Store LO16
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint16_()
{
    // SFPLOADMACRO achieves high throughput by scheduling Load/Simple/MAD/Round/Store
    // sub-units across pipeline stages. [v] notation means scheduled by SFPLOADMACRO with VD=v.
    //
    // Pipeline diagram (2 cycles per row):
    // t=0: Load[v]
    // t=1: NOP, Simple[v]=max(v,0.0)  -- clamp negative values to zero
    // t=0: ...idle...
    // t=1: Round[v] L16=rnd(v)         -- SFPSTOCHRND converts FP32 to uint16, clamping to 65535
    // t=0: Store[v] L16                -- write low-16 result back to DEST

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1; // Alternate between LREG0 and LREG1 for pipelining
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, v >> 2);
        TTI_SFPNOP; // Required NOP for pipeline timing
    }
    // Drain pipeline
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ===========================================================================
// UInt16 -> Float16_b (BFloat16)
// Uses SFPLOADMACRO for 1-cycle throughput per row.
// Pipeline: Load LO16 -> SFPCAST uint->fp32 -> SFPSTOCHRND fp32->fp16b -> Store
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp16b_()
{
    // 1-cycle throughput: the simplest conversion pipeline
    // Load with LO16 mode zero-extends uint16 to 32 bits, then:
    // Simple: cast(v) converts unsigned int to FP32
    // Round: SFPSTOCHRND converts FP32 to FP16B (truncate mantissa to 7 bits)
    // Store: write FP16B result

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_2, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ===========================================================================
// Int32 -> Float16_b
// Uses SFPLOADMACRO for 4-cycle throughput per row.
// Handles signed integers: extracts sign, converts absolute value, restores sign.
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_()
{
    // Strategy for signed int32:
    // 1. Load as INT32
    // 2. Take absolute value (SFPABS)
    // 3. Store sign bit in LREG7 via shift right 31
    // 4. Cast absolute value to FP32 (SFPCAST)
    // 5. Use SFPMAD with indirect VA (LREG[LREG7]) to handle sign:
    //    - LREG0 = 0.0, LREG1 = -2^31
    //    - If sign=0: 0.0*1.0 + v = v (positive case)
    //    - If sign=1: -2^31*1.0 + v (but abs(v)=0 for -2^31, so result = -2^31)
    // 6. SFPSTOCHRND converts FP32 to FP16B
    //
    // The trick with LREG7 indexing avoids conditional branches entirely.

    constexpr int t = p_sfpu::LREG4; // Temporary register for abs value

    // Load constants: LREG0 = 0.0 (for positive values), LREG1 = -2^31 (for INT_MIN)
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00); // -2^31 in bfloat16

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // Alternate LREG2/LREG3 (LREG0/1 hold constants)
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPABS(0, v, t, 0);                                        // t = |v|
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5);          // LREG7 = t >> 31 (sign bit)
        TTI_SFPCAST(t, t, 0);                                         // t = (float)t
    }
    // Drain: 5 NOPs for the deep pipeline (load+abs+shift+cast+mad+round+store)
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ===========================================================================
// FP32 -> Int32 (SFPI implementation)
// Converts IEEE 754 single-precision float to two's complement int32.
// Uses SFPI (C++ style) rather than TTI macros.
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_int32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0]; // Load FP32 value from DEST

        // Extract debiased exponent: for 1.0, exp=0; for 2.0, exp=1; etc.
        sfpi::vInt exp = sfpi::exexp(in);

        // Extract mantissa with implicit 1 at bit 23 (24-bit integer)
        sfpi::vUInt man = sfpi::exman8(in);

        // The mantissa from exman8 has the form 1.mmmmmmm * 2^23 (as integer)
        // To get the true integer value, shift by (exp - 23):
        //   exp=0  -> shift right by 23 -> result ~ 1
        //   exp=23 -> no shift          -> result ~ 2^23
        //   exp=30 -> shift left by 7   -> result ~ 2^30
        sfpi::vInt shift_amt = exp - 23;
        sfpi::vInt result = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(man, shift_amt));

        // Overflow: |value| >= 2^31 cannot fit in int32, clamp to INT_MIN
        v_if (exp >= 31) {
            result = 0x80000000;  // INT_MIN (-2147483648)
        }
        v_endif;

        // Underflow: |value| < 1 rounds to zero
        v_if (exp < 0) {
            result = 0;
        }
        v_endif;

        // Apply two's complement negation for negative inputs: -x = ~x + 1
        v_if (in < 0.0f) {
            result = ~result + 1;
        }
        v_endif;

        sfpi::dst_reg[0] = result; // Store int32 result back to DEST
        sfpi::dst_reg++;           // Advance to next row (32 elements)
    }
}

// ===========================================================================
// FP32 -> UInt32
// Extracts exponent and mantissa, shifts to integer, handles overflow/underflow.
// Uses TTI macros for maximum throughput.
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);          // result = 0

        // LaneEnabled = (in >= 0): skip negative values (they stay 0)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
        // Extract exponent, set condition code if exp >= 0
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // result = 0xFFFFFFFF (saturated max for uint32)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
        // exp -= 32, LaneEnabled = (exp < 32) -- values >= 2^32 stay at 0xFFFFFFFF
        TTI_SFPIADD(-32 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        TTI_SFPIADD(9, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // Extract mantissa and shift to get integer magnitude
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPENCC(0, 0, 0, 0); // LaneEnabled = true

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

// ===========================================================================
// FP32 -> Float16_b
// Manual truncation with round-to-nearest-even via SFPLOADMACRO.
// 3-cycle throughput per row.
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_fp16b_()
{
    // This performs precise FP32->BF16 conversion by:
    // 1. Loading two copies of the value (a for LSB extraction, b for the value)
    // 2. Shifting a right by 16 to get the rounding bit
    // 3. Masking a to get just bit 0 (the LSB of the BF16 result)
    // 4. Adding 0x7FFF to b (round-to-nearest bias)
    // 5. Adding a (the LSB) to b for round-to-nearest-even
    // 6. Storing b as FP16B (upper 16 bits of the 32-bit result)
    //
    // The net effect: if the truncated bits are exactly 0x8000 (halfway),
    // the round-to-even tie-breaking is correct.

    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (a & 3), 0, ADDR_MOD_3, a >> 2);        // Load a
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), 0, ADDR_MOD_2, b >> 2);       // Load b
        TT_SFPAND(0, p_sfpu::LREG12, a, 0);                                 // a &= 1 (extract LSB)
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ===========================================================================
// UInt16 -> FP32
// 1-cycle throughput per row. Load as LO16, cast to FP32, store as FP32.
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp32_()
{
    constexpr int v = p_sfpu::LREG0;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Load with LO16: zero-extends the uint16 value to 32 bits in LREG
        // SFPLOADMACRO then schedules: SFPCAST (uint->fp32), store as FP32
        TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_2, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ===========================================================================
// Int32 -> FP32
// 4-cycle throughput per row. Same sign-handling strategy as int32_to_fp16b.
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp32_()
{
    // Same strategy as int32_to_fp16b:
    // abs(v) -> store sign in LREG7 -> cast to fp32 -> use SFPMAD with indirect VA
    // Difference: stores as FP32 instead of applying SFPSTOCHRND for FP16B

    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);             // 0.0
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00);        // -2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1);
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5);
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ===========================================================================
// UInt32 -> Float16_b
// 3-cycle throughput. Handles the sign bit (bit 31) of unsigned value as magnitude.
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp16b_()
{
    // For uint32, bit 31 cannot be represented directly as positive FP32 mantissa.
    // Strategy:
    // 1. Store bit 31 in LREG7
    // 2. Clear the sign bit (setsgn to 0) -- now we have a 31-bit unsigned value
    // 3. Cast to FP32 (this handles bits 0-30)
    // 4. Use SFPMAD with indirect VA:
    //    - LREG0=0.0, LREG1=2^31
    //    - If bit31=0: 0.0*1.0 + v = v
    //    - If bit31=1: 2^31*1.0 + v = 2^31 + v (adds back the high bit)
    // 5. SFPSTOCHRND to FP16B

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);             // 0.0
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);        // +2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1);
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPSHFT2(v, p_sfpu::LREG12, p_sfpu::LREG7, 5);                  // LREG7 = v >> 31
        TT_SFPSETSGN(0, v, v, 1);                                            // v = setsgn(v, 0) = clear sign
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ===========================================================================
// UInt32 -> FP32
// Same strategy as uint32_to_fp16b but stores as FP32 instead of FP16B.
// Uses 3 SFPLOADMACRO calls per iteration for complex pipeline scheduling.
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp32_()
{
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);        // +2^31

    constexpr int a  = p_sfpu::LREG2;
    constexpr int b  = p_sfpu::LREG3;
    constexpr int L7 = p_sfpu::LREG7;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Three macro calls per iteration:
        // Macro 0 [a]: loads value, schedules setsgn(a, 0) to clear sign
        // Macro 1 [b]: loads into b, schedules SFPCAST on a's cleared value
        // Macro 2 [L7]: stores sign bit, schedules SFPMAD and final store
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_3, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_3, b >> 2);
        TTI_SFPLOADMACRO((2 << 2) | (L7 & 3), InstrModLoadStore::INT32, ADDR_MOD_2, L7 >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ===========================================================================
// UInt16 -> UInt32
// 1-cycle throughput. Simply loads LO16 (zero-extends) and stores as INT32.
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_uint32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Load with LO16 zero-extends uint16 to 32 bits, then stores as INT32
        TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::LO16, ADDR_MOD_2, 0);
    }
    TTI_SFPNOP;
}

// ===========================================================================
// UInt32 -> UInt16
// 2-cycle throughput. Extracts low 16 bits by loading high/low and combining.
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_uint16_()
{
    // Strategy to extract low 16 bits:
    // 1. Load as LO16 -- this gives the HIGH 16 bits (shifted to low position)
    // 2. Negate it: a = 0 - a (so a = -high16)
    // 3. Shift right by 16 to align
    // 4. Load full INT32 value into b
    // 5. OR a and b: b | a combines low16 with the negated-shifted high16
    //    The OR with the negated high bits effectively masks to low 16 bits
    // 6. Store as LO16

    constexpr int a = p_sfpu::LREG0;
    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 9
    for (int d = 0; d < ITERATIONS + 1; d++)
    {
        int a          = d & 1;
        int macroIndex = 1 + (d & 1);
        if (d < ITERATIONS)
        {
            TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::LO16, ADDR_MOD_2, a >> 2);
        }
        else
        {
            TTI_SFPNOP; // Final iteration: no more loads
        }
        if (d == 0)
        {
            TTI_SFPNOP; // First iteration: pipeline not yet filled
        }
        else if (d < ITERATIONS)
        {
            TTI_SFPLOADMACRO((macroIndex << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_3, (-4 & 0x3ff) | (b >> 2));
        }
        else
        {
            TTI_SFPLOADMACRO((macroIndex << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_2, (-2 & 0x3ff) | (b >> 2));
        }
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ===========================================================================
// Int32 -> UInt16
// 3-cycle throughput. Casts to FP32 first, then clamps to [0, 65535] via SFPSTOCHRND.
// ===========================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_uint16_()
{
    // Strategy:
    // 1. Load as INT32
    // 2. SFPCAST: convert sign-magnitude int to FP32
    // 3. SFPSWAP with 0.0: max(0.0, v) -- clamp negative values to zero
    // 4. SFPSTOCHRND FP32_TO_UINT16: convert FP32 to uint16, clamping to 65535

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_2, a >> 2);
        TT_SFPCAST(a, a, 0);     // Cast int32 to FP32
        TTI_SFPNOP;              // Pipeline timing requirement
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ===========================================================================
// Initialization functions
// These configure SFPLOADMACRO instruction templates, store modes, and constants
// for the corresponding conversion kernels.
// ===========================================================================

template <bool APPROXIMATION_MODE>
inline void _init_typecast_fp32_to_fp16b_()
{
    constexpr int b = p_sfpu::LREG2;

    sfpi::vConstIntPrgm0 = 1;       // Used for LSB mask
    sfpi::vConstIntPrgm1 = 0x7fff;  // Round-to-nearest bias

    // InstructionTemplate[0]: shift right by 16 (extract upper 16 bits position)
    TTI_SFPSHFT2(-16 & 0xfff, 0, 12, 6);

    // InstructionTemplate[1]: add vConstIntPrgm1 (0x7FFF rounding bias)
    TTI_SFPIADD(0, p_sfpu::LREG13, 13, sfpi::SFPIADD_MOD1_CC_NONE);

    // InstructionTemplate[2]: add b (combined LSB + bias)
    TTI_SFPIADD(0, b, 14, sfpi::SFPIADD_MOD1_CC_NONE);

    // Configure Macro 0 [a]: schedules AND for LSB extraction, shift, store
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x40 | (3 << 3) | (4 + 2);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t store_bits  = 0x00 | 0x00 | (3 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Configure Macro 1 [b]: schedules add and FP16B store
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (1 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (3 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // StoreMod0=FP16B, configure store format and pipeline delays
    TTI_SFPCONFIG(0x310 | InstrModLoadStore::FP16B, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint16_to_uint32_()
{
    // Simplest init: just load and store with INT32 format
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x00 | (0 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // StoreMod0=INT32
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint32_to_fp32_()
{
    sfpi::vConstIntPrgm0 = -31; // Used for exponent comparison

    constexpr int a = p_sfpu::LREG2;

    // InstructionTemplate[0]: SFPSETSGN with immediate 0 (clear sign bit)
    TTI_SFPSETSGN(0, 0, 12, 1);
    // InstructionTemplate[1]: SFPCAST from sign-magnitude int to FP32
    TTI_SFPCAST(a, 13, 0);
    // InstructionTemplate[2]: SFPSHFT2 to extract sign bit into LREG7
    TTI_SFPSHFT2(0, p_sfpu::LREG12, 14, 5);
    // InstructionTemplate[3]: SFPMAD with indirect VA for sign correction
    TTI_SFPMAD(0, p_sfpu::LCONST_1, 0, 15, 4);

    // Macro 0 [a]: initial load
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0;
        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 0, 1);
    }
    // Macro 1 [b]: cast + MAD
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits    = 0x00 | 0x40 | (2 << 3) | (4 + 3);
        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1);
    }
    // Macro 2 [L7]: shift + store
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0x80 | 0x00 | (0 << 3) | (4 + 2);
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (3 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
    // StoreMod0=FP32
    TTI_SFPCONFIG(0x700 | InstrModLoadStore::FP32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_int32_to_fp32_()
{
    constexpr int t = p_sfpu::LREG4;
    sfpi::vConstIntPrgm0 = -31;

    // InstructionTemplate[0]: SFPSETSGN to restore sign from original value
    TTI_SFPSETSGN(0, t, 12, 0);
    // InstructionTemplate[1]: SFPMAD with indirect VA (sign-dependent addition)
    TTI_SFPMAD(0, p_sfpu::LCONST_1, 0, 13, 4);

    // Macro 0 [v]: full pipeline
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (3 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0x00 | 0x40 | (4 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (6 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // StoreMod0=FP32
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::FP32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_int32_to_fp16b_()
{
    constexpr int t = p_sfpu::LREG4;
    sfpi::vConstIntPrgm0 = -31;

    TTI_SFPSETSGN(0, t, 12, 0);
    TTI_SFPMAD(0, p_sfpu::LCONST_1, 0, 13, 4);
    // Additional: SFPSTOCHRND for FP32->FP16B final conversion
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 1);

    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (3 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0x00 | 0x00 | (4 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits  = 0x00 | 0x40 | (6 << 3) | (4 + 2);
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (7 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // StoreMod0=DEFAULT (FP16B is the default store format)
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::DEFAULT, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint16_to_fp32_()
{
    // InstructionTemplate[0]: SFPCAST (uint to FP32)
    TTI_SFPCAST(0, 12, 0);

    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (1 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // StoreMod0=FP32
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::FP32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint16_to_fp16b_()
{
    TTI_SFPCAST(0, 12, 0);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, 1); // FP32_TO_FP16B

    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0x00 | 0x40 | (1 << 3) | (4 + 1);
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (2 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::DEFAULT, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint32_to_fp16b_()
{
    TTI_SFPCAST(0, 12, 0);
    TTI_SFPMAD(0, p_sfpu::LCONST_1, 0, 13, 4);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 1);

    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (2 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0x00 | 0x00 | (3 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits  = 0x00 | 0x40 | (5 << 3) | (4 + 2);
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (6 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::DEFAULT, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_fp32_to_uint16_()
{
    // InstructionTemplate[0]: SFPSWAP with 0.0 -> max(0, v) clamp
    TTI_SFPSWAP(0, p_sfpu::LCONST_0, 12, 0xf);
    // InstructionTemplate[1]: SFPSTOCHRND FP32_TO_UINT16
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, 6);

    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0x00 | 0x40 | (2 << 3) | (4 + 1);
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (3 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // StoreMod0=LO16 (store low 16 bits)
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::LO16, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint32_to_uint16_()
{
    constexpr int a0 = p_sfpu::LREG0;
    constexpr int a1 = p_sfpu::LREG1;

    // InstructionTemplate[0]: negate (0 - a) for high-16 cancellation
    TTI_SFPIADD(0, p_sfpu::LCONST_0, 12, sfpi::SFPIADD_MOD1_CC_NONE | sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
    // InstructionTemplate[1]: shift right by 16
    TTI_SFPSHFT2(-16 & 0xfff, 0, 13, 6);
    // InstructionTemplate[2]: OR with a1
    TTI_SFPOR(0, a1, 14, 0);
    // InstructionTemplate[3]: OR with a0
    TTI_SFPOR(0, a0, 15, 0);

    // Macros 0, 1, 2 for the 3-stage pipeline
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0x80 | 0x00 | (1 << 3) | (4 + 1);
        constexpr std::uint32_t store_bits  = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x40 | (0 << 3) | (4 + 2);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (1 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x40 | (0 << 3) | (4 + 3);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (1 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
    // StoreMod0=LO16
    TTI_SFPCONFIG(0x700 | InstrModLoadStore::LO16, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_int32_to_uint16_()
{
    // Same as fp32_to_uint16 init, but the calculate function does SFPCAST first
    TTI_SFPSWAP(0, p_sfpu::LCONST_0, 12, 0xf);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, 6);

    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (1 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0x00 | 0x40 | (3 << 3) | (4 + 1);
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (4 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // StoreMod0=LO16
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::LO16, 8, 1);
}

} // namespace sfpu
} // namespace ckernel
```

**Architecture-specific wrappers** (Wormhole B0 and Blackhole) additionally define two functions not in the shared file:

```cpp
// FP32/Float16_b/Bfp8_b/Bfp4_b -> UInt8
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);         // Extract exponent
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);         // Extract mantissa
        TTI_SFPIADD(-23 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2,    // exp -= 23
            sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);          // mantissa <<= (exp-23)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0); // if in < 0
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1,           // negate (two's complement)
            sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(256, p_sfpu::LREG1, p_sfpu::LREG1,            // add 256 bias
            sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPENCC(0, 0, 0, 0);                                    // Enable all lanes
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);          // mask to 0xFF
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

// UInt16/UInt32/Int32 -> UInt8
template <bool APPROXIMATION_MODE, int ITERATIONS, bool u16 = false>
inline void calculate_typecast_uint_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        if constexpr (u16) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_3, 0);  // Load uint16
        } else {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0); // Load uint32/int32
        }
        TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG0,            // Add 256 bias
            sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);          // Mask to 0xFF
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

// Init for FP32->UInt8 and uint->UInt8: just set the 0xFF mask constant
template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint8() { sfpi::vConstIntPrgm0 = 0xFF; }

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint_to_uint8() { sfpi::vConstIntPrgm0 = 0xFF; }
```

#### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` | Load data from DEST registers to LREG with format conversion |
| `SFPSTORE` | Store data from LREG to DEST registers with format conversion |
| `SFPLOADMACRO` | Macro-scheduled load that pipelines Load/Simple/MAD/Round/Store sub-units |
| `SFPLOADI` | Load immediate value into LREG (supports USHORT, SHORT, FLOATB modes) |
| `SFPCAST` | Convert between sign-magnitude integer and FP32 representations |
| `SFPEXEXP` | Extract debiased exponent from FP32 value |
| `SFPEXMAN` | Extract mantissa with implicit 1 bit from FP32 value |
| `SFPSHFT` | Barrel shift by register-specified amount |
| `SFPSHFT2` | Shift by immediate or register, with multiple modes |
| `SFPIADD` | Integer add with immediate, supports condition code setting and 2's complement |
| `SFPSETCC` | Set condition code based on LREG value (LT0, GTE0, etc.) |
| `SFPENCC` | Enable all lanes (clear condition codes) |
| `SFPAND` | Bitwise AND between LREGs |
| `SFPOR` | Bitwise OR between LREGs |
| `SFPABS` | Absolute value |
| `SFPSETSGN` | Set/clear sign bit of a value |
| `SFPSWAP` | Swap/min/max operations |
| `SFPMAD` | Multiply-add with indirect register addressing |
| `SFPSTOCHRND` | Stochastic rounding for format conversion (FP32->FP16B, FP32->UINT16) |
| `SFPCONFIG` | Configure SFPLOADMACRO templates and store modes |
| `SFPNOP` | No-operation for pipeline timing |

#### SFPU Register Usage

**Local Registers (LREGs)**:
- `LREG0`-`LREG3`: Primary working registers. Alternating between pairs (0/1 or 2/3) enables pipelining
- `LREG4` (`t`): Temporary register for absolute value in signed conversions
- `LREG7`: Stores sign bit (0 or 1) for indirect SFPMAD addressing
- `LREG12`: Used as zero constant and mask source (vConstIntPrgm0)
- `LREG13`: Used for vConstIntPrgm1 constant

**Destination Registers (DEST)**:
- Tiles are loaded from CB into DEST by the unpacker
- SFPU reads from and writes to DEST via SFPLOAD/SFPSTORE
- Each DEST row holds 32 elements (one row of a 32x32 tile)
- 8 ITERATIONS process 8 rows at a time; the LLK framework calls the kernel 4 times to cover all 32 rows

**Programmable Constants**:
- `vConstIntPrgm0`: Set per-conversion (e.g., 0xFF for uint8 mask, 1 for LSB mask, -31 for exponent comparison)
- `vConstIntPrgm1`: Set to 0x7FFF for FP32->FP16B round-to-nearest bias

#### SFPU Execution Flow

1. **Tile Acquisition**: The compute kernel calls `cb_wait_front(input_cb, 1)` to wait for the reader to produce one page in CB 0
2. **Unpack to DEST**: `copy_tile(input_cb, 0, 0)` invokes the unpacker to move data from L1 (CB page) into DEST registers, converting from the CB's data format to the internal DEST format
3. **SFPU Init**: `TYPECAST_LLK_INIT()` calls the appropriate `init_typecast_*` function, which:
   - Sets programmable constants (`vConstIntPrgm0`, `vConstIntPrgm1`)
   - Configures SFPLOADMACRO instruction templates (Simple, MAD, Round, Store sub-unit scheduling)
   - Sets the store format via `SFPCONFIG` (FP32, FP16B, INT32, LO16, etc.)
4. **SFPU Computation**: `TYPECAST_LLK(0)` calls `typecast_tile<IN_DTYPE, OUT_DTYPE>(0)`, which dispatches to `llk_math_eltwise_unary_sfpu_typecast`. This:
   - Calls `_llk_math_eltwise_unary_sfpu_params_` which iterates over the tile in groups of 8 rows
   - For each group, the `calculate_typecast_*` function runs 8 ITERATIONS
   - Each iteration processes one row of 32 elements using SFPU vector operations
   - The SFPU reads from DEST, performs conversion in LREGs, and writes back to DEST
5. **Pack to Output**: `pack_tile(0, output_cb)` invokes the packer to move converted data from DEST to the output CB, applying the output data format
6. **CB Release**: `cb_pop_front(input_cb, 1)` frees the input page; `cb_push_back(output_cb, per_core_block_dim)` signals the writer

#### SFPU Configuration

**Compile-time defines**:
- `TYPECAST_LLK_INIT` = `typecast_tile_init<IN_FORMAT_UINT, OUT_FORMAT_UINT>` (DataFormat enum as uint32_t)
- `TYPECAST_LLK` = `typecast_tile<IN_FORMAT_UINT, OUT_FORMAT_UINT>`

**Math fidelity**: HiFi4 (maximum precision, though typecast does not use the FPU)

**Approximation mode**: Always false (`math_approx_mode = false`)

**Dest accumulator**: `fp32_dest_acc_en` enables 32-bit DEST registers, required when either input or output is a 32-bit type (Float32, Int32, UInt32)

**UnpackToDestFp32**: When `preserve_fp32_precision` is true, the unpacker writes FP32 directly to DEST (bypassing the normal intermediate format), preserving full 32-bit precision for the SFPU to operate on

**SFPLOADMACRO configuration**: Most typecast kernels use SFPLOADMACRO to achieve near-optimal throughput by scheduling the SFPU's internal sub-units (Load, Simple, MAD, Round, Store) across pipeline stages. The init functions configure:
- Instruction templates (what operations each sub-unit performs)
- Macro definitions (how sub-units are scheduled per macro invocation)
- Store format (FP32, FP16B, INT32, LO16)
- Pipeline delay configuration

#### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations are structurally identical. Both:
- Include the same shared implementation from `tt_llk` submodule
- Define the same architecture-specific `fp32_to_uint8` and `uint_to_uint8` functions
- Use the same SFPLOADMACRO configurations and pipeline schedules

The only difference is the `ADDR_MOD` values used in the architecture-specific functions:
- Wormhole B0: uses `ADDR_MOD_3` for loads and `ADDR_MOD_2` for stores
- Blackhole: uses `ADDR_MOD_7` for loads and `ADDR_MOD_6` for stores

These address modifier differences reflect different SFPU register file addressing configurations between the two architectures but do not affect the conversion algorithms themselves.

---

## Conversion Dispatch Table

The LLK dispatch layer (`llk_math_eltwise_unary_sfpu_typecast`) uses a large `if constexpr` chain to select the correct SFPU kernel. Key observations:

**Conversions that require NO SFPU kernel** (handled by unpacker/packer hardware):
- Float16_b -> Float32 (packer widens)
- Bfp8_b -> Float16_b (unpacker converts)
- Float16_b -> Bfp8_b (packer compresses)
- Bfp8_b -> Float32 (unpacker/packer)
- Float32 -> Bfp8_b (packer compresses)
- Bfp4_b -> Float16_b, Bfp8_b, Float32 (unpacker)
- Float16_b, Float32 -> Bfp4_b (packer)
- UInt8 -> Int32, UInt32 (no conversion needed, just reinterpret)

**Conversions that reuse the same SFPU kernel**:
- Float16_b -> UInt16 reuses `fp32_to_uint16` (since Float16_b is widened to FP32 in DEST)
- Bfp8_b -> UInt16/Int32/UInt32 reuses the same kernels as Float16_b -> UInt16/Int32/UInt32
- UInt16 -> Int32 reuses `uint16_to_uint32` (same bit pattern, just reinterpret)
- UInt8 -> Float32 reuses `uint32_to_fp32` (UInt8 is loaded as 32-bit in DEST)

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Typecast operation architecture, program factory selection, kernel file locations
- `tenstorrent/tt-llk`: SFPU typecast kernel implementations, SFPLOADMACRO patterns, LLK API dispatch
- `tenstorrent/tt-isa-documentation`: SFPU instruction details (SFPLOAD, SFPSTORE, SFPCAST, SFPSTOCHRND format conversion semantics)

### Confluence References
Not consulted for this analysis. DeepWiki provided sufficient detail on SFPU instruction behavior for the typecast operation.

### Glean References
Not consulted for this analysis. The open-source kernel implementations provided complete coverage of the conversion algorithms.

---

## Files Analyzed

| File | Purpose |
|------|---------|
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp` | Program factory (interleaved + subgrid variants) |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.hpp` | Program factory header with shared_variables_t |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op_types.hpp` | TypecastParams and TypecastInputs structs |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op.hpp` | Device operation with program factory variant selection |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp` | Compute kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer kernel |
| `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h` | Compute API: typecast_tile, typecast_tile_init |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h` | LLK dispatch layer (WH) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h` | LLK dispatch layer (BH) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h` | Architecture-specific SFPU wrappers (WH) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h` | Architecture-specific SFPU wrappers (BH) |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h` | Shared SFPU kernel implementations |
