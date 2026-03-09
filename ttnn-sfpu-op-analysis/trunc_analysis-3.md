# TTNN SFPU Operation Analysis: Trunc

## Operation Overview

| Property | Value |
|---|---|
| **Operation Name** | `trunc` |
| **Operation Type** | Unary Elementwise (SFPU) |
| **UnaryOpType Enum** | `UnaryOpType::TRUNC` |
| **Program Factory** | `UnaryProgramFactory` (shared with all standard unary ops) |
| **Compute Kernel** | `eltwise_sfpu.cpp` (generic unary SFPU kernel) |
| **SFPU Kernel** | `ckernel_sfpu_rounding_ops.h` (`_calculate_trunc_` / `_trunc_body_`) |
| **Include Guard Macro** | `SFPU_OP_ROUND_FAMILY_INCLUDE` |
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/rounding.h` |
| **Python API** | `ttnn.trunc(input_tensor)` |
| **Parameters** | None (non-parametrized unary op) |
| **Math Approx Mode** | `false` (always HiFi) |
| **Math Fidelity** | `MathFidelity::HiFi4` |

### Mathematical Definition

`trunc(x)` returns the integer part of `x` by removing the fractional digits, rounding toward zero. Equivalently:
- For `x >= 0`: `trunc(x) = floor(x)`
- For `x < 0`: `trunc(x) = ceil(x)`

This is distinct from `floor` (always rounds toward negative infinity) and `ceil` (always rounds toward positive infinity).

---

## Program Factory Analysis

### File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

### Factory Selection

The `trunc` operation uses the standard `UnaryProgramFactory`, which is shared by most unary SFPU operations. The factory is selected by the `ttnn::prim::UnaryDeviceOperation` dispatcher based on tensor sharding configuration:
- **Interleaved tensors** -> `UnaryProgramFactory` (this analysis)
- **Sub-core grid tensors** -> `UnarySubCoreGridProgramFactory` (same file, similar structure)
- **Sharded tensors** -> `UnaryShardedProgramFactory` (separate file)

### Program Structure

The factory creates a program with three kernels running on each assigned Tensix core:

1. **Reader kernel** -- reads input tiles from DRAM via NoC into input circular buffer (CB0)
2. **Compute kernel** -- unpacks tiles from CB0, applies SFPU trunc operation in DST registers, packs result to output circular buffer (CB2)
3. **Writer kernel** -- writes output tiles from CB2 back to DRAM via NoC

### Circular Buffer Configuration

| CB Index | Name | Purpose | Page Count | Data Format |
|---|---|---|---|---|
| `c_0` | Input | Holds input tiles for unpacking | 2 | Input tensor dtype |
| `c_1` | Temporary | Not used for trunc (only for HARDSHRINK/LOGIT) | N/A | N/A |
| `c_2` | Output | Holds packed output tiles for writing | 2 | Output tensor dtype |

The double-buffering (2 pages per CB) allows overlap between data movement and compute.

### Work Distribution

Work is split across all available compute cores using `split_work_to_cores`:
- `core_group_1`: cores that each process `num_pages_per_core_group_1` tiles
- `core_group_2`: cores that each process `num_pages_per_core_group_2` tiles (remainder distribution)

Each core receives identical kernel binaries but different runtime arguments specifying which tile range to process.

### Compile-Time Defines

For `trunc`, the following defines are injected into the compute kernel:

| Define | Value | Purpose |
|---|---|---|
| `SFPU_OP_ROUND_FAMILY_INCLUDE` | `1` | Gates inclusion of `rounding.h` via `sfpu_split_includes.h` |
| `SFPU_OP_CHAIN_0` | `rounding_op_tile_init(); trunc_tile(0);` | The actual SFPU operation call chain |
| `INP_FLOAT32` or `INP_FLOAT` | `1` | Input data type indicator |

### Runtime Arguments

| Kernel | Arg 0 | Arg 1 | Arg 2 |
|---|---|---|---|
| Reader | `src_buffer->address()` | `num_pages_per_core` | `num_pages_written` (start offset) |
| Writer | `dst_buffer->address()` | `num_pages_per_core` | `num_pages_written` (start offset) |
| Compute | `packed_scalar1 = 0` | `packed_scalar2 = 0` | N/A |

The compute kernel receives zero-valued scalar arguments because `trunc` has no runtime parameters. These are passed via `SetRuntimeArgs` but ignored by the SFPU function.

### Compute Configuration

```cpp
tt::tt_metal::ComputeConfig{
    .math_fidelity = MathFidelity::HiFi4,
    .fp32_dest_acc_en = args.fp32_dest_acc_en,    // caller-controlled
    .unpack_to_dest_mode = unpack_to_dest_mode,    // Default unless preserve_fp32_precision
    .bfp8_pack_precise = args.bfp8_pack_precise,
    .math_approx_mode = false,                     // trunc always uses exact mode
    .compile_args = {num_pages_per_core, 1},        // block_cnt, block_size
    .defines = unary_defines                        // includes SFPU_OP_ROUND_FAMILY_INCLUDE
}
```

---

## Kernel Implementations

### Reader Kernel

#### File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);   // DRAM address of source tensor buffer
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of pages (tiles) this core must read
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting page index for this core

    constexpr auto src_args = TensorAccessorArgs<0>();      // Compile-time tensor accessor config (bank mapping, etc.)

    constexpr uint32_t cb_id_in0 = 0;                      // Input circular buffer index (CB0 = c_0)

    // Page size is determined by the CB configuration set by the host, which
    // matches the tile size for TILE layout or row size for ROW_MAJOR layout
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    constexpr uint32_t onepage = 1;                        // Process one page at a time

    // Create a TensorAccessor that maps logical page IDs to physical DRAM addresses
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    // Read pages sequentially from start_id to start_id + num_pages
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_id_in0, onepage);               // Wait for space in CB0
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0); // Get L1 write address
        noc_async_read_page(i, s, l1_write_addr);          // Initiate NoC read from DRAM to L1
        noc_async_read_barrier();                           // Wait for NoC read to complete
        cb_push_back(cb_id_in0, onepage);                  // Signal compute kernel that a tile is ready
    }
}
```

### Writer Kernel

#### File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);    // DRAM address of destination tensor buffer
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of pages (tiles) this core must write
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting page index for this core

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // Output CB index (CB2 = c_2)
    constexpr auto dst_args = TensorAccessorArgs<1>();      // Compile-time tensor accessor config for destination

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

    // Write pages sequentially from start_id to start_id + num_pages
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_id_out, onepage);                 // Wait for compute kernel to produce a tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);   // Get L1 read address
        noc_async_write_page(i, s, l1_read_addr);          // Initiate NoC write from L1 to DRAM
        noc_async_writes_flushed();                        // Flush write buffer
        cb_pop_front(cb_id_out, onepage);                  // Free the CB slot for compute kernel
    }
    noc_async_write_barrier();                             // Final barrier to ensure all writes complete
}
```

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
#include "api/compute/common.h"                            // Common compute APIs (tile_regs_acquire, etc.)
#include "api/compute/tile_move_copy.h"                    // copy_tile, pack_tile
#include "api/compute/eltwise_unary/eltwise_unary.h"       // init_sfpu and unary framework
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // Conditionally includes rounding.h when SFPU_OP_ROUND_FAMILY_INCLUDE=1
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tile blocks to process on this core
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for trunc)

    // Initialize the SFPU pipeline: configures unpack from CB0, pack to CB2,
    // and sets up the SFPU math state (calls rounding_op_tile_init() via SFPU_OP_CHAIN_0 init)
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve space in output CB for the entire block (1 tile)
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to DST register file for writing
            // This synchronizes with the packer thread (T2) which may still be
            // reading DST from a previous iteration
            tile_regs_acquire();

            // Wait for reader kernel to push a tile into CB0
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Unpack tile from CB0 into DST register at index 0
            // This triggers the unpacker (T0 thread) to read from L1 and
            // format-convert the data into DST registers
            copy_tile(tt::CBIndex::c_0, 0, 0);

            // Execute the SFPU operation chain. For trunc, this expands to:
            //   rounding_op_tile_init();  -- one-time SFPU init (no-op after first call)
            //   trunc_tile(0);            -- apply trunc to tile at DST index 0
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Signal that DST register writes are complete; hand off to packer
            tile_regs_commit();

            // Wait for packer to be ready to read from DST
            tile_regs_wait();

            // Pack tile from DST register 0 into output CB2
            pack_tile(0, tt::CBIndex::c_2);

            // Release input tile from CB0 so reader can reuse the slot
            cb_pop_front(tt::CBIndex::c_0, 1);

            // Release DST registers for next iteration
            tile_regs_release();
        }
        // Push the completed block to CB2 so the writer kernel can read it
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File (Wormhole B0)
`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_rounding_ops.h`

#### SFPU Kernel File (Blackhole)
`tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_rounding_ops.h`

#### Annotated SFPU Kernel Source (Wormhole B0)

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

// Core truncation implementation. Operates on SFPU local registers:
//   Input:  LREG0 contains the FP32 value to truncate
//   Output: LREG1 contains the truncated (integer part) result
//   Clobbered: LREG2, LREG3
//
// Algorithm: Constructs a bitmask that zeros out the fractional mantissa bits,
// then applies it via bitwise AND to produce the integer part.
//
// IEEE 754 FP32 layout: [1 sign][8 exponent][23 mantissa]
// For a value with unbiased exponent E (where E = biased_exp - 127):
//   - If E >= 23: the value is already an integer (all 23 mantissa bits represent integer digits)
//   - If E < 0: |value| < 1, so trunc(value) = 0 (handled by disabling those lanes)
//   - Otherwise: the top E mantissa bits are integer, the bottom (23 - E) are fractional
//     We mask off the fractional bits by AND-ing with 0xFFFFFFFF << (23 - E)
inline void _trunc_body_()
{
    // Load constant 23 into LREG3 (number of mantissa bits in FP32)
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_SHORT, 23);

    // Load 0x8000 as float-b into LREG1. In SFPLOADI_MOD0_FLOATB mode, the 16-bit
    // immediate is placed in the upper 16 bits, producing 0x8000_0000.
    // This serves as the initial mask value (sign bit only).
    // WHY: For values with exponent < 0 (magnitude < 1), the mask stays as
    // 0x8000_0000, which preserves only the sign bit, effectively producing +/- 0.
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);

    // Extract unbiased exponent from LREG0 into LREG2, and set condition codes:
    // - SFPEXEXP_MOD1_SET_CC_SGN_EXP: sets CC based on sign of extracted exponent
    // - SFPEXEXP_MOD1_SET_CC_COMP_EXP: compares exponent (disables lanes where exp < 0)
    // WHY: When exponent < 0, the value is between -1 and 1, so trunc = 0.
    // By disabling those lanes, they retain the 0x8000_0000 mask (preserving sign, zeroing magnitude).
    TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);

    // Overwrite LREG1 with 0xFFFF_FFFF (all bits set) for enabled lanes only.
    // Disabled lanes (exp < 0) keep the previous value 0x8000_0000.
    // WHY: This is the starting point for the mask; we will shift it left to zero
    // out the fractional mantissa bits.
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);

    // Compute shift amount: LREG2 = 23 - exponent.
    // SFPIADD with ARG_2SCOMP_LREG_DST negates LREG2 before adding to LREG3 (which holds 23).
    // CC_GTE0: only execute for lanes where (23 - exp) >= 0, i.e., exp <= 23.
    // For exp > 23, the value is already a large integer with no fractional part,
    // so no masking is needed; those lanes are disabled and keep mask = 0xFFFFFFFF.
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);

    // Shift mask left by (23 - exp) bits. This creates a bitmask like:
    //   exp=0  -> shift 23 -> mask = 0xFF800000 (only sign+exponent preserved)
    //   exp=10 -> shift 13 -> mask = 0xFFFFE000 (sign+exponent+10 mantissa bits preserved)
    //   exp=22 -> shift 1  -> mask = 0xFFFFFFFE (only LSB of mantissa cleared)
    // SFPSHFT2_MOD1_SHFT_LREG: shift amount comes from LREG2 (the shift register operand)
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SHFT_LREG);

    // Re-enable all SFPU lanes by clearing condition code state.
    // WHY: Subsequent instructions need to operate on all lanes unconditionally.
    TTI_SFPENCC(0, 0, 0, 0);

    // Apply the mask: LREG1 = LREG0 AND LREG1.
    // This zeros out the fractional mantissa bits, leaving only the integer part.
    // For lanes where exp < 0: mask was 0x8000_0000, so result preserves sign but
    // zeros exponent and mantissa, producing +/-0.
    // For lanes where exp > 23: mask was 0xFFFFFFFF, so value is unchanged (already integer).
    TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
}

// Top-level trunc function called by the LLK dispatch.
// Template parameters:
//   APPROXIMATION_MODE: not used by trunc (always exact)
//   ITERATIONS: number of 8-row sub-tiles to process (default 8 = one full face of 16x16)
//
// Each iteration processes 1 row of the SFPU's 32-wide SIMD, loading from DST,
// applying _trunc_body_, and storing back.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_trunc_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Load one row from DST register file into LREG0.
        // ADDR_MOD_3 auto-increments the DST row pointer after each SFPLOAD/SFPSTORE pair.
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);

        // Apply truncation: input in LREG0, result in LREG1
        _trunc_body_();

        // Store the truncated result from LREG1 back to DST
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);

        // Advance the DST register pointer to the next row
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
```

#### Annotated SFPU Kernel Source (Blackhole)

The Blackhole implementation is structurally identical to Wormhole for `_trunc_body_()` and `_calculate_trunc_()`. The key differences appear in the helper functions `_floor_body_()` and `_ceil_body_()`, which use `TTI_SFPGT` (available on Blackhole) instead of the `SFPSETCC` + `SFPIADD` workaround required on Wormhole. Since `_trunc_body_()` itself does not use `SFPGT`, the trunc implementation is identical across both architectures.

The only minor difference is the address modifier constant used: Blackhole uses `ADDR_MOD_7` while Wormhole uses `ADDR_MOD_3`. This is an architecture-specific address mode constant for the DST register auto-increment behavior.

```cpp
// Blackhole version -- only showing the differences from Wormhole
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_trunc_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);    // Note: ADDR_MOD_7 instead of ADDR_MOD_3
        _trunc_body_();                                     // Identical to Wormhole
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);   // Note: ADDR_MOD_7 instead of ADDR_MOD_3
        sfpi::dst_reg++;
    }
}
```

#### SFPU Instructions Used

| Instruction | Opcode | Description | Usage in Trunc |
|---|---|---|---|
| `SFPLOADI` | 0x71 | Load immediate value into a local register | Loads constants 23 (mantissa bit count), 0x80000000 (sign mask), and 0xFFFFFFFF (all-ones mask) |
| `SFPEXEXP` | 0x77 | Extract exponent field from FP32 value, optionally set condition codes | Extracts unbiased exponent from input value; disables lanes where exponent < 0 (value magnitude < 1) |
| `SFPIADD` | 0x79 | Integer addition/subtraction with condition code control | Computes `23 - exponent` (the number of fractional bits to mask off); disables lanes where result < 0 |
| `SFPSHFT2` | 0x94 | Bitwise shift with register-specified shift amount | Shifts 0xFFFFFFFF left by `(23 - exp)` to create the integer-preserving mask |
| `SFPENCC` | 0x8A | Reset condition code enable flags on all lanes | Re-enables all lanes after conditional operations |
| `SFPAND` | 0x7E | Bitwise AND between two registers | Applies the constructed mask to zero out fractional mantissa bits |
| `SFPLOAD` | 0x70 | Load value from DST register file into SFPU local register | Loads each row of the tile from DST into LREG0 for processing |
| `SFPSTORE` | 0x72 | Store value from SFPU local register to DST register file | Writes the truncated result from LREG1 back to DST |

#### SFPU Register Usage

| Register | Role | Lifetime |
|---|---|---|
| `LREG0` | Input value loaded from DST; also used as source operand in final AND | Per-iteration; loaded at start, consumed by SFPAND |
| `LREG1` | Mask register; final output after AND with LREG0 | Per-iteration; initialized to 0x80000000, overwritten with 0xFFFFFFFF for enabled lanes, shifted, then AND-ed |
| `LREG2` | Exponent value; then shift amount `(23 - exp)` | Per-iteration; set by SFPEXEXP, modified by SFPIADD, consumed by SFPSHFT2 |
| `LREG3` | Constant 23 (mantissa bit count) | Per-iteration; loaded fresh each iteration (could theoretically be cached, but rdiv also uses LREG3) |
| `DST registers` | Source and destination for tile data | Row-by-row access; pointer auto-incremented via ADDR_MOD and `dst_reg++` |
| `CC (Condition Codes)` | Per-lane enable/disable flags | Used to skip lanes where exponent < 0 or shift amount < 0; reset by SFPENCC at end |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` to lock the DST register file, then `copy_tile(CB0, 0, 0)` to unpack one tile from CB0 into DST[0]. The unpacker (T0 thread) performs format conversion from the source data format to FP32 in DST.

2. **SFPU dispatch**: `trunc_tile(0)` is called, which expands via the `SFPU_TWO_PARAM_KERNEL` macro to `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::_calculate_trunc_<false, 8>, 0, (int)VectorMode::RC)`.

3. **LLK params orchestration**: `_llk_math_eltwise_unary_sfpu_params_` sets the DST write address, stalls until the SFPU is ready (`TTI_STALLWAIT`), then iterates over all 4 faces of the 32x32 tile (VectorMode::RC). For each face, it calls `_calculate_trunc_<false, 8>()` and advances the DST pointer by 2 groups of 8 rows.

4. **Per-row SFPU processing** (inside `_calculate_trunc_`, 8 iterations per face, 4 faces = 32 rows total):
   - `SFPLOAD`: Load one row (32 lanes) from DST into LREG0
   - `_trunc_body_()`:
     a. Load constant 23 into LREG3
     b. Load sign-bit mask (0x80000000) into LREG1
     c. Extract exponent from LREG0, disable lanes where exponent < 0
     d. Load all-ones mask (0xFFFFFFFF) into LREG1 for enabled lanes
     e. Compute shift amount: `23 - exponent`; disable lanes where shift < 0 (exponent > 23)
     f. Shift mask left by shift amount to create integer-preserving bitmask
     g. Re-enable all lanes (SFPENCC)
     h. AND input with mask to produce truncated value in LREG1
   - `SFPSTORE`: Write LREG1 back to DST
   - Advance DST row pointer

5. **Pack and output**: After the SFPU completes, `tile_regs_commit()` signals the packer. `tile_regs_wait()` waits for the packer to be ready. `pack_tile(0, CB2)` triggers the packer (T2 thread) to convert from DST format to the output format and write to CB2. `cb_push_back(CB2, 1)` notifies the writer kernel.

#### SFPU Configuration

- **Math fidelity**: `MathFidelity::HiFi4` -- highest fidelity, though trunc is a bitwise operation and fidelity does not affect its result
- **Math approx mode**: `false` -- the `APPROXIMATION_MODE` template parameter is always false for trunc. The trunc algorithm is exact (pure bit manipulation), so approximation mode has no effect on the implementation.
- **Compile-time defines**: `SFPU_OP_ROUND_FAMILY_INCLUDE=1` gates the inclusion of `rounding.h` which provides `trunc_tile()`. This is a code-size optimization -- only the rounding family functions are compiled into the kernel binary.
- **ITERATIONS**: Fixed at 8 (processes 8 rows per SFPU invocation, matching one face's worth of data in the 32x32 tile)
- **VectorMode**: `RC` (Row-Column) -- processes all 4 faces of the 32x32 tile (each face is 16x16, organized as 2 groups of 8 rows each)
- **FP32 dest accumulation**: Controlled by the caller (`args.fp32_dest_acc_en`). When enabled, unpacker loads FP32 values directly into DST without format conversion, preserving full precision.

#### Hardware Compatibility Notes

**Wormhole B0 vs Blackhole differences for the rounding family:**

- **`_trunc_body_()`**: Identical on both architectures. The truncation algorithm uses only basic SFPU instructions (SFPLOADI, SFPEXEXP, SFPIADD, SFPSHFT2, SFPENCC, SFPAND) that are available on both platforms.

- **`_floor_body_()` and `_ceil_body_()`**: These differ between architectures. Blackhole has the `SFPGT` instruction for direct floating-point greater-than comparison. Wormhole lacks `SFPGT`, so floor/ceil use a workaround with `SFPSETCC` (check sign) combined with `SFPIADD` (integer subtraction for comparison). Since trunc does not call floor or ceil, this difference does not affect the trunc operation.

- **Address modifier**: Wormhole uses `ADDR_MOD_3` while Blackhole uses `ADDR_MOD_7` for the DST register auto-increment mode in SFPLOAD/SFPSTORE. This is an architectural constant difference that does not affect behavior.

- **Instruction latencies**: Both architectures have the same IPC and latency characteristics for the instructions used by trunc (all are 1 IPC, 1 cycle latency except SFPSTORE which has 2-3 cycle latency).

---

## Dispatch Chain Summary

The complete dispatch chain from Python API to SFPU hardware instructions:

```
ttnn.trunc(tensor)
  -> ttnn::operations::unary::ExecuteUnary<UnaryOpType::TRUNC>
    -> ttnn::prim::UnaryDeviceOperation (selects UnaryProgramFactory)
      -> UnaryProgramFactory::create()
        -> utils::get_block_defines() produces SFPU_OP_CHAIN_0 = "rounding_op_tile_init(); trunc_tile(0);"
        -> utils::get_compute_kernel_path() returns "eltwise_sfpu.cpp" (default path)
        -> CreateKernel(eltwise_sfpu.cpp, defines={SFPU_OP_ROUND_FAMILY_INCLUDE=1, ...})
          -> eltwise_sfpu.cpp::kernel_main()
            -> SFPU_OP_CHAIN_0 expands to:
              -> rounding_op_tile_init()  [SFPU_UNARY_KERNEL_INIT -> llk_math_eltwise_unary_sfpu_init]
              -> trunc_tile(0)            [SFPU_TWO_PARAM_KERNEL -> _llk_math_eltwise_unary_sfpu_params_]
                -> ckernel::sfpu::_calculate_trunc_<false, 8>()
                  -> TTI_SFPLOAD (load row from DST)
                  -> _trunc_body_()
                    -> TTI_SFPLOADI (load 23)
                    -> TTI_SFPLOADI (load 0x80000000)
                    -> TTI_SFPEXEXP (extract exponent, set CC)
                    -> TTI_SFPLOADI (load 0xFFFFFFFF)
                    -> TTI_SFPIADD  (23 - exp, conditional)
                    -> TTI_SFPSHFT2 (shift mask)
                    -> TTI_SFPENCC  (reset CC)
                    -> TTI_SFPAND   (apply mask)
                  -> TTI_SFPSTORE (write back to DST)
```

---

## Relationship to Other Rounding Operations

The `trunc` operation is part of the "rounding family" (guarded by `SFPU_OP_ROUND_FAMILY_INCLUDE`), which shares `ckernel_sfpu_rounding_ops.h`. The relationship between these operations:

| Operation | Implementation | Relationship to Trunc |
|---|---|---|
| `trunc` | `_trunc_body_()` directly | Base operation |
| `floor` | `_trunc_body_()` + conditional subtract 1 for negative non-integers | Calls `_trunc_body_()` as subroutine |
| `ceil` | `_trunc_body_()` + conditional add 1 for positive non-integers | Calls `_trunc_body_()` as subroutine |
| `frac` | `_trunc_body_()` + subtract: `x - trunc(x)` | Calls `_trunc_body_()` as subroutine |
| `round` | Separate algorithm using FP32 add-subtract trick for round-to-nearest-even | Independent implementation |
| `stochastic_round` | Uses `SFPSTOCHRND` hardware instruction | Independent implementation |

The `_trunc_body_()` function is also reused by `rdiv` (reciprocal division with truncation rounding mode) and `binary_fmod` (floating-point modulo), demonstrating its role as a foundational SFPU primitive.

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Unary program factory structure, compute kernel dispatch, SFPU_OP_ROUND_FAMILY_INCLUDE macro
- `tenstorrent/tt-llk`: `_trunc_body_()` implementation details, SFPU instruction usage, architecture-specific differences
- `tenstorrent/tt-isa-documentation`: SFPU instruction opcodes and truncation via SFPSTOCHRND (not used by this implementation)
- `tenstorrent/sfpi`: SFPI intrinsics for stochastic rounding (context only; trunc uses raw TTI instructions)

### Confluence References
- **Tensix SFPU Instruction Set Architecture** (Page ID: 1170505767): Referenced for authoritative instruction specifications:
  - `SFPEXEXP` (0x77): Exponent extraction with CC setting, 1 IPC, 1 cycle latency
  - `SFPIADD` (0x79): Integer add/subtract with CC control, 1 IPC, 1 cycle latency
  - `SFPSHFT2` (0x94): Register-specified shift, 0.5-1 IPC, 1-2 cycle latency
  - `SFPAND` (0x7E): Bitwise AND, 1 IPC, 1 cycle latency
  - `SFPENCC` (0x8A): CC reset on all lanes, 1 IPC, 1 cycle latency
  - `SFPLOADI` (0x71): Immediate load, 1 IPC, 1 cycle latency
  - `SFPLOAD` (0x70): DST register file load, 1 IPC, 1 cycle latency
  - `SFPSTORE` (0x72): DST register file store, 1 IPC, 2-3 cycle latency

### Glean References
Not consulted. The trunc implementation is fully documented through DeepWiki and Confluence sources.
