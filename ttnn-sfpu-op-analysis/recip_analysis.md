# RECIP (Reciprocal) Operation Analysis

## Operation Overview

| Property | Value |
|---|---|
| **Operation Name** | `recip` (Reciprocal) |
| **TTNN Op Type** | `UnaryOpType::RECIP` |
| **Mathematical Function** | `f(x) = 1/x` |
| **Program Factory** | `UnaryProgramFactory` (shared with all unary SFPU ops) |
| **Compute Kernel** | `eltwise_sfpu.cpp` (generic SFPU dispatch kernel) |
| **SFPU Kernel** | `ckernel_sfpu_recip.h` (architecture-specific) |
| **Supported Data Formats** | Float32, Float16_b, Bfp8_b (full accuracy) |
| **Approximation Mode** | `false` (always uses full-precision path) |
| **Parameters** | None (parameterless unary op) |

## Program Factory Analysis

### Source File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

### Factory Class
`UnaryProgramFactory` in namespace `ttnn::prim`. This is a shared factory used by all standard unary SFPU operations. A second variant, `UnarySubCoreGridProgramFactory`, supports custom sub-core-grid partitioning.

### Work Distribution Strategy
The factory uses `tt::tt_metal::split_work_to_cores()` to distribute tiles (or row-major pages) across the compute grid. This produces two core groups:
- **Core Group 1**: Cores assigned `num_pages_per_core_group_1` pages each
- **Core Group 2**: Cores assigned `num_pages_per_core_group_2` pages each (may be empty)

Each core group gets its own compute kernel handle, but both use the same kernel source and defines. The only difference is the compile-time `per_core_block_cnt` argument.

### Circular Buffer Configuration

| CB Index | Name | Size (pages) | Data Format | Purpose |
|---|---|---|---|---|
| `c_0` | Input | 2 | Input tensor format | Holds tiles read from DRAM by the reader kernel |
| `c_1` | Temp | 2 | Input tensor format | Only allocated for HARDSHRINK/LOGIT (not used by RECIP) |
| `c_2` | Output | 2 | Output tensor format | Holds computed tiles for the writer kernel |

Double-buffering (2 pages per CB) enables pipelining: one tile can be processed while the next is being read or the previous is being written.

### Kernel Registration

Three kernels are registered per program:

#### Reader Kernel
- **Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Config**: `ReaderDataMovementConfig` with `TensorAccessorArgs` for the source buffer
- **Runtime Args**: `{src_buffer_address, num_pages_per_core, start_page_id}`

#### Writer Kernel
- **Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Config**: `WriterDataMovementConfig` with output CB index and `TensorAccessorArgs` for the destination buffer
- **Runtime Args**: `{dst_buffer_address, num_pages_per_core, start_page_id}`

#### Compute Kernel
- **Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **Config**: `ComputeConfig` with:
  - `.math_fidelity = MathFidelity::HiFi4`
  - `.fp32_dest_acc_en` from operation attributes
  - `.math_approx_mode = false` (RECIP always returns false from `get_op_approx_mode`)
  - `.compile_args = {per_core_block_cnt, 1}` (block count, block size of 1 tile)
  - `.defines` includes `SFPU_OP_RECIP_INCLUDE=1` and `SFPU_OP_CHAIN_0` macro
- **Runtime Args**: `{packed_scalar1=0, packed_scalar2=0}` (RECIP has no scalar parameters)

### Preprocessor Define Chain for RECIP

The `get_block_defines()` utility generates these defines for RECIP:

1. `SFPU_OP_RECIP_INCLUDE` = `"1"` -- enables `#include "api/compute/eltwise_unary/recip.h"` in `sfpu_split_includes.h`
2. `SFPU_OP_CHAIN_0_INIT_0` = `"recip_tile_init<false>();"` -- initialization call
3. `SFPU_OP_CHAIN_0_FUNC_0` = `"recip_tile<false>(0);"` -- per-tile compute call
4. `SFPU_OP_CHAIN_0` = `"SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0"` -- concatenated into the compute kernel's main loop

The `<false>` template parameter is `legacy_compat`, set to false for the standard path.

## Kernel Implementations

### Reader Kernel

#### Reader Kernel File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

#### Annotated Reader Kernel Source
```cpp
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime arguments set by the host program factory
    const uint32_t src_addr = get_arg_val<uint32_t>(0);   // DRAM address of the input tensor buffer
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of pages (tiles) this core must read
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // First page ID for this core (offset into the tensor)

    // Compile-time args encode the TensorAccessor configuration (interleaved layout, bank mapping, etc.)
    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_id_in0 = 0;  // CB index 0 = input circular buffer (c_0)

    // Page size comes from the CB interface; works for both tile and row-major layouts
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    constexpr uint32_t onepage = 1;  // Process one page at a time

    // Construct a TensorAccessor from compile-time args, runtime base address, and page size
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    // Sequential page-by-page read loop (can also run backwards if BACKWARDS is defined)
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_reserve_back(cb_id_in0, onepage);             // Wait for space in the input CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0); // Get the L1 write pointer
        noc_async_read_page(i, s, l1_write_addr);        // Issue async NoC read from DRAM to L1
        noc_async_read_barrier();                          // Wait for the read to complete
        cb_push_back(cb_id_in0, onepage);                 // Signal that one page is available for the compute kernel
    }
}
```

### Writer Kernel

#### Writer Kernel File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

#### Annotated Writer Kernel Source
```cpp
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);   // DRAM address of the output tensor buffer
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of pages this core must write
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // First page ID for this core

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // Output CB index (c_2 = 2)
    constexpr auto dst_args = TensorAccessorArgs<1>();           // TensorAccessor config for output buffer

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    // For sharded output, just wait for all pages in CB (no explicit write-back needed)
    cb_wait_front(cb_id_out, num_pages);
#else
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_wait_front(cb_id_out, onepage);                // Wait for the compute kernel to produce a result tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);  // Get the L1 read pointer for the output CB
        noc_async_write_page(i, s, l1_read_addr);         // Issue async NoC write from L1 to DRAM
        noc_async_writes_flushed();                        // Flush pending writes (non-blocking barrier)
        cb_pop_front(cb_id_out, onepage);                  // Free the CB slot for reuse
    }
    noc_async_write_barrier();  // Final barrier to ensure all writes complete
#endif
}
```

### Compute Kernel

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

#### Annotated Compute Kernel Source
```cpp
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // Conditionally includes recip.h when SFPU_OP_RECIP_INCLUDE=1
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tile blocks to process
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for RECIP)

    // Initialize the SFPU for this operation. For RECIP, this expands to:
    // recip_tile_init<false>() which calls sfpu::recip_init<APPROX, DST_ACCUM_MODE, legacy_compat=false>()
    // This sets up architecture-specific constants and instruction templates.
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // Reserve output CB space for one block
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // Acquire exclusive access to DST registers

            cb_wait_front(tt::CBIndex::c_0, 1);  // Wait for reader to produce one input tile

            copy_tile(tt::CBIndex::c_0, 0, 0);   // Unpack: copy tile from CB c_0 slot 0 to DST register 0

            // SFPU_OP_CHAIN_0 expands to:
            // recip_tile_init<false>(); recip_tile<false>(0);
            // This computes 1/x for all 32x32 elements in DST[0] in-place.
            // The init call sets up polynomial coefficients (Wormhole) or SFPLOADMACRO templates (Blackhole).
            // The recip_tile call invokes calculate_reciprocal<APPROX, DST_ACCUM_MODE, 8, false>().
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();   // Signal that DST registers have been written and are ready to pack

            tile_regs_wait();     // Wait for pack engine to be ready

            pack_tile(0, tt::CBIndex::c_2);  // Pack: move result from DST[0] to output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // Release the input tile slot in CB c_0

            tile_regs_release();  // Release DST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // Signal that output tiles are available for writer
    }
}
```

### SFPU Kernel Implementation

This section provides a deep dive into the underlying SFPU kernel function that the compute kernel dispatches to. The reciprocal operation has significantly different implementations on Wormhole B0 and Blackhole architectures.

#### SFPU API Layer

##### File: `tt_metal/hw/inc/api/compute/eltwise_unary/recip.h`

This is the hardware-abstraction API layer that the compute kernel calls. It bridges between the generic `SFPU_OP_CHAIN_0` macro and the architecture-specific LLK implementation.

```cpp
#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_recip.h"                     // Architecture-specific recip implementation
#include "llk_math_eltwise_unary_sfpu_macros.h"     // Macro framework for SFPU dispatch
#endif

namespace ckernel {

// Initialization: configures SFPU constants and instruction templates for reciprocal.
// Expands to: llk_math_eltwise_unary_sfpu_init<SfpuType::reciprocal, APPROX>(
//     sfpu::recip_init<APPROX, DST_ACCUM_MODE, legacy_compat>)
template <bool legacy_compat = true>
ALWI void recip_tile_init() {
    MATH(SFPU_THREE_TEMPLATE_PARAM_INIT(reciprocal, sfpu::recip_init, APPROX, DST_ACCUM_MODE, legacy_compat));
}

// Per-tile computation: applies reciprocal to all elements in DST[idst].
// Expands to: _llk_math_eltwise_unary_sfpu_params_<APPROX>(
//     ckernel::sfpu::calculate_reciprocal<APPROX, DST_ACCUM_MODE, 8, legacy_compat>, idst, vector_mode)
// The "8" is the ITERATIONS parameter = number of 32-element sub-tiles in a 32x32 tile
// (32 rows / 4 rows per sub-tile = 8 iterations when VectorMode::RC).
template <bool legacy_compat = true>
ALWI void recip_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC) {
    MATH(SFPU_FOUR_PARAM_KERNEL_FP32_FIRST_FN(
        calculate_reciprocal, APPROX, DST_ACCUM_MODE, 8, legacy_compat, idst, vector_mode));
}

}  // namespace ckernel
```

#### Architecture-Specific Wrapper (Blackhole)

##### File: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h`

```cpp
#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

// High-level SFPI reciprocal: dispatches to _sfpu_reciprocal_ with 0 or 2 Newton-Raphson iterations.
// APPROXIMATE=true -> 0 iterations (hardware approx only)
// APPROXIMATE=false -> 2 iterations (full precision)
template <bool APPROXIMATE = false, bool save_reg = true>
sfpi_inline vFloat sfpu_reciprocal(const vFloat in) {
    return _sfpu_reciprocal_<APPROXIMATE ? 0 : 2>(in);
}

template <bool APPROXIMATE = false>
sfpi_inline void sfpu_reciprocal_init() {
    _init_sfpu_reciprocal_<APPROXIMATE>();
}

// Main entry point called by recip_tile via the macro framework.
// ITERATIONS=8 means process 8 sub-tiles of 4 rows x 32 columns = 128 elements each.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8, bool legacy_compat = false>
inline void calculate_reciprocal() {
    _calculate_reciprocal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en, legacy_compat>(ITERATIONS);
}

// Initialization entry point called by recip_tile_init.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool legacy_compat = false>
void recip_init() {
    _init_reciprocal_<APPROXIMATION_MODE, is_fp32_dest_acc_en, legacy_compat>();
}

}  // namespace sfpu
}  // namespace ckernel
```

#### Architecture-Specific Wrapper (Wormhole B0)

##### File: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h`

Identical to the Blackhole wrapper, except it additionally includes `"sfpu/ckernel_sfpu_recip.h"` from the tt_llk submodule, which provides the Wormhole-specific `_sfpu_reciprocal_` implementation using software Newton-Raphson (no SFPARECIP hardware instruction).

#### Blackhole Low-Level SFPU Implementation

##### File: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h`

##### Annotated Blackhole SFPU Kernel Source
```cpp
#pragma once

#include <cstdint>
#include "ckernel_sfpu_rsqrt_compat.h"
#include "lltt.h"    // Provides lltt::replay for replay buffer execution
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Core reciprocal function using Blackhole's hardware SFPARECIP instruction.
// max_iter controls Newton-Raphson refinement iterations:
//   0 = hardware approximation only (~7-bit precision)
//   1 = one NR iteration (~15-bit precision, sufficient for BF16)
//   2 = two NR iterations (~24-bit precision, sufficient for FP32)
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat x) {
    // SFPARECIP hardware instruction: computes approximate 1/x for all 32 lanes.
    // Returns +/-0 for +/-inf or x >= +/-2^126, and +/-inf for x = +/-0.
    sfpi::vFloat y = sfpi::approx_recip(x);

    // Newton-Raphson refinement: y_{n+1} = y_n * (2 - x * y_n)
    // Rewritten as: t = x*y - 2.0, then y = y * (-t) - 0 (using vConst0=0)
    // The negated form makes NaN detection easier: when x=0 and y=inf (or vice versa),
    // t=+NaN, and checking t<0 correctly skips the NaN case.
    if constexpr (max_iter > 0) {
        sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0;  // vConstFloatPrgm0 = 2.0f

        if constexpr (max_iter > 1) {
            // Two-iteration path (FP32 precision):
            sfpi::vFloat y1 = y * -t - sfpi::vConst0;     // First NR update
            // NaN guard: t>=0 means NaN occurred (0*inf), skip the refinement
            v_if (t < 0) {
                t = x * y1 - sfpi::vConstFloatPrgm0;      // Recompute residual
                y = y1 * -t - sfpi::vConst0;               // Second NR update
            }
            v_endif;
        } else {
            // Single-iteration path (BF16 precision):
            v_if (t < 0) {
                y = y * -t - sfpi::vConst0;
            }
            v_endif;
        }
    }

    return y;
}

// Fast 7-bit approximate reciprocal using SFPLOADMACRO pipeline.
// Throughput: 1 cycle per 32 elements. Uses only the hardware SFPARECIP approximation.
// The SFPLOADMACRO template is pre-configured in _init_reciprocal_fast_7b_ to:
//   Load -> SFPARECIP -> Store
inline void _calculate_reciprocal_fast_7b_(const int iterations) {
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_6, 0);  // Execute macro 0 on LREG0
    }
    TTI_SFPNOP;  // Pipeline drain
    TTI_SFPNOP;
}

// BF16 reciprocal with LSB correction, throughput: 3 cycles per 32 elements.
// Uses SFPLOADMACRO pipeline with three macro slots to achieve:
//   1. Load input, compute SFPARECIP
//   2. Apply MAD-based LSB correction for BF16 precision
//   3. Store corrected result
// The LSB correction uses the formula: y = arecip(x) | (1<<15), then
//   y = x*y - 1, extract upper 16 bits, add to y. This corrects the
//   least-significant bit of the BF16 representation.
inline void _calculate_reciprocal_fast_8b_3c_(const int iterations) {
    constexpr int x           = p_sfpu::LREG1;
    constexpr int t           = p_sfpu::LREG1;
    constexpr int offset      = 0;
    constexpr int prev_offset = -4 & 0x3ff;

    // L0 = 0x80000000 (used for setting bit 15 via MOD0_LO16_ONLY load)
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);
    // L7 = register index for indirect VD addressing
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, x);

    // Prologue: first two iterations fill the pipeline (2nd instruction is SFPNOP)
    const int fill_end = iterations < 2 ? iterations : 2;
#pragma GCC unroll 2
    for (int d = 0; d < fill_end; d++) {
        int y = 3 + (d % 3);  // Rotate through LREGs 3,4,5
        TT_SFPLOADMACRO((0 << 2) | (y & 3), 0, ADDR_MOD_7, offset | (y >> 2));
        TTI_SFPNOP;
        TT_SFPLOADMACRO((1 << 2) | (y & 3), 14, ADDR_MOD_6, offset | (y >> 2));
    }

    // Main loop: all three SFPLOADMACRO stages active simultaneously
#pragma GCC unroll 6
    for (int d = 2; d < iterations; d++) {
        int y = 3 + (d % 3);
        TT_SFPLOADMACRO((0 << 2) | (y & 3), 0, ADDR_MOD_7, offset | (y >> 2));
        TT_SFPLOADMACRO((2 << 2) | (t & 3), 9, ADDR_MOD_7, prev_offset | (t >> 2));
        TT_SFPLOADMACRO((1 << 2) | (y & 3), 14, ADDR_MOD_6, offset | (y >> 2));
    }

    // Fill gap with NOPs if iterations < 2
#pragma GCC unroll 2
    for (int d = iterations; d < 2; d++) {
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
    }

    // Epilogue: drain the pipeline
    const int drain_start = iterations < 2 ? 2 : iterations;
#pragma GCC unroll 2
    for (int d = drain_start; d < iterations + 2; d++) {
        TTI_SFPNOP;
        TT_SFPLOADMACRO((2 << 2) | (t & 3), 9, ADDR_MOD_6, prev_offset | (t >> 2));
        TTI_SFPNOP;
    }

    TTI_SFPNOP;
}

// FP32 reciprocal using replay buffer, throughput: 5 cycles per 32 elements.
// Computes: y = arecip(x), e = 1-x*y, t = e*e+e, t2 = t*e+e,
//           t2 = min(t2, 1.0), y = t2*y + y
// This achieves full 24-bit (FP32 mantissa) precision through a modified
// Newton-Raphson scheme that accumulates higher-order error terms.
inline void _calculate_reciprocal_fast_24b_5c_(const int iterations) {
    lltt::replay(0, 4);                             // Execute replay buffer slots 0-3 (prologue)
    TTI_SFPLOAD(7, 0, ADDR_MOD_6, 0);              // Load first element

#pragma GCC unroll 7
    for (int d = 0; d < iterations - 1; d++) {
        lltt::replay(0, 5);                         // Execute full 5-instruction replay sequence
    }

    TTI_SFPNOP;
    lltt::replay(1, 1);                             // Partial replay for drain
    TTI_SFPNOP;
    lltt::replay(3, 2);                             // Final drain

    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Dispatch function: selects the appropriate precision path based on template parameters.
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void _calculate_reciprocal_internal_(const int iterations) {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_reciprocal_fast_7b_(iterations);       // ~7-bit, 1 cycle/32 elements
    } else if constexpr (is_fp32_dest_acc_en) {
        _calculate_reciprocal_fast_24b_5c_(iterations);   // ~24-bit, 5 cycles/32 elements
    } else {
        _calculate_reciprocal_fast_8b_3c_(iterations);    // ~8-bit, 3 cycles/32 elements
    }
}

// Top-level entry point. Supports legacy_compat mode for backward compatibility.
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _calculate_reciprocal_(const int iterations) {
    if constexpr (legacy_compat) {
        _calculate_reciprocal_compat_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    } else {
        _calculate_reciprocal_internal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    }
}

// Initialization for 7-bit fast path.
// Configures SFPLOADMACRO template 0 to perform: Load -> SFPARECIP -> Store
inline void _init_reciprocal_fast_7b_() {
    TTI_SFPARECIP(0, 0, 12, 0);  // InstructionTemplate[0]: SFPARECIP

    // Configure macro 0 scheduling:
    // simple_bits: SFPARECIP in simple sub-unit, reading from template 0
    // store_bits: Store to L16 (lower 16 bits) with delay
    constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0);
    constexpr std::uint32_t mad_bits    = 0;
    constexpr std::uint32_t round_bits  = 0;
    constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (1 << 3) | 3;

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
    TTI_SFPCONFIG(0, 4, 0);

    // Misc: UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1
    TTI_SFPCONFIG(0x110, 8, 1);
}

// Initialization for 8-bit BF16 path.
// Configures three SFPLOADMACRO templates for the 3-cycle pipeline.
inline void _init_reciprocal_fast_8b_3c_() {
    constexpr int x = p_sfpu::LREG1;
    constexpr int t = p_sfpu::LREG1;

    TTI_SFPARECIP(0, 0, 12, 0);                                // Template[0]: SFPARECIP
    TTI_SFPMAD(p_sfpu::LCONST_0, p_sfpu::LCONST_0, 0, 13, 8); // Template[1]: x = 0*0 + y (indirect VD)
    TTI_SFPMAD(x, 0, p_sfpu::LCONST_neg1, 14, 0);             // Template[2]: y = x*y - 1
    TTI_SFPIADD(0, t, 15, sfpi::SFPIADD_MOD1_CC_NONE);        // Template[3]: y += t (integer add for bit correction)

    // Configure macro scheduling for all 3 SFPLOADMACRO slots
    // Macro 0: Load from SRCB, execute SFPARECIP+MAD, store to L0
    { /* macro 0 config */ }
    // Macro 1: Load with LO16_ONLY (sets bit 15), execute MAD for LSB correction
    { /* macro 1 config */ }
    // Macro 2: Store corrected result
    { /* macro 2 config */ }

    TTI_SFPCONFIG(0x700, 8, 1);  // Misc: StoreMod0=MOD0_FMT_SRCB, WaitForElapsedInstructions=1
}

// Initialization for 24-bit FP32 path.
// Configures four SFPLOADMACRO templates and loads a replay buffer with the
// 6-instruction sequence implementing the higher-order Newton-Raphson scheme.
inline void _init_reciprocal_fast_24b_5c_() {
    // Uses 4 LREGs: e=LREG0, t2=LREG1, z=LREG2, y=LREG3
    TTI_SFPARECIP(0, 0, 12, 0);                                      // Template[0]: SFPARECIP
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, 0, 13, 0);             // Template[1]: e = -e*y + 1.0
    TTI_SFPMAD(/*t2*/ 1, p_sfpu::LREG0, /*z*/ 2, 14, 0);            // Template[2]: t2 = t2*e + e (or z)
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, 15, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX); // Template[3]: min(t2, 1.0)

    // Configure 4 macros (y, e, t2, z) with appropriate scheduling
    // Macro 0 [y]: Load, SFPARECIP, Store
    // Macro 1 [e]: Load, SFPARECIP+L16, MAD(e=-e*y+1)
    // Macro 2 [t2]: SWAP(min), MAD(t2*e+e)
    // Macro 3 [z]: MAD(t2*y+y), Store

    TTI_SFPCONFIG(0xff0, 8, 1);  // All macros: UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1

    // Load the replay buffer with the 6-instruction main loop body
    load_replay_buf(0, 6, [/*captures*/] {
        TTI_SFPLOADMACRO(/* macro 0: y */);
        TTI_SFPLOADMACRO(/* macro 2: t2 */);
        TTI_SFPLOADMACRO(/* macro 1: e */);
        TTI_SFPMAD(/* e = -e*y + 1.0 */);
        TTI_SFPLOADMACRO(/* macro 3: z */);
        TTI_SFPLOADMACRO(/* macro 3: z */);
    });
}

// SFPI-level init: sets vConstFloatPrgm0 = 2.0f for Newton-Raphson (used by _sfpu_reciprocal_)
template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_() {
    if constexpr (!APPROXIMATION_MODE) {
        sfpi::vConstFloatPrgm0 = 2.0f;  // The "2" in Newton-Raphson: y = y * (2 - x*y)
    }
}

// Top-level init dispatch
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _init_reciprocal_() {
    if constexpr (!legacy_compat) {
        if constexpr (APPROXIMATION_MODE) {
            _init_reciprocal_fast_7b_();      // Configure 1-cycle pipeline
        } else if constexpr (is_fp32_dest_acc_en) {
            _init_reciprocal_fast_24b_5c_();  // Configure 5-cycle pipeline + replay buffer
        } else {
            _init_reciprocal_fast_8b_3c_();   // Configure 3-cycle pipeline
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

#### Wormhole B0 Low-Level SFPU Implementation

##### File: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`

##### Annotated Wormhole B0 SFPU Kernel Source
```cpp
#pragma once

#include "ckernel_sfpu_rsqrt_compat.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Core reciprocal using software Newton-Raphson (no SFPARECIP hardware instruction on Wormhole).
// max_iter=0: same as max_iter=1 currently (may become cheaper approximation in future)
// max_iter=1: sufficient for BF16/FP16 precision (<=0.5 ulps)
// max_iter=2: sufficient for FP32 precision (<=1 ulps)
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in) {
    // Step 1: Normalize input to [1.0, 2.0) range.
    // setman copies the mantissa bits of `in` into the mantissa field of -1.0,
    // effectively creating: negative_x = -(1.0 + mantissa(in)) = -x where x in [1.0, 2.0)
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in));

    // Step 2: Quadratic initial estimate via minimax polynomial (Sollya-optimized).
    // y = k2 + k1*(-x) + k0*(-x)^2 = k2 - k1*x + k0*x^2
    // where k0=0.3232325, k1=-(-0.3232325*2+1.4545459)=..., k2=2.121212
    // These coefficients minimize maximum relative error for 1/x over [1,2).
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;

    // Step 3: Compute scale factor.
    // Since in = x * 2^(in.Exp - 127), we need 1/in = (1/x) * 2^(127 - in.Exp).
    // The scale exponent should be 254 - in.Exp (after float bias).
    // Using bitwise NOT gives 255 - in.Exp, which is off by one but handles edge cases:
    //   in.Exp=0 (denorm/zero) -> scale=inf, in.Exp=255 (inf/NaN) -> scale=0
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);

    // Step 4: Complete quadratic estimate.
    y = sfpi::vConstFloatPrgm2 + y * negative_x;

    // Step 5: Clear mantissa from scale factor (keep only exponent and sign).
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0);

    // Step 6: First Newton-Raphson iteration.
    // t = 1 - x*y (residual; should converge to 0)
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;

    // Step 7: Adjust scale by *0.5 to correct the off-by-one from using ~Exp instead of 254-Exp.
    // Also handles edge cases: inf*0.5=inf, 0*0.5=0.
    scale *= 0.5f;

    // Step 8: Apply NR correction: y = y + y*t = y*(1+t) = y*(2-x*y_old)
    y = y + y * t;

    if constexpr (max_iter > 1) {
        // Step 9: Second NR iteration for FP32-level precision.
        t = sfpi::vConst1 + negative_x * y;
        y = y + y * t;
    }

    // Step 10: Apply scaling and restore original sign.
    y = y * scale;
    y = sfpi::setsgn(y, in);  // Copy sign from original input

    return y;
}

// Main calculation loop: iterates over sub-tiles in DST register.
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void _calculate_reciprocal_internal_(const int iterations) {
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];  // Read 32 elements from current DST position

        if constexpr (APPROXIMATION_MODE) {
            sfpi::dst_reg[0] = _sfpu_reciprocal_<0>(in);   // 0 NR iterations
        } else {
            if constexpr (is_fp32_dest_acc_en) {
                sfpi::dst_reg[0] = _sfpu_reciprocal_<2>(in); // 2 NR iterations (FP32)
            } else {
                sfpi::vFloat out = _sfpu_reciprocal_<1>(in);  // 1 NR iteration (BF16)
                sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(out, 0));
                // float_to_fp16b truncates to BF16 precision, avoiding rounding artifacts
            }
        }

        sfpi::dst_reg++;  // Advance to next 32-element sub-tile in DST
    }
}

// Top-level dispatch (same pattern as Blackhole)
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _calculate_reciprocal_(const int iterations) {
    if constexpr (legacy_compat) {
        _calculate_reciprocal_compat_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    } else {
        _calculate_reciprocal_internal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    }
}

// Initialize polynomial coefficients for the quadratic initial estimate.
// These are loaded into programmable constant registers accessible as vConstFloatPrgm{0,1,2}.
template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_() {
    // Minimax polynomial coefficients for 1/x over [1,2), computed via Sollya:
    sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;    // k0 (quadratic coefficient)
    sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;          // k1 (linear coefficient)
    sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;        // k2 (constant term)
}

// Init dispatch: on Wormhole, both legacy and non-legacy paths call _init_sfpu_reciprocal_.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _init_reciprocal_() {
    if constexpr (!legacy_compat) {
        _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Architecture | Description |
|---|---|---|
| `SFPARECIP` | Blackhole only | Hardware approximate reciprocal. Computes ~7-bit approximation of 1/x for 32 lanes in 1 cycle. |
| `SFPLOADMACRO` | Both | Pipelined macro execution: loads data from DST and schedules Simple/MAD/Round/Store sub-unit instructions. Key to achieving high throughput on Blackhole. |
| `SFPMAD` | Both | Multiply-Add: VD = VA * VB + VC. Used for Newton-Raphson iterations and LSB correction. |
| `SFPIADD` | Blackhole | Integer add on float representation. Used in BF16 path for LSB bit correction. |
| `SFPSWAP` | Blackhole | Vector min/max swap. Used in FP32 path to clamp NaN to 1.0. |
| `SFPCONFIG` | Blackhole | Configures SFPLOADMACRO templates (instruction scheduling, delays, format modes). |
| `SFPLOADI` | Both | Load immediate value to LREG. Used for constants (0x8000, register indices, polynomial coefficients). |
| `SFPNOP` | Both | No-operation. Required for pipeline scheduling and hazard avoidance. |
| `SFPLOAD` | Both | Load from DST to LREG (non-macro variant). |
| `setman` | Wormhole | SFPI intrinsic: set mantissa bits of a float. Used for input normalization. |
| `setsgn` | Both | SFPI intrinsic: set sign bit. Used to restore input sign to output. |
| `setexp` | Wormhole (software recip) | SFPI intrinsic: set exponent bits. |
| `exexp` | Wormhole (software recip) | SFPI intrinsic: extract exponent. |
| `approx_recip` | Blackhole | SFPI intrinsic wrapping `__builtin_rvtt_sfparecip`. |
| `reinterpret` | Both | SFPI intrinsic: reinterpret cast between vFloat/vInt/vUInt. |

#### SFPU Register Usage

**Wormhole B0:**
- `sfpi::dst_reg[0]`: Source and destination for each 32-element sub-tile. Iterates via `dst_reg++`.
- `vConstFloatPrgm0/1/2`: Hold the three minimax polynomial coefficients (k0, k1, k2).
- `vConst1`, `vConstNeg1`: Hardware constants used for NR iterations and input normalization.
- Temporary `vFloat` variables (`negative_x`, `y`, `scale`, `t`) are allocated to SFPU local registers by the compiler.

**Blackhole (SFPLOADMACRO paths):**
- `LREG0` through `LREG7`: Explicitly managed local registers.
  - 7-bit path: Only LREG0 used (load/store target).
  - 8-bit BF16 path: LREG0=0x80000000 constant, LREG1=x/t, LREG3-5=y (rotating), LREG7=indirect VD index.
  - 24-bit FP32 path: LREG0=e, LREG1=t2, LREG2=z, LREG3=y, LREG7=DST load target.
- `vConstFloatPrgm0`: Set to 2.0f for Newton-Raphson in the SFPI-level path.
- `vConst0`, `vConst1`, `vConstNeg1`, `LCONST_neg1`, `LCONST_0`, `LCONST_1`: Hardware constants used in MAD operations.
- DST register rows are accessed through SFPLOAD/SFPLOADMACRO address offsets.

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `cb_wait_front(c_0, 1)` to wait for the reader to produce an input tile, then `tile_regs_acquire()` to get exclusive DST register access.

2. **Unpack**: `copy_tile(c_0, 0, 0)` unpacks the tile from CB c_0 into DST register 0. The unpacker converts from the CB data format to the DST register format (FP32 if `fp32_dest_acc_en`, otherwise native format).

3. **SFPU Init** (once per kernel): `recip_tile_init<false>()` calls `recip_init<APPROX, DST_ACCUM_MODE, false>()` which:
   - **Wormhole**: Loads three polynomial coefficients into `vConstFloatPrgm0/1/2`.
   - **Blackhole**: Configures SFPLOADMACRO instruction templates and (for FP32) loads the replay buffer.

4. **SFPU Compute**: `recip_tile<false>(0)` calls `calculate_reciprocal<APPROX, DST_ACCUM_MODE, 8, false>()` which loops 8 times over sub-tiles (4 rows of 32 elements each):
   - **Wormhole**: Each iteration reads `dst_reg[0]`, computes the software reciprocal (normalize, polynomial estimate, Newton-Raphson), and writes back to `dst_reg[0]`, then advances `dst_reg++`.
   - **Blackhole (non-approx, BF16)**: Executes a pipelined SFPLOADMACRO sequence that loads each sub-tile, applies SFPARECIP, corrects the LSB, and stores back, at 3 cycles per sub-tile.
   - **Blackhole (non-approx, FP32)**: Uses replay buffer for a 5-cycle-per-sub-tile pipeline with higher-order NR error accumulation.
   - **Blackhole (approx)**: 1-cycle-per-sub-tile pipeline using only SFPARECIP.

5. **Pack**: After `tile_regs_commit()` and `tile_regs_wait()`, `pack_tile(0, c_2)` packs the result from DST[0] into CB c_2 in the output data format.

6. **Release**: `cb_pop_front(c_0, 1)` frees the input CB slot, `tile_regs_release()` releases DST registers. After the inner loop, `cb_push_back(c_2, per_core_block_dim)` signals the writer that output tiles are ready.

#### SFPU Configuration

- **Math Fidelity**: Always `MathFidelity::HiFi4` (set by program factory).
- **Math Approx Mode**: Always `false` for RECIP (`get_op_approx_mode` returns false for all default ops). This means the full-precision path is always selected.
- **FP32 Dest Accumulation**: Controlled by `args.fp32_dest_acc_en`. When enabled:
  - Unpacker converts to FP32 in DST registers.
  - Blackhole uses the 24-bit/5-cycle pipeline. Wormhole uses 2 NR iterations.
  - Packer converts back from FP32.
- **BFP8 Pack Precise**: Controlled by `args.bfp8_pack_precise`. Enables higher precision when packing to BFP8 format.
- **Legacy Compat**: The `recip_tile<false>` call passes `legacy_compat=false`, selecting the optimized (non-legacy) code paths.

#### Hardware Compatibility Notes

| Feature | Wormhole B0 | Blackhole |
|---|---|---|
| **SFPARECIP instruction** | Not available | Available (hardware ~7-bit approximation) |
| **Initial estimate** | Software quadratic polynomial via Sollya-optimized minimax coefficients (k0, k1, k2) using `setman`/bitwise NOT for normalization | Hardware `approx_recip()` intrinsic |
| **Newton-Raphson** | Software NR using SFPI vFloat arithmetic; 1 iteration for BF16, 2 for FP32 | Blackhole SFPI path also uses NR; SFPLOADMACRO paths use MAD-based refinement |
| **BF16 precision path** | 1 NR iteration + `float_to_fp16b` truncation | SFPLOADMACRO 3-cycle pipeline with LSB correction (no explicit BF16 truncation needed) |
| **FP32 precision path** | 2 NR iterations | SFPLOADMACRO 5-cycle pipeline with replay buffer, higher-order error terms, and NaN clamping via SFPSWAP |
| **Throughput (approx)** | N/A (software loop, many cycles per sub-tile) | 1 cycle per 32 elements |
| **Throughput (BF16)** | ~10+ cycles per 32 elements (estimated, software NR) | 3 cycles per 32 elements |
| **Throughput (FP32)** | ~15+ cycles per 32 elements (estimated, 2 NR iterations) | 5 cycles per 32 elements |
| **Programmable constants used** | 3 (vConstFloatPrgm0/1/2 for polynomial) | 1 (vConstFloatPrgm0=2.0 for SFPI path) or 0 (SFPLOADMACRO paths use templates) |
| **Edge case handling** | Bitwise NOT of exponent gives correct inf/zero behavior; `setsgn` restores sign | SFPARECIP natively returns 0 for inf and inf for 0; NaN guard via `t<0` check |

## External Knowledge Sources

### DeepWiki References
- **tenstorrent/tt-metal**: Queried for program factory structure, compute kernel dispatch, and define chain for RECIP.
- **tenstorrent/tt-llk**: Queried for `_sfpu_reciprocal_`, `_calculate_reciprocal_`, Newton-Raphson implementation details, and architecture differences.
- **tenstorrent/sfpi**: Queried for SFPI reciprocal intrinsics (`approx_recip`, `setman`, `setsgn`, `setexp`, `exexp`).
- **tenstorrent/tt-isa-documentation**: Queried for SFPARECIP instruction semantics, SFPLOADMACRO pipeline behavior, and encoding details.

### Confluence References
Not consulted for this analysis. The DeepWiki and ISA documentation queries provided sufficient detail on the SFPARECIP instruction and SFPLOADMACRO pipeline.

### Glean References
Not consulted for this analysis. The open-source tt_llk sources provided complete implementation details for both architectures.

## File Reference Index

| File | Role |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Program factory: creates kernels, CBs, sets runtime args |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp` | Factory class declarations and shared_variables_t types |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Op type to kernel path mapping, define generation, approx mode |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | `UnaryOpType::RECIP` enum definition |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader kernel source |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer kernel source |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Compute kernel source (generic SFPU dispatch) |
| `tt_metal/hw/inc/api/compute/eltwise_unary/recip.h` | Compute API: `recip_tile_init`, `recip_tile` |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Conditional include for `recip.h` via `SFPU_OP_RECIP_INCLUDE` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h` | Blackhole architecture wrapper |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h` | Wormhole B0 architecture wrapper |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h` | Blackhole low-level SFPU implementation (SFPARECIP + SFPLOADMACRO) |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h` | Wormhole B0 low-level SFPU implementation (software NR) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` | Macro framework for SFPU dispatch |
