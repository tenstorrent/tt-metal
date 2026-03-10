# RELU Operation Analysis

## Operation Overview

| Property | Value |
|---|---|
| **Operation Name** | RELU |
| **Operation Type** | Unary Elementwise (SFPU) |
| **UnaryOpType Enum** | `UnaryOpType::RELU` |
| **Mathematical Definition** | `relu(x) = max(0, x)` |
| **Program Factory** | `UnaryProgramFactory` (shared with all standard unary ops) |
| **Compute Kernel** | `eltwise_sfpu.cpp` (generic SFPU dispatch kernel) |
| **SFPU Kernel** | `_relu_min_` with threshold = 0 |
| **Supported Data Types** | BFLOAT16, FLOAT32, INT32 |
| **Approximation Mode** | `false` (no approximation needed for relu) |

## Program Factory Analysis

### Source File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

### Factory Pattern
RELU uses the shared `UnaryProgramFactory::create()` method. There are two factory variants:

1. **`UnaryProgramFactory`** -- Standard interleaved layout, splits work across all compute cores using `split_work_to_cores`. Creates two core groups if tiles do not divide evenly.
2. **`UnarySubCoreGridProgramFactory`** -- Used when `sub_core_grids` is specified in the operation params. Supports custom core allocation.

### Operation Parameters (`UnaryParams`)

```cpp
struct UnaryParams {
    const std::vector<EltwiseUnaryWithParam> op_chain;  // Chain of unary ops to fuse
    const tt::tt_metal::DataType output_dtype;
    const tt::tt_metal::MemoryConfig output_memory_config;
    const bool fp32_dest_acc_en = false;                // FP32 accumulation in DEST registers
    const bool preserve_fp32_precision = false;         // Unpack to DEST in FP32 mode
    const bool bfp8_pack_precise = false;               // Higher precision BFP8 packing
    const std::optional<CoreRangeSet> sub_core_grids;   // Optional custom core grids
};
```

For RELU specifically:
- `op_chain` contains a single `EltwiseUnaryWithParam` with type `UnaryOpType::RELU` and no parameters
- `math_approx_mode` is computed as `false` (RELU returns `false` from `get_op_approx_mode`)
- No packed scalar runtime args are needed (RELU has no threshold parameter at the op level; the threshold of 0 is hardcoded in the API)

### Compile-Time Define Generation

The program factory calls `utils::get_block_defines(args.op_chain, "0", "0", input.dtype())` which produces:

For float types:
```cpp
"SFPU_OP_CHAIN_0"        -> "SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0"
"SFPU_OP_CHAIN_0_INIT_0" -> "relu_tile_init();"
"SFPU_OP_CHAIN_0_FUNC_0" -> "        relu_tile(0);"
"SFPU_OP_RELU_FAMILY_INCLUDE" -> "1"
```

For INT32:
```cpp
"SFPU_OP_CHAIN_0_INIT_0" -> "relu_tile_init();"
"SFPU_OP_CHAIN_0_FUNC_0" -> "relu_tile_int32(0);"
"SFPU_OP_RELU_FAMILY_INCLUDE" -> "1"
"INP_INT32" -> "1"
```

The `SFPU_OP_RELU_FAMILY_INCLUDE` define causes `sfpu_split_includes.h` to include `api/compute/eltwise_unary/relu.h`, which provides the `relu_tile()` and `relu_tile_init()` functions.

### Compute Kernel Path Resolution

```cpp
// In get_compute_kernel_path():
// RELU falls through to the default case
default: return "eltwise_sfpu.cpp";
```

The full path becomes: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

### Circular Buffer Configuration

| CB Index | Name | Size | Data Format | Purpose |
|---|---|---|---|---|
| `c_0` | Input | 2 tiles | Input data format | Source tiles from reader |
| `c_2` | Output | 2 tiles | Output data format | Result tiles for writer |

Note: CB `c_1` (tmp0) is NOT allocated for RELU -- it is only allocated for HARDSHRINK, CBRT, and LOGIT operations.

### Core Distribution

```cpp
auto [num_cores, all_cores, core_group_1, core_group_2,
      num_pages_per_core_group_1, num_pages_per_core_group_2] =
    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);
```

Work is split across all available compute cores. If tiles do not divide evenly, two core groups are created with different tile counts. Each core group gets its own compute kernel handle with the appropriate `per_core_block_cnt`.

### Runtime Arguments

| Kernel | Arg 0 | Arg 1 | Arg 2 |
|---|---|---|---|
| Reader | `src_buffer->address()` | `num_pages_per_core` | `num_pages_written` (start offset) |
| Writer | `dst_buffer->address()` | `num_pages_per_core` | `num_pages_written` (start offset) |
| Compute | `packed_scalar1` (= 0) | `packed_scalar2` (= 0) |

For RELU, both packed scalars are 0 because RELU does not use runtime scalar parameters.

## Kernel Implementations

### Reader Kernel

#### File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

#### Annotated Source
```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime arguments: buffer address, number of pages to process, starting page ID
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    // Compile-time args encode tensor accessor configuration (interleaved vs sharded, bank mapping)
    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_id_in0 = 0; // CB index 0 = input circular buffer

    // Page size is read from the CB interface -- works for both tile and row-major layouts
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    constexpr uint32_t onepage = 1; // Process one page at a time

    // Construct a TensorAccessor that knows how to map page IDs to NoC addresses
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

// Optionally support backwards iteration (used by some ops, not RELU)
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) { // Forward iteration for RELU
#endif
        cb_reserve_back(cb_id_in0, onepage);        // Wait for space in input CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0); // Get L1 write pointer
        noc_async_read_page(i, s, l1_write_addr);   // Issue NoC read from DRAM to L1
        noc_async_read_barrier();                    // Wait for read to complete
        cb_push_back(cb_id_in0, onepage);            // Signal tile is available to compute
    }
}
```

### Writer Kernel

#### File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

#### Annotated Source
```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime arguments: destination buffer address, page count, start offset
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    // Compile-time arg 0: output CB index (c_2 for RELU)
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    // Compile-time args starting at index 1: tensor accessor configuration for destination
    constexpr auto dst_args = TensorAccessorArgs<1>();

    // Page size from CB interface
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    // For sharded output: just wait for all pages to be ready (compute writes directly to output CB)
    cb_wait_front(cb_id_out, num_pages);
#else
    constexpr uint32_t onepage = 1;

    // Construct tensor accessor for destination buffer
    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) { // Forward iteration for RELU
#endif
        cb_wait_front(cb_id_out, onepage);           // Wait for compute to produce a tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out); // Get L1 read pointer
        noc_async_write_page(i, s, l1_read_addr);   // Issue NoC write from L1 to DRAM
        noc_async_writes_flushed();                  // Ensure write is sent
        cb_pop_front(cb_id_out, onepage);            // Free the CB slot for compute
    }
    noc_async_write_barrier();                       // Final barrier: all writes complete
#endif
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
#include "api/compute/common.h"          // Common compute API (init, tile management)
#include "api/compute/tile_move_copy.h"  // copy_tile: unpacks tile from CB to DST register
#include "api/compute/eltwise_unary/eltwise_unary.h"  // Unary operation infrastructure
#include "api/compute/eltwise_unary/sfpu_split_includes.h" // Conditional includes based on SFPU_OP_*_INCLUDE defines
// For RELU: SFPU_OP_RELU_FAMILY_INCLUDE=1 causes inclusion of relu.h
// which provides relu_tile(), relu_tile_init(), relu_tile_int32()
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    // Compile-time args set by program factory:
    // arg 0: number of tile blocks this core processes
    // arg 1: tiles per block (always 1 for standard unary)
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    // Initialize SFPU pipeline: configures unpack (CB c_0 -> SRC) and pack (DST -> CB c_2)
    // This sets up data format conversions and register configurations
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    // Outer loop: iterate over tile blocks assigned to this core
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve output space for this block before processing
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to DST (destination) registers
            // This synchronizes with the packer -- ensures previous tile is packed before overwriting
            tile_regs_acquire();

            // Wait for reader to push a tile into input CB
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Unpack tile from input CB (c_0) slot 0 into DST register 0
            // This invokes the unpacker RISC-V to move data from L1 CB into source registers,
            // then the FPU copies from SRC to DST
            copy_tile(tt::CBIndex::c_0, 0, 0);

            // SFPU_OP_CHAIN_0 expands to the init + function calls for the operation chain.
            // For RELU this becomes:
            //   relu_tile_init();    -- configures SFPU for relu_min operation type
            //   relu_tile(0);        -- applies relu to DST register tile 0
            // The init is called per-tile here (inside the loop), which is safe but
            // slightly redundant; it sets SfpuType::relu_min in the LLK state
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Signal that DST registers are written and ready for packing
            tile_regs_commit();

            // Wait for packer to be ready (previous pack must complete)
            tile_regs_wait();

            // Pack tile from DST register 0 into output CB (c_2)
            // The packer converts from DST format to the output data format
            pack_tile(0, tt::CBIndex::c_2);

            // Release the consumed input tile back to the reader
            cb_pop_front(tt::CBIndex::c_0, 1);

            // Release DST registers for the next iteration
            tile_regs_release();
        }
        // Push the completed block to the writer
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU API Layer

The compute kernel calls `relu_tile(0)`, defined in `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h`:

```cpp
// relu_tile dispatches to _relu_min_ with threshold = 0, implementing max(x, 0)
ALWI void relu_tile(uint32_t idst) {
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, idst, 0));
}

// For INT32 inputs, an INT-specific variant is used
ALWI void relu_tile_int32(uint32_t idst) {
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN_INT(_relu_min_, RC, APPROX, idst, 0));
}

// Initialization configures the SFPU for relu_min operation type
ALWI void relu_tile_init() {
    MATH(SFPU_UNARY_KERNEL_INIT(relu_min, APPROX));
}
```

The key design insight: **RELU is implemented as `relu_min` with threshold = 0**. The `relu_min(x, L)` function computes `max(x, L)`. When `L = 0`, this becomes `max(x, 0) = relu(x)`. This reuse of `relu_min` means RELU shares its SFPU kernel with the `RELU_MIN` operation.

#### Macro Expansion Chain

`SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, idst, 0)` expands to:

```cpp
_llk_math_eltwise_unary_sfpu_params_<APPROX>(
    ckernel::sfpu::_relu_min_<sfpi::vFloat, APPROX, 8, uint32_t>,
    idst, (int)VectorMode::RC, 0)
```

This calls the LLK dispatch function which:
1. Sets the DST write address for the target tile
2. Stalls until the SFPU is available
3. In `VectorMode::RC` mode, iterates over all 4 faces of the 32x32 tile
4. For each face, calls the SFPU kernel function which processes 8 rows (4 rows per iteration x 2 iterations per face via `SETRWC` advancement)

#### SFPU Kernel File (Wormhole B0)
`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h`

#### Annotated SFPU Kernel Source (Wormhole B0)
```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_load_config.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel
{
namespace sfpu
{

template <typename T>
constexpr bool is_supported_relu_type_v = std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>;

// Leaky ReLU -- included here because it is part of the relu family
// Uses raw TTI instructions for performance: loads value, checks sign,
// conditionally multiplies by slope, stores result
template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope)
{
    // Load slope value into LREG2 (two 16-bit halves)
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);      // Lower 16 bits
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);           // Upper 16 bits
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        // Load current element from DST register into LREG0
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        // Set condition code if LREG0 is negative (sign bit check)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);
        // Conditionally multiply: LREG0 = LREG0 * LREG2 (x * slope) -- only executes if CC is set
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        // Clear condition code
        TTI_SFPENCC(0, 0, 0, 0);
        // Store result back to DST register
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        sfpi::dst_reg++;  // Advance to next row in the face
    }
}

// Helper for relu_max: clamps to [0, threshold]
sfpi_inline sfpi::vFloat _relu_max_body_(sfpi::vFloat val, sfpi::vFloat threshold)
{
    sfpi::vFloat result = val;
    v_if (result > threshold)  // Clamp to upper bound
    {
        result = threshold;
    }
    v_endif;
    v_if (result < 0.0f)      // Clamp negative to zero
    {
        result = 0.0f;
    }
    v_endif;
    return result;
}

// relu_max implementation: processes all iterations
template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_impl_(const int iterations, VecType threshold)
{
    for (int d = 0; d < iterations; d++)
    {
        VecType result = sfpi::dst_reg[0];
        v_if (result > threshold) { result = threshold; }
        v_endif;
        v_if (result < 0) { result = 0; }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// relu_max wrapper: converts threshold type and dispatches to implementation
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_max_(T threshold)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>,
                  "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            v_threshold = static_cast<int>(Converter::as_float(threshold));
        }
        else
        {
            v_threshold = Converter::as_float(threshold);
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>,
                      "Threshold type must be float or uint32_t");
    }

    _relu_max_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

// THIS IS THE CORE RELU KERNEL FUNCTION
// _relu_min_impl_ computes max(x, threshold) for each element
// For standard RELU, threshold = 0, so this computes max(x, 0)
//
// On Wormhole B0, this uses raw TTI (Tenstorrent Instruction) intrinsics
// rather than the SFPI v_if/v_endif abstraction, for a hardware-optimized
// comparison-and-swap approach
template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, [[maybe_unused]] VecType threshold, int sfpload_instr_mod)
{
    for (int d = 0; d < iterations; d++)
    {
        // Step 1: Load the current element from DST register into LREG0
        // sfpload_instr_mod controls the load mode: DEFAULT for float, INT32_2S_COMP for int32
        TTI_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, 0);

        // Step 2: Copy the threshold value from LREG2 to LREG1
        // LREG2 was pre-loaded with the threshold (0 for RELU) in the wrapper function
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);

        // Step 3: SFPSWAP with mod bit 1 -- this is the key operation
        // SFPSWAP compares LREG1 and LREG0 in sign+magnitude format and swaps them
        // such that LREG1 gets the maximum and LREG0 gets the minimum
        // For RELU: if x >= 0, LREG1 = x (the max); if x < 0, LREG1 = 0 (the threshold)
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);

        // Step 4: Store the maximum (LREG1) back to the DST register
        TTI_SFPSTORE(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, 0);

        sfpi::dst_reg++;  // Advance to next row in the face
    }
}

// _relu_min_ wrapper: loads threshold into LREG2 and calls the implementation
// For RELU: threshold = 0 (passed as uint32_t from relu_tile)
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>,
                  "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    // Convert threshold from 2's complement to sign+magnitude if negative (for int path)
    int scalar = threshold;
    if (scalar < 0)
    {
        scalar  = -scalar;
        int res = 0x80000000 | (scalar & 0x7FFFFFFF);
        scalar  = res;
    }
    int sfpload_instr_mod = DEFAULT; // Default = float mode

    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            // For INT32: load threshold as integer in sign+magnitude format
            _sfpu_load_imm32_(p_sfpu::LREG2, scalar);
            sfpload_instr_mod = INT32_2S_COMP; // Load/store in 2's complement mode
        }
        else
        {
            // For float: load threshold as raw float bits
            // For RELU, threshold = 0 which is 0x00000000 in both float and int
            _sfpu_load_imm32_(p_sfpu::LREG2, threshold);
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>,
                      "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold, sfpload_instr_mod);
}

} // namespace sfpu
} // namespace ckernel
```

#### SFPU Kernel File (Blackhole)
`tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h`

#### Annotated SFPU Kernel Source (Blackhole)

The Blackhole implementation differs significantly in `_relu_min_impl_` -- it uses the SFPI high-level abstraction (`v_if`/`v_endif`) instead of raw TTI instructions:

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel
{
namespace sfpu
{

template <typename T>
constexpr bool is_supported_relu_type_v = std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>;

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope)
{
    // Same as Wormhole except uses ADDR_MOD_7 instead of ADDR_MOD_3
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

sfpi_inline sfpi::vFloat _relu_max_body_(sfpi::vFloat val, sfpi::vFloat threshold)
{
    sfpi::vFloat result = val;
    v_if (result > threshold) { result = threshold; }
    v_endif;
    v_if (result < 0.0f) { result = 0.0f; }
    v_endif;
    return result;
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_impl_(const int iterations, VecType threshold)
{
    for (int d = 0; d < iterations; d++)
    {
        VecType result = sfpi::dst_reg[0];
        v_if (result > threshold) { result = threshold; }
        v_endif;
        v_if (result < 0) { result = 0; }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_max_(T threshold)
{
    // Same wrapper logic as Wormhole
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>);
    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>) { v_threshold = threshold; }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
            v_threshold = static_cast<int>(Converter::as_float(threshold));
        else
            v_threshold = Converter::as_float(threshold);
    }
    _relu_max_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

// BLACKHOLE _relu_min_impl_: uses SFPI v_if/v_endif instead of TTI SFPSWAP
// This is a simpler, more portable implementation
template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, VecType threshold)
{
    for (int d = 0; d < iterations; d++)
    {
        VecType a = sfpi::dst_reg[0];  // Load current element from DST
        v_if (a < threshold)           // Predicated comparison: per-lane condition codes
        {
            sfpi::dst_reg[0] = threshold;  // Replace values below threshold with threshold
        }
        v_endif;
        sfpi::dst_reg++;               // Advance to next row
    }
}

// Blackhole _relu_min_ wrapper: simpler than Wormhole (no sign+magnitude conversion needed)
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>);
    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>) { v_threshold = threshold; }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
            v_threshold = static_cast<int>(threshold);
        else
            v_threshold = Converter::as_float(threshold);
    }
    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

} // namespace sfpu
} // namespace ckernel
```

#### SFPU Instructions Used

**Wormhole B0 (`_relu_min_impl_`):**

| Instruction | Description |
|---|---|
| `TTI_SFPLOAD` | Loads a value from the DST register file into an SFPU local register (LREG0). The `sfpload_instr_mod` parameter selects float (DEFAULT) or integer (INT32_2S_COMP) load mode. |
| `TTI_SFPMOV` | Copies a value from one SFPU local register to another. Here, copies threshold from LREG2 to LREG1. |
| `TTI_SFPSWAP` | Compares two SFPU local registers in sign+magnitude format and swaps them so that the larger value ends up in the first operand (LREG1) and the smaller in the second (LREG0). The mod bit `1` selects this max/min swap behavior. This is the key instruction that makes Wormhole RELU a single-instruction comparison. |
| `TTI_SFPSTORE` | Stores a value from an SFPU local register (LREG1) back to the DST register file. |
| `_sfpu_load_imm32_` | Helper that uses two `TT_SFPLOADI` instructions (mod 10 for lower 16 bits, mod 8 for upper 16 bits) to load a 32-bit immediate into an SFPU local register. Used to pre-load the threshold into LREG2. |

**Blackhole (`_relu_min_impl_`):**

| Instruction / Intrinsic | Description |
|---|---|
| `sfpi::dst_reg[0]` (read) | Loads a value from the DST register file into an SFPI vector variable. Compiles to `SFPLOAD`. |
| `v_if (a < threshold)` | Predicated comparison using SFPI condition codes. Compiles to `SFPSETCC` or `SFPXFCMPS`/`SFPXICMPS` depending on type. Sets per-lane flags for lanes where condition is true. |
| `sfpi::dst_reg[0] = threshold` | Conditionally stores threshold to DST for lanes where the condition is active. Compiles to `SFPSTORE` with lane predication via `SFPENCC`. |
| `v_endif` | Clears the condition code / restores unconditional execution. |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| **LREG0** | (Wormhole) Holds the input value loaded from DST. After SFPSWAP, contains the minimum of (input, threshold). |
| **LREG1** | (Wormhole) Holds the threshold value (copied from LREG2). After SFPSWAP, contains the maximum of (input, threshold) -- this is the RELU result. |
| **LREG2** | (Wormhole) Pre-loaded with the threshold value (0 for RELU) before the iteration loop. Remains constant across all iterations. |
| **DST registers** | Both architectures read input from and write results to the DST register file. DST is organized as 4 faces of 16x16 elements within a 32x32 tile. |
| **Condition codes** | (Blackhole) Per-lane predication flags set by `v_if` comparisons. Used to selectively update only the lanes where `x < threshold`. |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `cb_wait_front(c_0, 1)` to wait for the reader to push a tile into the input circular buffer.

2. **Unpack to DST**: `copy_tile(c_0, 0, 0)` unpacks the tile from the input CB into DST register 0. The unpacker handles data format conversion (e.g., BF16 to FP32 in DST).

3. **SFPU initialization**: `relu_tile_init()` calls `llk_math_eltwise_unary_sfpu_init<SfpuType::relu_min, APPROX>()` which configures the SFPU for the relu_min operation type.

4. **SFPU dispatch**: `relu_tile(0)` calls `_llk_math_eltwise_unary_sfpu_params_` which:
   - Sets DST write address for tile index 0
   - Sets address modifier base
   - Issues `TTI_STALLWAIT(STALL_SFPU, MATH)` to synchronize SFPU availability
   - In `VectorMode::RC` mode, loops over all 4 faces:
     - Calls `_relu_min_<vFloat, APPROX, 8, uint32_t>(0)` which:
       - Pre-loads threshold (0) into LREG2 via `_sfpu_load_imm32_`
       - Loops 8 iterations (8 rows per face half)
       - **Wormhole**: For each row: SFPLOAD -> SFPMOV -> SFPSWAP -> SFPSTORE (4 instructions per row)
       - **Blackhole**: For each row: load from dst_reg, compare, conditionally write threshold
     - After the kernel function returns, `TTI_SETRWC` advances the DST pointer by 8 rows twice (to cover 16 rows = one face)
   - After all 4 faces: `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)` to wait for SFPU completion
   - Clears address modifiers

5. **Pack to output CB**: After `tile_regs_commit()` and `tile_regs_wait()`, `pack_tile(0, c_2)` packs the result from DST register 0 into the output circular buffer, converting back to the output data format.

6. **Release and advance**: `cb_pop_front(c_0, 1)` frees the input CB slot. `cb_push_back(c_2, per_core_block_dim)` signals the writer that results are ready.

#### SFPU Configuration

- **Math fidelity**: `MathFidelity::HiFi4` (highest fidelity, though RELU is exact and does not depend on fidelity)
- **Math approximation mode**: `false` -- RELU does not use any approximation
- **FP32 destination accumulation**: Controlled by `args.fp32_dest_acc_en` parameter
- **Unpack to dest mode**: Either `Default` or `UnpackToDestFp32` (when `preserve_fp32_precision` is true)
- **BFP8 pack precise**: Controlled by `args.bfp8_pack_precise`
- **SfpuType**: `SfpuType::relu_min` (used for LLK init state configuration)

#### Hardware Compatibility Notes

**Wormhole B0 vs Blackhole -- Key Differences in RELU Implementation:**

1. **Algorithm approach**:
   - **Wormhole**: Uses `SFPSWAP` instruction with mod bit 1 for hardware-accelerated comparison and swap in sign+magnitude format. This is a 4-instruction-per-element approach (LOAD, MOV, SWAP, STORE) that leverages dedicated comparison hardware.
   - **Blackhole**: Uses SFPI `v_if`/`v_endif` conditional execution with predicated store. This is a higher-level, more portable approach that compiles down to SFPLOAD + comparison + conditional SFPSTORE.

2. **Threshold loading**:
   - **Wormhole**: Must pre-load threshold into LREG2 using `_sfpu_load_imm32_` and handle 2's complement to sign+magnitude conversion for integer types.
   - **Blackhole**: Converts threshold to the appropriate vector type directly, without sign+magnitude conversion.

3. **Address modifiers**:
   - **Wormhole**: Uses `ADDR_MOD_3` for DST register addressing in leaky relu.
   - **Blackhole**: Uses `ADDR_MOD_7` for the same operation.

4. **`_relu_min_impl_` signature**:
   - **Wormhole**: Takes an extra `int sfpload_instr_mod` parameter to switch between float and int32 load modes.
   - **Blackhole**: Does not need this parameter; uses the SFPI abstraction which handles type dispatch internally.

5. **`ckernel_sfpu_load_config.h`**:
   - **Wormhole**: Includes this header for `_sfpu_load_imm32_` helper.
   - **Blackhole**: Does not need this header.

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Confirmed RELU dispatch path, compute kernel selection, define generation, and program factory structure.
- `tenstorrent/tt-llk`: Documented the `_relu_min_` implementation in both Wormhole and Blackhole variants, and the three-stage pipeline architecture.
- `tenstorrent/tt-isa-documentation`: Identified SFPSWAP, SFPSETCC, SFPENCC, SFPLOAD, SFPSTORE as the relevant SFPU instructions for RELU.
- `tenstorrent/sfpi`: Confirmed the `v_if`/`v_endif` SFPI abstraction for conditional execution and how condition codes work for comparisons.

### Confluence References
Not consulted -- DeepWiki provided sufficient SFPU instruction detail for RELU analysis.

### Glean References
Not consulted -- the RELU kernel implementation is fully documented in open-source code and DeepWiki.

## File Inventory

| File | Role |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Program factory: creates program, circular buffers, and registers all three kernels |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp` | Program factory header: defines `UnaryProgramFactory` and `UnarySubCoreGridProgramFactory` structs |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_device_operation_types.hpp` | Defines `UnaryParams` and `UnaryInputs` structs |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | Defines `UnaryOpType` enum including `RELU` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Utility functions: `get_block_defines`, `get_compute_kernel_path`, `get_op_approx_mode` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader dataflow kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer dataflow kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Generic SFPU compute kernel |
| `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` | Compute API: `relu_tile()`, `relu_tile_init()`, `relu_tile_int32()` |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Conditional include mechanism for SFPU op families |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` | LLK macro definitions for SFPU dispatch |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` | `_llk_math_eltwise_unary_sfpu_params_` dispatch function |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` | Wormhole B0 SFPU kernel: `_relu_min_`, `_relu_max_`, `_calculate_lrelu_` |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h` | Blackhole SFPU kernel: `_relu_min_`, `_relu_max_`, `_calculate_lrelu_` |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h` | Metal-layer SFPU relu wrapper (uses SFPI `v_if` abstraction) |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_load_config.h` | `_sfpu_load_imm32_` helper for loading 32-bit immediates |
