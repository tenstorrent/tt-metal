# SILU Operation Analysis

## Operation Overview

| Property | Value |
|---|---|
| **Operation Name** | SILU (Sigmoid Linear Unit) |
| **TTNN API** | `ttnn::silu` |
| **Aliases** | `ttnn::swish` (Swish is mathematically identical to SiLU) |
| **UnaryOpType** | `UnaryOpType::SILU` |
| **Category** | Eltwise Unary / Activation Function |
| **Mathematical Definition** | `silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))` |
| **Program Factory** | `UnaryProgramFactory` (also `UnarySubCoreGridProgramFactory` for sub-core-grid variant) |
| **Compute Kernel** | `eltwise_sfpu.cpp` (shared generic SFPU kernel) |
| **SFPU Kernel** | `ckernel_sfpu_silu.h` (delegates to `ckernel_sfpu_sigmoid.h`) |
| **Parameters** | None (no scalar parameters) |
| **Approximation Mode** | Always `false` (no approximation mode for SILU) |

## TTNN API Registration

SILU is registered as a simple unary operation with no parameters:

```cpp
// In ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
REGISTER_UNARY_OPERATION(silu, SILU);
```

This expands to:
```cpp
constexpr auto silu = ttnn::register_operation<
    "ttnn::silu",
    ttnn::operations::unary::ExecuteUnary<ttnn::operations::unary::UnaryOpType::SILU>>();
```

The `ExecuteUnary` template provides the signature:
```cpp
static Tensor invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
```

Swish is separately registered as a distinct struct but dispatches `UnaryOpType::SILU` internally:
```cpp
constexpr auto swish = ttnn::register_operation<"ttnn::swish", ttnn::operations::unary::Swish>();
```

## Program Factory

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

The program factory is shared across all unary operations. SILU uses the default `UnaryProgramFactory` (interleaved layout). A `UnarySubCoreGridProgramFactory` variant exists for sub-core-grid scheduling.

### Program Factory Selection

The `UnaryDeviceOperation::select_program_factory` method picks the factory based on tensor layout and attributes. For standard interleaved tensors, `UnaryProgramFactory` is selected. For sharded tensors, `UnaryShardedProgramFactory` is used.

### Circular Buffer Configuration

| CB Index | Name | Double-Buffered | Size | Purpose |
|---|---|---|---|---|
| `c_0` | Input | Yes (2 pages) | `2 * tile_size(input_format)` | Holds input tiles from DRAM |
| `c_2` | Output | Yes (2 pages) | `2 * tile_size(output_format)` | Holds output tiles for write-back |

SILU does not require the temporary buffer `c_1` (that is reserved for HARDSHRINK and LOGIT only).

### Compile-Time Defines

The program factory generates these compile-time defines for SILU:

| Define | Value | Purpose |
|---|---|---|
| `SFPU_OP_INIT_0` | `silu_tile_init();` | Initialization macro inserted into the compute kernel |
| `SFPU_OP_FUNC_0` | `silu_tile(0);` | Per-tile SFPU operation macro inserted into the compute kernel |
| `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` | `1` | Triggers `#include "api/compute/compute_kernel_api.h"` in the split-include header |

The define `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` is the **fallback** -- SILU does not have a dedicated split-include macro. Instead, it relies on the full `compute_kernel_api.h` header which includes `silu_tile` and `silu_tile_init` declarations.

### Work Distribution

The factory uses `split_work_to_cores` to distribute tiles across the compute grid:
- Tiles are divided as evenly as possible across all available cores
- Two core groups handle remainder: `core_group_1` gets `num_pages_per_core_group_1` tiles, `core_group_2` gets `num_pages_per_core_group_2`
- Each core processes tiles sequentially, one at a time (`per_core_block_size = 1`)

### Compute Configuration

```cpp
tt::tt_metal::ComputeConfig{
    .math_fidelity = MathFidelity::HiFi4,
    .fp32_dest_acc_en = args.fp32_dest_acc_en,
    .unpack_to_dest_mode = unpack_to_dest_mode,
    .bfp8_pack_precise = args.bfp8_pack_precise,
    .math_approx_mode = false,  // SILU always returns false from get_op_approx_mode
    .compile_args = {num_pages_per_core, 1},
    .defines = unary_defines
}
```

Key points:
- **Math fidelity** is always `HiFi4` (highest accuracy)
- **math_approx_mode** is always `false` because `get_op_approx_mode` returns `false` for all ops by default (the switch statement has only a `default: return false` case)
- **fp32_dest_acc_en** is passed through from operation attributes, controlling whether DEST registers operate in FP32 mode

### Runtime Arguments

| Kernel | Arg 0 | Arg 1 | Arg 2 |
|---|---|---|---|
| Reader | `src_buffer->address()` | `num_pages_per_core` | `start_tile_id` |
| Writer | `dst_buffer->address()` | `num_pages_per_core` | `start_tile_id` |
| Compute | `packed_scalar1 = 0` | `packed_scalar2 = 0` | -- |

SILU does not use scalar runtime arguments (both are zero). The scalar packing switch statement does not have a `SILU` case.

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);   // DRAM address of the input tensor buffer
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of tiles this core must read
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Global tile index offset for this core

    constexpr auto src_args = TensorAccessorArgs<0>();      // Compile-time tensor accessor config (interleaved layout info)

    constexpr uint32_t cb_id_in0 = 0;                      // CB index 0 = input circular buffer

    // Page size is determined by the CB interface, which was configured by the host
    // For TILE layout this equals tile_size(data_format), for ROW_MAJOR it equals buffer page_size
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    constexpr uint32_t onepage = 1;                         // Process one page (tile) at a time

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);  // Construct accessor with DRAM base address

// Read tiles one by one from DRAM into the input CB
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {         // Forward iteration is the default path for SILU
#endif
        cb_reserve_back(cb_id_in0, onepage);                // Block until CB has space for one tile
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0); // Get the L1 write pointer for the CB slot
        noc_async_read_page(i, s, l1_write_addr);          // Issue async NoC read of tile i from DRAM to L1
        noc_async_read_barrier();                           // Wait for the read to complete
        cb_push_back(cb_id_in0, onepage);                   // Signal to compute kernel: one tile is available
    }
}
```

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);     // DRAM address of the output tensor buffer
    const uint32_t num_pages = get_arg_val<uint32_t>(1);     // Number of tiles this core must write
    const uint32_t start_id = get_arg_val<uint32_t>(2);      // Global tile index offset for this core

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // Output CB index (c_2 = 2)
    constexpr auto dst_args = TensorAccessorArgs<1>();            // Compile-time tensor accessor config for output

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_pages);    // Sharded path: wait for all pages, no DRAM write needed
#else

    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {          // Forward iteration for standard SILU
#endif
        cb_wait_front(cb_id_out, onepage);                   // Block until compute kernel has produced a tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);    // Get L1 address of the completed tile
        noc_async_write_page(i, s, l1_read_addr);           // Issue async NoC write of tile i from L1 to DRAM
        noc_async_writes_flushed();                          // Ensure write is flushed (not necessarily completed)
        cb_pop_front(cb_id_out, onepage);                    // Free the CB slot for reuse by compute
    }
    noc_async_write_barrier();                               // Final barrier: all writes must complete before exit
#endif
}
```

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

This is the generic SFPU compute kernel shared by most unary SFPU operations. The operation-specific behavior is injected via preprocessor macros (`SFPU_OP_INIT_0`, `SFPU_OP_FUNC_0`, etc.) that are defined by the host program factory through the `unary_defines` map.

For SILU, `SFPU_OP_CHAIN_0` expands to: `silu_tile_init(); silu_tile(0);`
(Actually, `SFPU_OP_CHAIN_0` expands to the concatenation of `SFPU_OP_INIT_0` and `SFPU_OP_FUNC_0` as generated by `get_block_defines`.)

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"                              // Common compute API (tile_regs_acquire, etc.)
#include "api/compute/tile_move_copy.h"                      // copy_tile API
#include "api/compute/eltwise_unary/eltwise_unary.h"         // init_sfpu, pack_tile APIs
#include "api/compute/eltwise_unary/sfpu_split_includes.h"   // Conditional includes based on SFPU_OP_*_INCLUDE defines
                                                              // For SILU: SFPU_OP_COMPUTE_KERNEL_API_INCLUDE=1
                                                              //   -> includes compute_kernel_api.h
                                                              //   -> provides silu_tile_init() and silu_tile()
#include "api/compute/eltwise_unary/trigonometry.h"           // Trig functions (not used by SILU but always included)
#include "api/compute/mul_int_sfpu.h"                         // Integer multiply SFPU (not used by SILU)
#include "api/compute/eltwise_unary/rpow.h"                   // Reverse power (not used by SILU)
#include "api/compute/eltwise_unary/rdiv.h"                   // Reverse division (not used by SILU)
#include "api/compute/eltwise_unary/fill.h"                   // Fill operation (not used by SILU)

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tile blocks this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for SILU)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);             // Initialize SFPU: configure unpack from CB0, pack to CB2

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim); // Reserve output CB space for one block of tiles

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();                                // Acquire exclusive access to DEST registers

            cb_wait_front(tt::CBIndex::c_0, 1);                // Wait for reader to provide one input tile

            copy_tile(tt::CBIndex::c_0, 0, 0);                 // Unpack tile from CB0 slot 0 into DEST register 0
                                                                // This moves data from L1 (CB) to DEST via the unpacker

// The SFPU_OP_CHAIN_0 macro expands to the init + func calls for the configured operation.
// For SILU, this expands to:
//   silu_tile_init();   -- Initialize SFPU for SiLU (sets up reciprocal LUT)
//   silu_tile(0);       -- Apply SiLU to tile in DEST register 0
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();                                 // Signal that DEST registers are ready for packing

            tile_regs_wait();                                   // Wait for packer to be ready

            pack_tile(0, tt::CBIndex::c_2);                     // Pack tile from DEST register 0 into output CB2

            cb_pop_front(tt::CBIndex::c_0, 1);                  // Free the input tile slot in CB0

            tile_regs_release();                                // Release DEST registers
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);    // Signal to writer: block of tiles is ready
    }
}
```

### SFPU Kernel Implementation

This section provides a deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### Compute API Layer

**File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`

The compute API provides the user-facing `silu_tile` and `silu_tile_init` functions:

```cpp
ALWI void silu_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE>(idst)));
}

ALWI void silu_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_silu_init<APPROX>()));
}
```

- `APPROX` is the global approximation mode template parameter (always `false` for SILU since `get_op_approx_mode` returns `false`)
- `DST_ACCUM_MODE` is the FP32 destination accumulation mode (controlled by `fp32_dest_acc_en`)
- `MATH(...)` ensures the code only runs on the math RISC-V processor

#### LLK API Layer

**File (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_silu.h`
**File (Wormhole)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_silu.h`

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"    // Generic SFPU init infrastructure
#include "llk_math_eltwise_unary_sfpu_params.h"   // Generic SFPU per-tile dispatch (iterates over tile faces)
#include "ckernel_sfpu_silu.h"                     // The actual SFPU microcode for SiLU

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_silu_init() {
    // Initialize the SFPU for SiLU. The init function is passed as a callback
    // to the generic SFPU init infrastructure, which handles register setup,
    // address modifier configuration, and counter resets.
    llk_math_eltwise_unary_sfpu_init<SfpuType::silu, APPROXIMATE>(sfpu::silu_init<APPROXIMATE>);
}

// Blackhole variant has template parameter ITERATIONS defaulted to 8
template <bool APPROXIMATE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_silu(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    // Dispatch the SiLU calculation across all faces of the tile at dst_index.
    // _llk_math_eltwise_unary_sfpu_params_ handles:
    //   1. Computing the DEST register offset for the given dst_index
    //   2. Iterating over the 4 faces (16x16 sub-tiles) of the 32x32 tile
    //   3. Calling calculate_silu for each face with ITERATIONS=8 (one row of 16 elements per iteration)
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_silu<is_fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

Note: The Wormhole variant is nearly identical but hardcodes `ITERATIONS` to 8 in the template call rather than using a defaulted template parameter:
```cpp
// Wormhole variant
template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_silu(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_silu<is_fp32_dest_acc_en, 8>, dst_index, vector_mode);
}
```

#### SFPU Kernel File (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_sigmoid.h"    // Provides _sfpu_sigmoid_ helper used by SiLU

namespace ckernel::sfpu {

// calculate_silu: Core SFPU microcode for SiLU activation
// Template parameters:
//   is_fp32_dest_acc_en: Whether DEST registers are in FP32 mode (affects sigmoid precision path)
//   ITERATIONS: Number of elements to process per invocation (default 8, one per row in a 16x16 face)
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
#pragma GCC unroll 8                     // Hint to unroll the loop for performance (avoid branch overhead)
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];   // Load the current element from DEST register into SFPU vFloat register
                                              // dst_reg[0] refers to the current row being processed

        // SiLU(x) = x * sigmoid(x)
        // _sfpu_sigmoid_ computes sigmoid(x) = 1 / (1 + exp(-x))
        // The template parameter controls precision:
        //   - FP32 mode: uses _sfpu_exp_improved_ and _sfpu_reciprocal_<2> (2 Newton-Raphson iterations)
        //   - BF16 mode: uses _sfpu_exp_21f_ and _sfpu_reciprocal_<1> (1 Newton-Raphson iteration)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);

        // If not in FP32 accumulation mode, round the result back to bfloat16
        // float_to_fp16b truncates/rounds the FP32 intermediate to BF16 precision
        // reinterpret casts the result back to vFloat without changing bits
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;           // Store the result back to the current DEST register row
        sfpi::dst_reg++;                     // Advance to the next row in the DEST register
    }
}

// silu_init: Initialize SFPU state for SiLU computation
// On Blackhole: delegates to sigmoid_init<false> which calls _init_sfpu_reciprocal_<false>()
// This sets up the reciprocal lookup table needed by the Newton-Raphson reciprocal approximation
template <bool APPROXIMATION_MODE>
inline void silu_init() {
    // SiLU always uses the non-approximate sigmoid path (the _sfpu_sigmoid_ function),
    // so initialization must match: sigmoid_init<false> sets up for exact sigmoid
    sigmoid_init<false>();
}

}  // namespace ckernel::sfpu
```

#### SFPU Kernel File (Wormhole B0)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_sigmoid.h"

namespace ckernel::sfpu {

// calculate_silu is identical between Blackhole and Wormhole
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// silu_init on Wormhole: delegates to _init_sfpu_reciprocal_ (or approximate variant)
// This differs from Blackhole which uses sigmoid_init<false>()
template <bool APPROXIMATION_MODE>
inline void silu_init() {
    if constexpr (!APPROXIMATION_MODE) {
        _init_sfpu_reciprocal_<false>();     // Non-approximate: initialize reciprocal LUT for Newton-Raphson
    } else {
        _init_sfpu_reciprocal_<true>();      // Approximate: initialize with approximate reciprocal
    }
}

}  // namespace ckernel::sfpu
```

#### Sigmoid Helper (Shared by Both Architectures)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h`
(Wormhole version is identical in structure)

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_sigmoid_appx.h"    // Approximate sigmoid (LUT-based, not used by SiLU)
#include "ckernel_sfpu_recip.h"            // Reciprocal helpers

namespace ckernel {
namespace sfpu {

// _sfpu_sigmoid_: Computes sigmoid(x) = 1 / (1 + exp(-x))
// This is the core function that SiLU depends on.
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    sfpi::vFloat exp_neg_x;

    // Step 1: Compute exp(-x)
    // FP32 mode uses a higher-accuracy exp implementation (_sfpu_exp_improved_)
    // BF16 mode uses a faster but less accurate exp (_sfpu_exp_21f_, ~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_improved_<true>(-x);       // Negate x, then compute exp
    } else {
        exp_neg_x = _sfpu_exp_21f_<true>(-x);            // Fast exp for BF16 precision
    }

    // Step 2: Compute denominator = 1 + exp(-x)
    sfpi::vFloat denominator = sfpi::vConst1 + exp_neg_x;  // vConst1 is the SFPI constant 1.0f

    // Step 3: Compute reciprocal of denominator = 1 / (1 + exp(-x))
    // FP32 mode uses 2 Newton-Raphson iterations for higher precision
    // BF16 mode uses 1 Newton-Raphson iteration (sufficient for BF16 accuracy)
    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = _sfpu_reciprocal_<2>(denominator);       // 2 Newton-Raphson iterations
    } else {
        result = _sfpu_reciprocal_<1>(denominator);       // 1 Newton-Raphson iteration
    }

    return result;
}

// Sigmoid init: sets up reciprocal LUT
template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    if constexpr (!APPROXIMATION_MODE) {
        _init_sfpu_reciprocal_<false>();    // Blackhole version
        // Wormhole uses: _init_reciprocal_<false, false>();
    } else {
        sigmoid_appx_init();                // LUT-based approximate init (not used by SiLU)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|---|---|
| `sfpi::dst_reg[0]` (SFPLOAD) | Load a vector element from the DEST register file into an SFPU local register |
| `sfpi::dst_reg[0] = result` (SFPSTORE) | Store a vector result from an SFPU local register back to the DEST register file |
| `sfpi::dst_reg++` (SFPINCRWC) | Increment the DEST register write cursor to advance to the next row |
| `operator*` (SFPMUL) | Floating-point vector multiply -- used for `x * sigmoid(x)` |
| `operator+` (SFPADD) | Floating-point vector add -- used for `1 + exp(-x)` in sigmoid |
| `_sfpu_exp_improved_` / `_sfpu_exp_21f_` | Compute exp(-x) using SFPU math library functions. These are composite operations that use SFPMUL, SFPADD, SFPLUT, and other instructions internally |
| `_sfpu_reciprocal_<N>` | Compute 1/x using an initial estimate plus N Newton-Raphson iterations. Uses SFPMUL and SFPADD |
| `sfpi::float_to_fp16b` | Convert FP32 to BF16 format (truncation/rounding). Maps to SFPLUT or SFPCONV |
| `sfpi::reinterpret<vFloat>` | Bitwise reinterpretation cast (no instruction emitted, just type change) |
| `sfpi::vConst1` | SFPU constant register holding 1.0f |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `dst_reg[0]` | DEST register -- input value is loaded from here, result is stored back here. Auto-incremented after each iteration. |
| SFPU local registers (implicit) | `x`, `exp_neg_x`, `denominator`, `result` are allocated to SFPU vector local registers (lreg0-lreg3) by the compiler |
| `vConst1` | Built-in constant register = 1.0f, used in `1 + exp(-x)` |

The SFPU has 4 local vector registers (lreg0-lreg3). The SiLU computation requires:
- lreg for `x` (must be preserved for the final multiply)
- lreg for `exp(-x)`
- lreg for `denominator`
- lreg for the reciprocal result

This is a tight fit within the 4-register budget, which is why the SFPI compiler must carefully schedule register usage.

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` to gain exclusive access to DEST registers, then `copy_tile(CB0, 0, 0)` unpacks one tile from CB0 into DEST register 0. This moves data from L1 SRAM through the unpacker into the 32x32 DEST register.

2. **SFPU init**: `silu_tile_init()` is called once (before the tile loop, via the macro). It initializes the reciprocal lookup table needed by the Newton-Raphson reciprocal used inside sigmoid. On Blackhole this calls `sigmoid_init<false>()` which calls `_init_sfpu_reciprocal_<false>()`. On Wormhole it directly calls `_init_sfpu_reciprocal_<false>()`.

3. **Per-tile SFPU dispatch**: `silu_tile(0)` calls `llk_math_eltwise_unary_sfpu_silu` which calls `_llk_math_eltwise_unary_sfpu_params_`. This function:
   - Computes the DEST register base address for tile index 0
   - Iterates over the 4 faces of the 32x32 tile (each face is 16x16)
   - For each face, calls `calculate_silu<is_fp32_dest_acc_en, 8>()`

4. **Per-face SFPU computation** (`calculate_silu`): For each of the 8 iterations (8 rows of 16 elements per face):
   a. Load `x` from `dst_reg[0]` (current row of 16 elements)
   b. Compute `sigmoid(x)`:
      - Negate x to get -x
      - Compute `exp(-x)` via `_sfpu_exp_improved_` (FP32) or `_sfpu_exp_21f_` (BF16)
      - Add 1.0 to get `1 + exp(-x)`
      - Compute reciprocal via Newton-Raphson: `1 / (1 + exp(-x))`
   c. Multiply: `x * sigmoid(x)` using SFPMUL
   d. If BF16 mode: round result to BF16 via `float_to_fp16b`
   e. Store result back to `dst_reg[0]`
   f. Increment `dst_reg` to next row

5. **Pack and write-back**: After SFPU completes, `tile_regs_commit()` signals the packer. `pack_tile(0, CB2)` packs the result from DEST into the output circular buffer CB2. `cb_pop_front(CB0, 1)` frees the input slot, and `cb_push_back(CB2, 1)` signals the writer kernel.

#### SFPU Configuration

| Configuration | Value | Notes |
|---|---|---|
| `APPROXIMATION_MODE` | `false` | SiLU always uses exact sigmoid, never LUT approximation |
| `is_fp32_dest_acc_en` | Depends on operation attributes | Controls FP32 vs BF16 precision path |
| `ITERATIONS` | 8 | 8 rows per face (16x16 face / 16 elements per row = 16, but SFPU processes 2 rows per SIMD lane group, yielding 8 iterations) |
| `math_fidelity` | `HiFi4` | Highest fidelity, but does not directly affect SFPU (SFPU is always full precision) |
| `math_approx_mode` | `false` | Mapped to the `APPROX` compile-time constant in compute kernels |
| Reciprocal Newton-Raphson iterations | 2 (FP32) or 1 (BF16) | Controlled by `is_fp32_dest_acc_en` in `_sfpu_sigmoid_` |

#### Hardware Compatibility Notes

**Blackhole vs Wormhole differences for SILU:**

1. **`silu_init()` implementation**:
   - **Blackhole**: Calls `sigmoid_init<false>()` which in turn calls `_init_sfpu_reciprocal_<false>()`
   - **Wormhole**: Directly calls `_init_sfpu_reciprocal_<false>()` (or `<true>` if approximation mode)
   - Functionally equivalent for the non-approximate case, but the Wormhole variant has an explicit `APPROXIMATION_MODE` branch while Blackhole always delegates to `sigmoid_init<false>()` regardless of the template parameter

2. **`sigmoid_init()` internals**:
   - **Blackhole**: Uses `_init_sfpu_reciprocal_<false>()`
   - **Wormhole**: Uses `_init_reciprocal_<false, false>()` (different function name, same purpose)
   - This reflects a naming convention difference in the reciprocal init between architectures

3. **`llk_math_eltwise_unary_sfpu_silu` template**:
   - **Blackhole**: Has a defaulted `ITERATIONS = 8` template parameter
   - **Wormhole**: Hardcodes `8` in the template argument to `calculate_silu`
   - Functionally identical

4. **`calculate_silu` function**: Identical source code on both architectures. The same SFPI code generates correct instructions for both because SFPI is a cross-architecture abstraction layer.

5. **Underlying `_sfpu_exp_improved_` / `_sfpu_exp_21f_` / `_sfpu_reciprocal_`**: These are SFPI library functions defined in the architecture-specific SFPI submodule. While the C++ source may be shared, the compiled instructions may differ due to hardware differences in the SFPU ALU between Wormhole and Blackhole.

## Data Flow Summary

```
DRAM --> [NoC Read] --> L1 CB0 --> [Unpacker] --> DEST Registers
                                                      |
                                                  [SFPU: SiLU]
                                                  x * sigmoid(x)
                                                      |
                                                  DEST Registers --> [Packer] --> L1 CB2 --> [NoC Write] --> DRAM
```

**Per-tile pipeline**:
```
Reader:  cb_reserve_back(CB0) -> noc_async_read -> cb_push_back(CB0)
                                                        |
Compute: cb_wait_front(CB0) -> copy_tile -> SFPU_OP -> pack_tile -> cb_pop_front(CB0) -> cb_push_back(CB2)
                                                                                              |
Writer:  cb_wait_front(CB2) -> noc_async_write -> cb_pop_front(CB2)
```

## External Knowledge Sources

### DeepWiki References

- **tenstorrent/tt-metal**: Confirmed program factory structure, SILU registration path, compute kernel selection via `get_compute_kernel_path` (SILU falls to default `eltwise_sfpu.cpp`), and the `get_op_init_and_func_default` dispatch for `silu_tile_init()`/`silu_tile()`.
- **tenstorrent/tt-llk**: Confirmed `calculate_silu` implementation in `ckernel::sfpu` namespace, the dependency on `_sfpu_sigmoid_`, and architecture-specific file locations (`tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_silu.h`, `tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_silu.h`).
- **tenstorrent/sfpi**: Confirmed SFPI instructions used (SFPLOAD, SFPSTORE, SFPMUL, SFPADD), the LUT-based sigmoid approximation path (not used by SILU), and the `float_to_fp16b` conversion function.

### Confluence References

Not consulted for this analysis. The DeepWiki sources provided sufficient SFPU instruction detail for documenting the SiLU kernel.

### Glean References

Not consulted for this analysis. The open-source ckernel headers contained all necessary implementation details.

## File Index

| File | Role |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | TTNN API registration (`REGISTER_UNARY_OPERATION(silu, SILU)`) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | `UnaryOpType::SILU` enum definition |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Maps SILU to `silu_tile_init()`/`silu_tile()`, kernel path, defines |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp` | Program factory struct declarations |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Host-side program construction (CB setup, kernel creation, work distribution) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader kernel (DRAM -> L1) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer kernel (L1 -> DRAM) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Generic SFPU compute kernel (macro-driven) |
| `tt_metal/hw/inc/api/compute/compute_kernel_api.h` | `silu_tile()` and `silu_tile_init()` API |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Conditional include routing |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_silu.h` | Blackhole LLK dispatch |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_silu.h` | Wormhole LLK dispatch |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h` | Blackhole SFPU microcode |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h` | Wormhole SFPU microcode |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h` | Blackhole sigmoid helper |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h` | Wormhole sigmoid helper |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h` | Reciprocal SFPU wrapper |
