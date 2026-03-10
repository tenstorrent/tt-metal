# TTNN Operation Analysis: ABS (SFPU)

## Overview

| Property | Value |
|---|---|
| **Operation Name** | ABS |
| **Operation Type** | Unary Eltwise (SFPU) |
| **Program Factory** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` |
| **Namespace** | `ttnn::prim` |
| **Compute Kernel** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| **SFPU Kernel** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h` |
| **Reader Kernel** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` |
| **Writer Kernel** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| **Supported Data Types** | BFLOAT16, BFLOAT8_B, FLOAT32, INT32 |
| **Layout** | TILE (32x32), ROW_MAJOR |
| **SFPU Instruction** | `SFPABS` (opcode 0x7D) |

## High-Level Entry Point

The ABS operation is exposed through the `ttnn::Abs` struct in `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`. It has two overloads:

1. **Tensor overload**: For standard tensors. Dispatches to `UnaryOpType::ABS` for floating-point types, or `UnaryOpType::ABS_INT32` for `DataType::INT32`.
2. **ComplexTensor overload**: For complex tensors. Computes `hypot(real, imag)` -- i.e., the magnitude of the complex number.

```cpp
// From unary.cpp
Tensor Abs::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::ABS;
    if (input_tensor.dtype() == DataType::INT32) {
        op_type = UnaryOpType::ABS_INT32;
    }
    return detail::unary_impl(input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}
```

The `detail::unary_impl` function routes to `UnaryProgramFactory::create` (or `UnarySubCoreGridProgramFactory::create` if sub-core grids are specified).

## Program Factory Analysis

### Factory Variants

The program factory file contains two factory structs:

1. **`UnaryProgramFactory`** -- Standard factory using the full compute grid. Splits work across all available cores using `split_work_to_cores`, producing two core groups that may have different tile counts.
2. **`UnarySubCoreGridProgramFactory`** -- Variant for sub-core grids. Uses a caller-specified subset of cores, requires uniform tile distribution (tiles must be evenly divisible by core count).

Both factories share the same kernel files and identical SFPU dispatch logic. The analysis below focuses on `UnaryProgramFactory` since it is the common path.

### Circular Buffer Configuration

| CB Index | Identifier | Purpose | Page Count | Data Format |
|---|---|---|---|---|
| `c_0` | `src0_cb_index` | Input tile buffer | 2 | Input tensor format (or output format for BITCAST) |
| `c_1` | `tmp0_cb_index` | Temporary buffer (only for HARDSHRINK/LOGIT) | 2 | Input tensor format |
| `c_2` | `output_cb_index` | Output tile buffer | 2 | Output tensor format |

For ABS, only `c_0` (input) and `c_2` (output) are used. The temporary buffer `c_1` is not allocated because ABS is neither HARDSHRINK nor LOGIT.

The double-buffering (2 pages per CB) enables overlap between data movement and compute -- the reader can fill one page while compute processes the other.

### Work Distribution

```cpp
auto [num_cores, all_cores, core_group_1, core_group_2,
      num_pages_per_core_group_1, num_pages_per_core_group_2] =
    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);
```

The total number of pages (tiles for TILE layout, rows for ROW_MAJOR) is distributed across the available compute grid. If the division is uneven, `core_group_1` gets `ceil(num_pages / num_cores)` pages and `core_group_2` gets the remainder. Each core group gets its own compute kernel instance with the appropriate `per_core_block_cnt`.

### Compile-Time Defines

For ABS, the following defines are injected into the compute kernel:

| Define | Value | Purpose |
|---|---|---|
| `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` | `1` | Includes `compute_kernel_api.h` which provides `abs_tile` and `abs_tile_init` |
| `SFPU_OP_INIT_0` | `abs_tile_init();` | Initialization macro for the SFPU operation |
| `SFPU_OP_FUNC_0` | `abs_tile(0);` | Per-tile SFPU function call |
| `SFPU_OP_CHAIN_0` | Concatenation of init + func | The full operation chain executed per tile |
| `INP_FLOAT` / `INP_FLOAT32` / `INP_INT32` / `INP_UINT32` | `1` | Data type indicator (exactly one is set) |

For `UnaryOpType::ABS_INT32`, the func define becomes `abs_tile_int32(0);` instead.

ABS falls through to the `default` case in `get_macro_definition`, so it uses `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`. This is because `abs_tile` and `abs_tile_init` are declared directly in `compute_kernel_api.h` rather than in a separate split-include header.

### Compute Configuration

```cpp
tt::tt_metal::ComputeConfig{
    .math_fidelity = MathFidelity::HiFi4,
    .fp32_dest_acc_en = args.fp32_dest_acc_en,
    .unpack_to_dest_mode = unpack_to_dest_mode,
    .bfp8_pack_precise = args.bfp8_pack_precise,
    .math_approx_mode = false,  // get_op_approx_mode returns false for ABS
    .compile_args = {num_pages_per_core_group, 1},
    .defines = unary_defines
}
```

- **Math fidelity**: Always `HiFi4` (highest precision). ABS does not involve approximations, so this is effectively a no-op for the abs computation itself, but it configures the math pipeline.
- **Approximation mode**: Always `false` for ABS (`get_op_approx_mode` returns false for all ops by default).
- **FP32 dest accumulation**: Controlled by the caller. When enabled, the destination register file operates in FP32 mode for higher precision.

### Runtime Arguments

| Kernel | Arg 0 | Arg 1 | Arg 2 |
|---|---|---|---|
| Reader | `src_buffer->address()` | `num_pages_per_core` | `num_pages_written` (start page ID) |
| Writer | `dst_buffer->address()` | `num_pages_per_core` | `num_pages_written` (start page ID) |
| Compute | `packed_scalar1` (0 for ABS) | `packed_scalar2` (0 for ABS) | -- |

ABS does not use scalar parameters, so both packed scalars are 0. The compute kernel receives them but ignores them since ABS has no parametric behavior.

### Program Caching

The `override_runtime_arguments` method enables program caching. On subsequent calls with different tensors (but same shapes and configs), only the buffer addresses are updated rather than recreating the entire program. This is a significant performance optimization for repeated inference.

```cpp
void UnaryProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const UnaryParams&,
    const UnaryInputs& tensor_args, Tensor& output) {
    // Only updates src_buffer->address() and dst_buffer->address() per core
}
```

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

The reader kernel fetches pages from DRAM into the input circular buffer (`c_0`). It uses the `TensorAccessor` API for address computation, supporting both interleaved and sharded memory layouts.

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime arguments: buffer address, number of pages to read, starting page ID
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    // Compile-time tensor accessor args encode memory layout (interleaved/sharded)
    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_id_in0 = 0;  // Input circular buffer index (c_0)

    // Page size is read from the CB interface, making this kernel layout-agnostic
    // (works for both tile pages and row-major pages)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    constexpr uint32_t onepage = 1;  // Process one page at a time

    // Construct tensor accessor with the DRAM address and page size
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    // Read pages sequentially from start_id to start_id + num_pages
    // The BACKWARDS variant (ifdef) reads in reverse order for certain operations
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_reserve_back(cb_id_in0, onepage);       // Wait for space in CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);  // Get L1 write address
        noc_async_read_page(i, s, l1_write_addr);  // Issue async NoC read from DRAM
        noc_async_read_barrier();                   // Wait for read to complete
        cb_push_back(cb_id_in0, onepage);           // Signal page is ready for consumer
    }
}
```

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

The writer kernel drains the output circular buffer (`c_2`) and writes pages to DRAM.

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime arguments: buffer address, number of pages to write, starting page ID
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    // Compile-time args: output CB index and tensor accessor config
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // c_2 for ABS
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    // For sharded output, just wait for all pages to be produced -- no NoC writes needed
    // because the output is already in L1 at the correct location
    cb_wait_front(cb_id_out, num_pages);
#else
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

    // Write pages sequentially
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_wait_front(cb_id_out, onepage);          // Wait for compute to produce a page
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);    // Get L1 read address
        noc_async_write_page(i, s, l1_read_addr);   // Issue async NoC write to DRAM
        noc_async_writes_flushed();                  // Ensure write is dispatched
        cb_pop_front(cb_id_out, onepage);            // Free CB space for compute
    }
    noc_async_write_barrier();  // Final barrier to ensure all writes complete
#endif
}
```

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

This is the generic eltwise SFPU compute kernel used by many unary operations. The actual operation is injected via the `SFPU_OP_CHAIN_0` preprocessor macro.

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
// sfpu_split_includes.h conditionally includes headers based on SFPU_OP_*_INCLUDE defines.
// For ABS, SFPU_OP_COMPUTE_KERNEL_API_INCLUDE=1 causes compute_kernel_api.h to be included,
// which provides abs_tile_init() and abs_tile().
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    // Compile-time args set by the program factory:
    // per_core_block_cnt = number of tiles this core processes
    // per_core_block_dim = tiles per block (always 1 for standard unary ops)
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    // Initialize the SFPU pipeline: configures unpack (A2D datacopy), pack,
    // and math hardware for the unary eltwise operation.
    // c_0 = input CB, c_2 = output CB
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    // Outer loop: iterate over all tiles assigned to this core
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve output space for the entire block (1 tile)
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to the DST register file.
            // This blocks until the packer releases the registers.
            tile_regs_acquire();

            // Wait for the reader to produce one input tile in c_0
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Copy tile from input CB (c_0) to DST register 0.
            // This triggers the unpacker to read the tile from L1 into source registers,
            // then the math engine performs an A2D (datacopy) to move it into DST[0].
            copy_tile(tt::CBIndex::c_0, 0, 0);

            // SFPU_OP_CHAIN_0 expands to:
            //   abs_tile_init(); abs_tile(0);
            // This initializes the SFPU for abs and then executes abs on DST[0] in-place.
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Signal that DST registers are written and ready for packing.
            tile_regs_commit();

            // Wait for the packer to acknowledge it can read DST.
            tile_regs_wait();

            // Pack tile from DST[0] into the output CB (c_2).
            pack_tile(0, tt::CBIndex::c_2);

            // Free the input tile from c_0 so the reader can reuse the space.
            cb_pop_front(tt::CBIndex::c_0, 1);

            // Release DST registers so the next iteration can acquire them.
            tile_regs_release();
        }
        // Push the completed output block to the writer
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File

- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h`

Both architectures share identical floating-point ABS logic. The INT32 path differs only in the address mode encoding used for SFPLOAD/SFPSTORE.

#### LLK Dispatch Layer

**File**: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_abs.h`

```cpp
// This file bridges the compute API (abs_tile) to the SFPU microcode (calculate_abs).

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_abs.h"

namespace ckernel {

// Called once by abs_tile_init(). Configures SFPU registers, address modifiers,
// and resets math counters for the abs operation type.
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::abs, APPROXIMATE>();
}

// Called per-tile by abs_tile(idst). Sets up the DST register pointer,
// invokes calculate_abs across all 8 face iterations, then finalizes.
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_abs<APPROXIMATE>, dst_index, vector_mode);
}

// INT32 variant -- uses calculate_abs_int32 which issues raw TTI instructions.
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs_int32(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_abs_int32<APPROXIMATE>, dst_index, vector_mode);
}

}  // namespace ckernel
```

#### Annotated SFPU Kernel Source (Floating-Point Path)

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// calculate_abs: Floating-point absolute value via SFPI
//
// APPROXIMATION_MODE is unused for abs (the operation is exact, not an approximation).
// ITERATIONS defaults to 8 because a 32x32 tile has 1024 elements, and the SFPU
// processes 32 elements per lane (one row) per iteration. The tile is organized as
// 4 faces of 16x16, and each face has 16 rows, but the SFPU processes 2 faces
// simultaneously (rows from face 0 and face 2, then face 1 and face 3). This gives
// 8 iterations to cover all 4 faces: 4 faces * 16 rows / (2 faces * 4 rows_per_iter) = 8.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() {
    // SFPU microcode -- executed on the SFPU vector engine
    for (int d = 0; d < ITERATIONS; d++) {
        // Load one row (32 elements) from DST register file into SFPU lane register LReg[0].
        // dst_reg[0] is an SFPI abstraction over the destination register file.
        vFloat v = dst_reg[0];

        // Apply sfpi::abs() which compiles to the SFPABS instruction with MOD1_FLOAT.
        // For FP32: clears the sign bit (bit 31) to produce the absolute value.
        // Special cases: NaN is preserved as-is (including sign), +/-Inf becomes +Inf.
        dst_reg[0] = sfpi::abs(v);

        // Advance the DST register pointer to the next row.
        // This is not a memory increment -- it advances the hardware row counter
        // that determines which row of the tile the SFPU reads/writes.
        dst_reg++;
    }
}

// calculate_abs_int32: Integer absolute value via raw TTI (Tenstorrent Instruction) macros
//
// This path is used when the input tensor has DataType::INT32. It uses direct
// instruction emission rather than SFPI abstractions because integer operations
// require explicit format selection in SFPLOAD/SFPSTORE.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // SFPU microcode for INT32 abs
    for (int d = 0; d < ITERATIONS; d++) {
        // SFPLOAD: Load a row from DST into LReg[1] with INT32/SMAG32 format.
        //   arg0=1: destination LReg index (LReg[1])
        //   arg1=4 (WH) or 12 (BH): InstrMod for SMAG32 format load
        //   arg2=3 (WH: literal 3) or ADDR_MOD_7 (BH): address modifier
        //   arg3=0: address offset
        // The format mode (4 on WH = SMAG32) tells the load unit to interpret
        // the DST register data as sign-magnitude 32-bit integers.
        TT_SFPLOAD(1, 4, 3, 0);  // Wormhole encoding shown; Blackhole uses (1, 12, ADDR_MOD_7, 0)

        // SFPABS: Compute absolute value of LReg[1], store result in LReg[0].
        //   arg0=0: destination LReg (LReg[0])
        //   arg1=1: source LReg (LReg[1])
        //   arg2=0: InstrMod[0]=0 means INT32 format (two's complement abs)
        //   arg3=0: reserved
        // For INT32: if value < 0, negate it. Special case: MIN_INT (-2147483648)
        // overflows to MAX_INT (2147483647) and sets the Overflow flag.
        TTI_SFPABS(0, 1, 0, 0);

        // SFPSTORE: Store LReg[0] back to DST with INT32/SMAG32 format.
        //   arg0=0: source LReg (LReg[0])
        //   arg1=4 (WH) or 12 (BH): InstrMod for SMAG32 format store
        //   arg2=3 (WH) or ADDR_MOD_7 (BH): address modifier
        //   arg3=0: address offset
        TTI_SFPSTORE(0, 4, 3, 0);  // Wormhole encoding shown

        // Advance to next row
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Opcode | Used In | Description |
|---|---|---|---|
| `SFPABS` | 0x7D | Both FP32 and INT32 paths | Computes absolute value. Format selected by `InstrMod[0]`: 1=FP32 (clear sign bit), 0=INT32 (two's complement negation if negative). IPC=1, Latency=1. |
| `SFPLOAD` | 0x70 | INT32 path only | Loads a value from the DST register file into an SFPU lane register (LReg). Applies format conversion (e.g., SMAG32). |
| `SFPSTORE` | -- | INT32 path only | Stores a value from an SFPU lane register back to the DST register file. |

For the FP32 path, `sfpi::abs(v)` compiles to a single `SFPABS` instruction with `InstrMod[0]=1`. The load and store are handled implicitly by the SFPI compiler through `dst_reg[0]` read/write syntax, which emits `SFPLOAD` (with IMPLIED format, auto-detecting FP32) and `SFPSTORE` respectively.

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `dst_reg[0]` (DST register file) | Source and destination for the tile data. Each access reads/writes one row (32 elements). |
| `LReg[0]` (Lane Register 0) | FP32 path: implicitly used by SFPI `dst_reg[0]` access. INT32 path: destination of SFPABS, source of SFPSTORE. |
| `LReg[1]` (Lane Register 1) | INT32 path: loaded from DST via SFPLOAD, source operand for SFPABS. |
| `vFloat v` | SFPI virtual register mapped to an LReg by the compiler. Holds the loaded value before abs is applied. |

The SFPU has 4 lane registers (LReg[0]-LReg[3]), each holding 32 elements (one per SIMD lane). For the simple FP32 abs path, the compiler manages register allocation. For the INT32 path, explicit LReg indices are used via TTI macros.

#### SFPU Execution Flow

1. **Initialization** (`abs_tile_init` -> `llk_math_eltwise_unary_sfpu_abs_init`):
   - Configures SFPU control registers via `_init_sfpu_config_reg()`
   - Sets up address modifiers for the abs operation type (`SfpuType::abs`)
   - Resets math counters

2. **Per-tile execution** (`abs_tile(0)` -> `llk_math_eltwise_unary_sfpu_abs`):
   - `_llk_math_eltwise_unary_sfpu_start_(dst_index)`: Sets the DST write address and waits for SFPU readiness
   - `calculate_abs()`: Loops 8 times over tile rows:
     - **Load**: Read 32-element row from DST into LReg via `dst_reg[0]` (implicit SFPLOAD)
     - **Compute**: Apply `sfpi::abs()` which emits `SFPABS` with FP32 mode
     - **Store**: Write result back to DST via `dst_reg[0] = ...` (implicit SFPSTORE)
     - **Advance**: `dst_reg++` moves to next row
   - `_llk_math_eltwise_unary_sfpu_done_()`: Clears DST address and waits for SFPU completion

3. **Data flow context** (within the compute kernel's per-tile loop):
   - `tile_regs_acquire()` -- Lock DST registers for exclusive math engine use
   - `cb_wait_front(c_0, 1)` -- Wait for reader to produce input tile
   - `copy_tile(c_0, 0, 0)` -- Unpack tile from L1 (CB c_0) into DST[0] via A2D datacopy
   - `abs_tile_init(); abs_tile(0);` -- Execute SFPU abs on DST[0] in-place
   - `tile_regs_commit()` -- Signal DST is ready for packing
   - `tile_regs_wait()` -- Wait for packer acknowledgment
   - `pack_tile(0, c_2)` -- Pack DST[0] into output CB c_2
   - `cb_pop_front(c_0, 1)` -- Free input CB space
   - `tile_regs_release()` -- Release DST for next iteration

#### SFPU Configuration

- **Math fidelity**: `MathFidelity::HiFi4` (always, for all unary ops in this factory). Does not affect SFPABS since it is an exact operation, but configures the FPU/SFPU pipeline width.
- **Approximation mode**: `false`. The `APPROXIMATION_MODE` template parameter is passed through to `calculate_abs` but is unused since abs is exact.
- **FP32 dest accumulation**: Configurable via `args.fp32_dest_acc_en`. When enabled, DST registers use FP32 format, which is the native format for SFPABS.
- **Unpack-to-dest mode**: Default for standard precision; `UnpackToDestFp32` when `preserve_fp32_precision` is set.
- **BFP8 pack precise**: Configurable via `args.bfp8_pack_precise`. Affects packing precision for BFP8 output formats.

#### Hardware Compatibility Notes

The floating-point `calculate_abs` function is identical between Wormhole B0 and Blackhole. The SFPI abstraction (`sfpi::abs()`) generates the same `SFPABS` instruction on both architectures.

The INT32 `calculate_abs_int32` function differs in SFPLOAD/SFPSTORE encoding:

| Parameter | Wormhole B0 | Blackhole |
|---|---|---|
| SFPLOAD InstrMod | `4` (SMAG32) | `12` (different encoding for SMAG32) |
| SFPLOAD AddrMod | `3` (literal) | `ADDR_MOD_7` (named constant) |
| SFPSTORE InstrMod | `4` | `12` |
| SFPSTORE AddrMod | `3` | `ADDR_MOD_7` |

These differences reflect the architectural register encoding changes between Wormhole and Blackhole, but the functional behavior is identical: load INT32 from DST, compute abs, store back to DST.

The SFPABS instruction itself has the same opcode (0x7D) and semantics on both architectures:
- **FP32 mode** (`InstrMod[0]=1`): Clears sign bit. NaN preserved with original sign. +/-Inf becomes +Inf.
- **INT32 mode** (`InstrMod[0]=0`): Two's complement negation if negative. MIN_INT saturates to MAX_INT with Overflow flag.

## External Knowledge Sources

### DeepWiki References

- **tenstorrent/tt-metal**: Confirmed the dispatch path from `ttnn::Abs::invoke` through `UnaryProgramFactory` to `eltwise_sfpu.cpp`. Identified the `SFPU_OP_CHAIN_0` macro pattern and the `get_compute_kernel_path` default case.
- **tenstorrent/tt-llk**: Confirmed the LLK dispatch pattern through `_llk_math_eltwise_unary_sfpu_params_` and the `calculate_abs` / `calculate_abs_int32` SFPU microcode functions.
- **tenstorrent/tt-isa-documentation**: Provided SFPABS instruction details including opcode 0x7D, O2 encoding, FP32/INT32 mode selection via InstrMod[0], and special case handling for NaN, Inf, and MIN_INT.
- **tenstorrent/sfpi**: Confirmed that `sfpi::abs()` maps to `__builtin_rvtt_sfpabs` with `SFPABS_MOD1_FLOAT` for vFloat and `SFPABS_MOD1_INT` for vInt.

### Confluence References

- **Tensix SFPU Instruction Set Architecture** (Page ID: 1170505767): Provided authoritative SFPABS specification including:
  - Opcode 0x7D, O2 encoding format
  - IPC=1, Latency=1 (single-cycle operation)
  - Algorithmic implementation pseudocode for both FP32 and INT32 modes
  - NaN preservation behavior (NaN input produces same NaN output, unlike standard IEEE abs)
  - INT32 overflow behavior (MIN_INT maps to MAX_POS with Overflow flag)
  - SFPLOAD instruction format table (InstrMod values for SMAG32, FP32, etc.)

### Glean References

Not consulted. DeepWiki and Confluence provided sufficient detail for the ABS operation analysis.

## File Reference Index

| File | Purpose |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | `Abs` struct declaration with two invoke overloads |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp` | `Abs::invoke` implementation -- routes to ABS or ABS_INT32 |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp` | Program factory struct and cached_program_t definitions |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Program factory create/override_runtime_arguments implementations |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | `get_op_init_and_func`, `get_macro_definition`, `get_compute_kernel_path` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Generic SFPU compute kernel (shared by many unary ops) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader data movement kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer data movement kernel |
| `tt_metal/hw/inc/api/compute/compute_kernel_api.h` | `abs_tile()` and `abs_tile_init()` API functions |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Conditional include system for SFPU operations |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_abs.h` | LLK dispatch layer (Wormhole) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_abs.h` | LLK dispatch layer (Blackhole) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h` | SFPU microcode: `calculate_abs`, `calculate_abs_int32` (Wormhole) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h` | SFPU microcode: `calculate_abs`, `calculate_abs_int32` (Blackhole) |
