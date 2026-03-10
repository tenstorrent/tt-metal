# RELU Implementation Analysis

## Overview
The RELU (Rectified Linear Unit) operation computes `relu(x) = max(x, 0)` element-wise on each element of the input tensor. It is implemented as a unary SFPU operation using the shared `UnaryProgramFactory`, which dispatches to the `relu_min` SFPU kernel with a threshold of 0.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition
One work unit is **one page** (one tile for TILE layout, or one row for ROW_MAJOR layout). The total number of pages (`num_pages`) is obtained from `input.buffer()->num_pages()`. Each core processes a contiguous slice of pages, tile-by-tile (or row-by-row), in sequence.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|---|---|
| Dimension Convention | N-dimensional (flattened to pages) |
| Tensor Layout | TILE (32x32) or ROW_MAJOR |
| Memory Layout | Interleaved |
| Buffer Type | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32, INT32, UINT32 (configurable) |

### Output Tensor

| Property | Value |
|---|---|
| Dimension Convention | Same as input |
| Tensor Layout | Same as input |
| Memory Layout | Interleaved |
| Buffer Type | DRAM or L1 |
| Data Type | Same as input (may differ for some unary ops, but RELU preserves dtype) |

### Layout Transformations
No layout transformations are performed. The output tensor has the same layout and shape as the input.

## Data Flow Pattern

1. **Reader kernel** reads one page at a time from the input buffer in DRAM/L1 via NoC, writes it into CB `c_0` (input circular buffer).
2. **Compute kernel** waits for one tile in `c_0`, copies it to the DST register using `copy_tile`, applies the SFPU `relu_tile` operation in-place in DST, then packs the result into CB `c_2` (output circular buffer).
3. **Writer kernel** waits for one page in `c_2`, reads it, and writes it to the output buffer in DRAM/L1 via NoC.

Each page flows through the pipeline independently: Reader -> CB c_0 -> Compute -> CB c_2 -> Writer.

## Circular Buffer Configuration

| CB ID | Index | Purpose | Page Size | Num Pages | Total Size | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| `c_0` | 0 | Input tile buffer | `input_cb_page_size` (tile_size or buffer page_size) | 2 | 2 * page_size | Double-buffered | Reader | Compute |
| `c_2` | 2 | Output tile buffer | `output_cb_page_size` (tile_size or buffer page_size) | 2 | 2 * page_size | Double-buffered | Compute | Writer |

**Note**: CB `c_1` (index 1) is only created for HARDSHRINK or LOGIT operations, not for RELU.

## Pipeline Pattern Summary
Both input and output circular buffers have capacity = 2 pages and are consumed/produced 1 page at a time. This is a **double-buffered** configuration, allowing the reader to fill the next page while the compute kernel processes the current one (and similarly for compute/writer overlap).

## Index Calculations
Pages are indexed sequentially from 0 to `num_pages - 1`. Each core receives a `start_id` (cumulative offset of pages processed by previous cores) and a `num_pages_per_core` count. The reader and writer use `TensorAccessor` to translate page indices to physical memory addresses, handling bank interleaving automatically.

## Memory Access Patterns

### Read Pattern
Sequential page reads. The reader iterates `i` from `start_id` to `start_id + num_pages_per_core`, issuing one `noc_async_read_page` per iteration. Each read is barrier-ed before pushing to the CB (`noc_async_read_barrier()`), so reads are serialized.

### Write Pattern
Sequential page writes. The writer iterates in the same order, issuing one `noc_async_write_page` per page. Writes are flushed per page (`noc_async_writes_flushed()`) with a final barrier at the end (`noc_async_write_barrier()`).

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Full compute grid (`compute_with_storage_grid_size`) |
| Work Splitting | `split_work_to_cores(grid_size, num_pages)` |
| Core Group 1 | Cores that each process `num_pages_per_core_group_1` pages |
| Core Group 2 | Cores that each process `num_pages_per_core_group_2` pages (remainder) |
| Core Indexing | Column-major: `core = {i / num_cores_y, i % num_cores_y}` |
| Load Balancing | At most 1 page difference between group 1 and group 2 |

The two core groups exist to handle the case where `num_pages` does not divide evenly across cores. Group 2 handles the remainder (fewer pages per core). If pages divide evenly, group 2 is empty.

## Arguments

### Compile-Time Arguments

**Reader Kernel** (`reader_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0+ | TensorAccessorArgs | uint32_t[] | Encodes buffer type, page size, bank info for source tensor |

**Writer Kernel** (`writer_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_out | uint32_t | Output circular buffer index (always 2 for `c_2`) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encodes buffer type, page size, bank info for destination tensor |

**Compute Kernel** (`eltwise_sfpu.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (pages) this core processes |
| 1 | per_core_block_dim | uint32_t | Pages per block (always 1 for standard unary factory) |

### Runtime Arguments

**Reader Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages this core reads |
| 2 | start_id | uint32_t | Starting page index for this core |

**Writer Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages this core writes |
| 2 | start_id | uint32_t | Starting page index for this core |

**Compute Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | packed_scalar1 | uint32_t | Always 0 for RELU (unused threshold parameter) |
| 1 | packed_scalar2 | uint32_t | Always 0 for RELU (unused) |

## Kernel Implementations

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

| Property | Value |
|---|---|
| Type | ReaderDataMovementConfig |
| Input CBs | None (reads from DRAM/L1) |
| Output CBs | `c_0` |
| Synchronization | `cb_reserve_back` / `cb_push_back` on `c_0`; `noc_async_read_barrier` per page |

**Key Logic**: Reads pages sequentially from `start_id` to `start_id + num_pages`. Uses `TensorAccessor` for address translation. Each page is read via `noc_async_read_page`, barrier-ed, then pushed to `c_0`.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

| Property | Value |
|---|---|
| Type | WriterDataMovementConfig |
| Input CBs | `c_2` |
| Output CBs | None (writes to DRAM/L1) |
| Synchronization | `cb_wait_front` / `cb_pop_front` on `c_2`; `noc_async_write_page` per page |

**Key Logic**: Writes pages sequentially. Waits for each page in `c_2`, reads pointer, writes via `noc_async_write_page`, flushes, and pops. Final write barrier at end.

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
#include "api/compute/eltwise_unary/sfpu_split_includes.h"       // conditionally includes relu.h when SFPU_OP_RELU_FAMILY_INCLUDE is defined
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);    // total number of tiles (pages) this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);    // tiles per block; always 1 for standard unary path

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);               // initializes SFPU pipeline: configures unpacker for c_0, packer for c_2
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);   // reserve space in output CB for 1 tile (producer side)
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();                                  // acquire exclusive access to DST register file (DEST)

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);                  // wait until reader has pushed at least 1 tile into c_0

            copy_tile(tt::CBIndex::c_0, 0, 0);                   // unpack tile at position 0 in c_0 into DST register 0

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0                                       // expands to: relu_tile_init(); relu_tile(0);
                                                                  // relu_tile_init() calls llk_math_eltwise_unary_sfpu_init<SfpuType::relu_min, APPROX>()
                                                                  // relu_tile(0) dispatches _relu_min_<vFloat, false, 8, uint32_t> on DST[0] with param=0
#endif

            tile_regs_commit();                                   // signal that DST register contents are ready for packing

            tile_regs_wait();                                     // wait for pack stage to be ready to consume DST

            pack_tile(0, tt::CBIndex::c_2);                       // pack tile from DST[0] into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);                   // free the consumed tile slot in input CB c_0

            tile_regs_release();                                  // release DST register file for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);      // publish 1 tile in c_2 for writer to consume
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h` (identical for Blackhole at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h`)

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void relu_min(uint uint_threshold) {
    // For RELU, uint_threshold is always 0 (passed as the PARAM0 argument from the macro).
    // Converter::as_float reinterprets the uint32_t bit pattern as a float value.
    // When uint_threshold = 0, threshold becomes 0.0f (IEEE 754 zero).
    vFloat threshold = Converter::as_float(uint_threshold);

    // Iterate over all 8 sub-tile rows (faces) within the tile.
    // Each iteration of this loop processes one "row" of the SFPU vector register,
    // which corresponds to a 1x32 slice of data (one row within a 32x32 tile face).
    // The SFPU processes 32 elements in parallel per iteration via SIMD.
    for (int d = 0; d < 8; d++) {
        // Load the current row of elements from DST register at offset 0.
        // dst_reg[0] accesses the current write pointer position in DEST.
        vFloat a = dst_reg[0];

        // Conditional assignment: if any element is less than threshold (0.0f),
        // set it to the threshold value (0.0f).
        // v_if / v_endif are SFPU conditional execution primitives that use
        // per-lane condition codes. Elements where the condition is false are
        // left unchanged (their original positive/zero values are preserved).
        v_if(a < threshold) {
            a = threshold;                                        // clamp negative values to 0.0
        }
        v_endif;

        // Write the result back to DST register at offset 0.
        dst_reg[0] = a;

        // Advance the DST register pointer to the next row.
        // This moves the internal pointer by 32 elements (one face row).
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void relu_max(uint uint_threshold) {
    vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[0];
        v_if(a > threshold) { a = threshold; }                   // clamp values above upper limit
        v_endif;
        v_if(a < 0.0f) { a = 0.0f; }                            // clamp negative values to zero
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lrelu(const uint slope) {
    _calculate_lrelu_<APPROXIMATION_MODE>(ITERATIONS, slope);    // leaky relu delegates to LLK internal implementation
}

}  // namespace sfpu
}  // namespace ckernel
```

#### API-Level Wrapper

The compute kernel calls `relu_tile(0)` which is defined in `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h`:

```cpp
ALWI void relu_tile(uint32_t idst) {
    // RELU is implemented as relu_min with threshold=0: max(x, 0)
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, idst, 0));
}

ALWI void relu_tile_init() {
    // Initializes SFPU pipeline for relu_min operation type
    MATH(SFPU_UNARY_KERNEL_INIT(relu_min, APPROX));
}
```

The `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT` macro expands to:
```cpp
_llk_math_eltwise_unary_sfpu_params_<APPROX>(
    ckernel::sfpu::_relu_min_<sfpi::vFloat, APPROX, 8, uint32_t>,
    idst, (int)VectorMode::RC, 0);
```

The `_relu_min_` template variant with `<vFloat, APPROX, 8, uint32_t>` is a wrapper (from the tt_llk submodule) that adapts the `relu_min<APPROXIMATION_MODE>(uint)` function to the standardized SFPU dispatch interface. It ultimately calls the same `relu_min` logic shown above.

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|---|---|
| `dst_reg[0]` (read) | Loads a vector of 32 float elements from the current DEST register row |
| `dst_reg[0]` (write) | Stores a vector of 32 float elements back to the current DEST register row |
| `dst_reg++` | Advances the DEST register pointer by one row (32 elements) |
| `v_if(condition)` | Sets per-lane condition codes based on the comparison; subsequent operations only execute on lanes where the condition is true |
| `v_endif` | Ends the conditional execution block, restoring all-lanes execution |
| `vFloat` comparison (`<`) | Element-wise less-than comparison producing per-lane boolean condition codes |
| `Converter::as_float()` | Reinterprets a `uint32_t` bit pattern as an IEEE 754 float (bit-cast) |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| DEST (dst_reg) | Holds the input tile data (loaded by `copy_tile`) and the output result (consumed by `pack_tile`). 8 rows of 32 elements each are processed per tile face. |
| SFPU condition codes | Per-lane flags set by `v_if(a < threshold)` to enable conditional execution. Only lanes where the condition is true execute the clamping assignment. |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `cb_wait_front(c_0, 1)` to wait for a tile from the reader, then `tile_regs_acquire()` to lock the DST register file.
2. **Unpack to DST**: `copy_tile(c_0, 0, 0)` unpacks the tile from CB `c_0` position 0 into DST register 0. This loads all 32x32 elements of the tile into the DEST register buffer.
3. **SFPU init**: `relu_tile_init()` initializes the SFPU pipeline for the `relu_min` operation type via `llk_math_eltwise_unary_sfpu_init<SfpuType::relu_min, APPROX>()`. This sets up SFPU configuration registers.
4. **SFPU dispatch**: `relu_tile(0)` calls `_llk_math_eltwise_unary_sfpu_params_` which:
   - Sets the DST write address to tile index 0
   - Stalls until SFPU is ready (`TTI_STALLWAIT`)
   - Iterates through all 4 faces (VectorMode::RC) of the tile
   - For each face, calls `relu_min<false>(0)` which processes 8 rows of 32 elements each
   - After each face, increments the DST face address
   - Calls done to clear state and wait for SFPU idle
5. **Pack**: `tile_regs_commit()` signals DST is ready. `tile_regs_wait()` waits for packer. `pack_tile(0, c_2)` packs DST[0] into the output CB `c_2`.
6. **Cleanup**: `cb_pop_front(c_0, 1)` frees the input tile. `tile_regs_release()` releases DST.

Total elements processed per `relu_tile` call: 4 faces * 8 rows * 32 elements = 1024 elements = one 32x32 tile.

#### SFPU Configuration

| Setting | Value | Description |
|---|---|---|
| `SFPU_OP_RELU_FAMILY_INCLUDE` | `1` | Compile-time define that includes `relu.h` via `sfpu_split_includes.h` |
| `SFPU_OP_CHAIN_0_INIT_0` | `relu_tile_init();` | Init function injected by macro expansion |
| `SFPU_OP_CHAIN_0_FUNC_0` | `relu_tile(0);` | Compute function injected by macro expansion |
| `math_approx_mode` | `false` | RELU's `get_op_approx_mode` returns `false` (exact comparison) |
| `math_fidelity` | `MathFidelity::HiFi4` | Highest fidelity mode |
| `APPROXIMATION_MODE` | `false` | Template parameter propagated to SFPU kernel |
| `SfpuType` | `relu_min` | Enum value used for SFPU init configuration |
| `VectorMode` | `RC` | Process all 4 faces (rows and columns) of the tile |

#### Hardware Compatibility Notes
The `relu_min` SFPU kernel implementation is **identical** between Wormhole B0 and Blackhole architectures (`ckernel_sfpu_relu.h` is the same in both). The operation uses only basic SFPU primitives (`vFloat` load/store, comparison, conditional execution) that are available on all supported architectures. No architecture-specific code paths or instruction differences exist for this operation.

## Implementation Notes

1. **RELU as relu_min(0)**: RELU is cleverly implemented by reusing the `relu_min` function with a threshold of 0. `relu_min(threshold)` clamps values below the threshold to the threshold value, so `relu_min(0)` implements `max(x, 0)`.

2. **INT32 variant**: When the input dtype is INT32, a separate path `relu_tile_int32(idst)` is used, which calls `SFPU_UNARY_ONE_PARAM_KERNEL_FN_INT` with `vInt` instead of `vFloat`. The logic is the same but operates on integer registers.

3. **No runtime scalars needed**: Unlike RELU_MAX, RELU_MIN with a non-zero threshold, or LEAKY_RELU, plain RELU passes `packed_scalar1 = 0` and `packed_scalar2 = 0` as runtime args to the compute kernel. These are unused.

4. **Double-buffering**: Both CBs have capacity 2 and block size 1, enabling double-buffering where the reader can fill the next tile while compute processes the current one.

5. **Program caching**: The factory supports `override_runtime_arguments` for program caching. Only buffer addresses are updated on subsequent calls; the kernel configuration (compile-time args, defines) remains fixed.

6. **Op chain support**: The `SFPU_OP_CHAIN_0` mechanism supports chaining multiple unary operations in a single kernel dispatch. For standalone RELU, the chain contains only one operation.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the unary program factory implemented for SFPU operations like relu?"
   **Reason**: Needed to understand the overall factory structure, kernel paths, and work distribution strategy before reading source code.
   **Key Findings**: Confirmed the three-kernel pattern (reader/compute/writer), `split_work_to_cores` for distribution, and the specific kernel file paths.

2. **Query**: "How does the relu SFPU kernel work in tt-metal?"
   **Reason**: Needed to locate the SFPU kernel implementation and understand the macro/define system.
   **Key Findings**: Confirmed `relu_tile` calls `_relu_min_` with threshold 0, located `ckernel_sfpu_relu.h`, and identified `SFPU_OP_RELU_FAMILY_INCLUDE` define.

3. **Query**: "What is the _llk_math_eltwise_unary_sfpu_params_ function?"
   **Reason**: Needed to understand how the SFPU dispatch layer works between the API-level `relu_tile` and the low-level `relu_min` kernel.
   **Key Findings**: Learned that `_llk_math_eltwise_unary_sfpu_params_` sets DST write address, stalls for SFPU readiness, iterates through tile faces based on VectorMode, calls the provided functor for each face, and cleans up.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to determine the compute kernel path, define macros, init/func strings, and approx mode for RELU.
   **Key Information**: RELU maps to `SFPU_OP_RELU_FAMILY_INCLUDE`, uses default kernel `eltwise_sfpu.cpp`, init = `relu_tile_init()`, func = `relu_tile(0)`, approx_mode = `false`.

2. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Needed to trace the macro expansion chain from `relu_tile()` to the SFPU dispatch.
   **Key Information**: `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT` expands to `_llk_math_eltwise_unary_sfpu_params_` with template args `<vFloat, APPROX, 8, uint32_t>`.

3. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h`
   **Reason**: Needed to read the API-level wrapper that bridges the compute kernel's `relu_tile()` call to the SFPU kernel.
   **Key Information**: `relu_tile(idst)` calls `_relu_min_` with param 0; `relu_tile_init()` initializes SfpuType::relu_min.
