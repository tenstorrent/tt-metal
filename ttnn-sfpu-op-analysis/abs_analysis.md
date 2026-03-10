# ABS (Absolute Value) Implementation Analysis

## Overview

The ABS operation computes the element-wise absolute value of a tensor: `output[i] = |input[i]|`. It supports both floating-point types (BFloat16, Float32) and integer types (Int32). For floating-point values, the operation clears the sign bit of each element. For integers, it uses a dedicated `SFPABS` instruction.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` (class `UnaryProgramFactory`)

ABS is one of many unary operations that share the generic `eltwise_sfpu.cpp` compute kernel. The operation-specific behavior is injected at compile time via preprocessor defines (`SFPU_OP_CHAIN_0`), which expand to `abs_tile_init(); abs_tile(0);`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles per block (always 1 for ABS) |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | Flattened to pages |
| **Tensor layout** | TILE_LAYOUT (also supports ROW_MAJOR) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input |

### Layout Transformations

None. The operation is a pure element-wise unary; input and output have identical layout and shape.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile`, `abs_tile`, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_reserve_back(c_2, block_dim)`, `cb_push_back(c_2, block_dim)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

Data flows linearly: Reader fills CB c_0 one tile at a time; Compute consumes from c_0, applies ABS via SFPU, and produces to c_2; Writer drains c_2 one tile at a time back to DRAM/L1.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Notes:
- CB c_1 (tmp0) is NOT allocated for ABS. It is only allocated for HARDSHRINK, CBRT, or LOGIT operations.
- Both input and output CBs have capacity of 2 tiles (double-buffered), allowing reader/compute and compute/writer overlap.
- Page size is `single_tile_size` (derived from data format, e.g. 2048 bytes for BFloat16 32x32 tiles).

## Pipeline Pattern Summary

Both CB c_0 and c_2 use double-buffering (capacity = 2 tiles, block size = 1 tile). This enables:
- Reader and Compute to overlap: while Compute processes tile N from c_0, Reader can fill tile N+1.
- Compute and Writer to overlap: while Writer writes tile N from c_2, Compute can produce tile N+1.

## Index Calculations

The reader and writer kernels use `TensorAccessor` to map a linear page index to a physical DRAM/L1 address. The page index `i` ranges from `start_id` to `start_id + num_pages`. `TensorAccessor` encapsulates the interleaved bank mapping logic (page-to-bank assignment, bank offset calculation) configured via `TensorAccessorArgs` at compile time.

In the compute kernel, tile indexing is trivial: only tile index 0 is used because tiles are processed one at a time from the circular buffer.

## Memory Access Patterns

### Read Pattern

Sequential page access. The reader iterates linearly from `start_id` to `start_id + num_pages`, reading one page at a time via `noc_async_read_page`. Each page is a single tile. The access pattern across DRAM banks is interleaved (page `i` maps to bank `i % num_banks`).

### Write Pattern

Sequential page access, identical pattern to reader. The writer iterates from `start_id` to `start_id + num_pages`, writing one page at a time via `noc_async_write_page`. A `noc_async_write_barrier` is issued after the entire loop to ensure all writes complete.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major iteration) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g. 8x8) |
| **Total cores** | `num_cores` (determined by `split_work_to_cores`) |
| **Work per core** | `num_pages / num_cores` tiles (with remainder handling) |
| **Load balancing** | Two-group split: core_group_1 gets `ceil(num_pages/num_cores)` tiles, core_group_2 gets `floor(num_pages/num_cores)` tiles |

The `split_work_to_cores` utility divides `num_pages` tiles across all available cores. If tiles divide evenly, all cores get the same amount. Otherwise, `core_group_1` cores each get one extra tile compared to `core_group_2` cores. Separate compute kernels are created for each group (with different `per_core_block_cnt` compile-time args).

Core indexing is column-major: `core = {i / num_cores_y, i % num_cores_y}`.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for src_buffer (bank mapping, page size, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for dst_buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tiles) this core processes |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for ABS) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages (tiles) to read |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages (tiles) to write |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for ABS (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for ABS (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 | CB c_0 | Read tiles sequentially via TensorAccessor |
| compute | RISCV_2 (MATH) | N/A | CB c_0 | CB c_2 | copy_tile, abs_tile (SFPU), pack_tile |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 | Write tiles sequentially via TensorAccessor |

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Creates a `TensorAccessor` from compile-time args, then loops from `start_id` to `start_id + num_pages`, reading one page per iteration into CB c_0. Each iteration does: `cb_reserve_back` -> `noc_async_read_page` -> `noc_async_read_barrier` -> `cb_push_back`. Supports optional `BACKWARDS` define for reverse iteration (not used by ABS).

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Mirror of reader. Creates `TensorAccessor` for destination, loops sequentially writing one page per iteration from CB c_2. Uses `noc_async_writes_flushed()` per tile (not a full barrier) for throughput, with a single `noc_async_write_barrier()` at the end. Supports `OUT_SHARDED` define (not used by ABS in interleaved mode).

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
#include "api/compute/eltwise_unary/sfpu_split_includes.h"   // conditionally includes abs_tile API via SFPU_OP_COMPUTE_KERNEL_API_INCLUDE
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // number of tile-blocks to process on this core
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block; always 1 for ABS

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // initializes unpack (c_0) and pack (c_2) pipelines for SFPU mode
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve space in output CB for one block of tiles
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DEST registers for math/SFPU operations

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // block until reader has produced 1 tile in input CB

            copy_tile(tt::CBIndex::c_0, 0, 0);  // unpack tile 0 from CB c_0 into DEST register 0

// For ABS, SFPU_OP_CHAIN_0 expands to:
//   SFPU_OP_CHAIN_0_INIT_0   -> abs_tile_init();
//   SFPU_OP_CHAIN_0_FUNC_0   -> abs_tile(0);
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // executes abs_tile_init() then abs_tile(0) on DEST[0]
#endif

            tile_regs_commit();  // signal that DEST registers are ready for pack stage

            tile_regs_wait();  // wait for pack stage to be ready to consume DEST

            pack_tile(0, tt::CBIndex::c_2);  // pack tile from DEST[0] into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // free the consumed tile in input CB c_0

            tile_regs_release();  // release DEST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // publish the block of output tiles to writer
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File

- **Wormhole**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h`

Both architectures share identical floating-point implementation. The integer variant has minor addressing differences.

#### Annotated SFPU Kernel Source

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

// Floating-point absolute value: clears the sign bit of each element
// APPROXIMATION_MODE is unused for abs (no approximation needed - it is exact)
// ITERATIONS defaults to 8, processing 8 rows of 64 elements = 512 elements
// A 32x32 tile has 4 faces of 16x16 = 1024 elements total;
// each call to calculate_abs processes one face (16 rows x 16 cols = 256 elements),
// but since SFPU processes 64 elements per row (vectorized),
// 8 iterations x 2 rows per dst_reg++ stride = 16 rows covers one face.
// The _llk_math_eltwise_unary_sfpu_params_ dispatcher calls this 4 times (RC mode)
// to cover all 4 faces.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() {
    // SFPU microcode: process one face of the tile
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];        // load a row of elements from current DEST register position
        dst_reg[0] = sfpi::abs(v);    // compute abs by clearing sign bit (bit 31); compiles to SFPABS with SFPABS_MOD1_FLOAT
        dst_reg++;                    // advance DEST register pointer by stride 2 (moves to next row pair)
    }
}

// Integer (int32) absolute value: uses raw SFPU instructions
// Same iteration structure as floating-point variant
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // SFPU microcode: process one face of the tile for int32 data
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, 4, 3, 0);      // load int32 value from DEST into SFPU LREG[0];
                                       // arg1=1: LREG index, arg2=4: int32 format, arg3=3: addr_mod
        TTI_SFPABS(0, 1, 0, 0);       // compute absolute value of int32 in LREG[1], store to LREG[0]
                                       // arg1=0: dst LREG, arg2=1: src LREG, arg3/4: mode/flags
        TTI_SFPSTORE(0, 4, 3, 0);     // store result from LREG[0] back to DEST;
                                       // arg2=4: int32 format, arg3=3: addr_mod
        dst_reg++;                     // advance DEST register pointer
    }
}
}  // namespace sfpu
}  // namespace ckernel
```

#### LLK Wrapper

The LLK layer (`llk_math_eltwise_unary_sfpu_abs.h`) connects `abs_tile()` to `calculate_abs()`:

```cpp
// abs_tile_init() -> llk_math_eltwise_unary_sfpu_abs_init<APPROX>()
//   -> llk_math_eltwise_unary_sfpu_init<SfpuType::abs, APPROX>()
//      Configures SFPU pipeline for abs operation type.

// abs_tile(idst) -> llk_math_eltwise_unary_sfpu_abs<APPROX>(idst)
//   -> _llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_abs<APPROX>, dst_index, VectorMode::RC)
//      Sets DEST write address, stalls until SFPU is ready, then calls calculate_abs
//      4 times (once per face in RC mode), advancing the face pointer between calls.
```

#### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `sfpi::abs(vFloat)` / `SFPABS` with `SFPABS_MOD1_FLOAT` | Clears the sign bit (bit 31) of a floating-point value, computing `|x|`. Compiles to the hardware `SFPABS` instruction. |
| `TTI_SFPABS(dst, src, 0, 0)` | Integer absolute value instruction. Computes the absolute value of an integer in an SFPU local register. |
| `TT_SFPLOAD` | Loads data from DEST register file into an SFPU local register (LREG). Used in int32 path. |
| `TTI_SFPSTORE` | Stores data from an SFPU local register back to the DEST register file. Used in int32 path. |
| `dst_reg[0]` (read) | Implicit `SFPLOAD` -- reads a vector of elements from the current DEST register row into SFPU. |
| `dst_reg[0]` (write) | Implicit `SFPSTORE` -- writes a vector of elements from SFPU back to the current DEST register row. |
| `dst_reg++` | Advances the DEST register pointer by `SFP_DESTREG_STRIDE` (2), moving to the next pair of rows. |

#### SFPU Register Usage

- **DEST register file**: The primary data storage. `copy_tile` unpacks a tile from CB c_0 into DEST. After SFPU processing, `pack_tile` reads from DEST into CB c_2. DEST holds the full 32x32 tile organized as 4 faces of 16x16.
- **SFPU Local Registers (LREGs)**: In the floating-point path, `dst_reg[0]` read/write implicitly uses LREGs for the `sfpi::abs` computation. In the int32 path, `TT_SFPLOAD` explicitly loads into LREG[1], `TTI_SFPABS` operates on LREG[1] producing to LREG[0], and `TTI_SFPSTORE` writes LREG[0] back.
- **vFloat**: SFPI's vector register type representing a row of elements processed in parallel by the SFPU.

#### SFPU Execution Flow

1. **Tile acquisition**: `tile_regs_acquire()` locks DEST registers for exclusive SFPU/math use.
2. **Unpack to DEST**: `copy_tile(c_0, 0, 0)` unpacks tile 0 from CB c_0 into DEST register 0. This converts from the CB data format to DEST's internal format.
3. **SFPU init**: `abs_tile_init()` calls `llk_math_eltwise_unary_sfpu_abs_init()`, which configures the SFPU pipeline for the abs operation type.
4. **SFPU dispatch**: `abs_tile(0)` calls `_llk_math_eltwise_unary_sfpu_params_` which:
   - Sets the DEST write address to tile index 0
   - Issues `TTI_STALLWAIT(STALL_SFPU, MATH)` to ensure SFPU is ready
   - In RC (full tile) mode, loops 4 times (one per face):
     - Calls `calculate_abs()` which processes 8 iterations x 1 row = 8 row-pairs per face
     - Advances DEST pointer by 16 rows (2 `SETRWC` instructions advancing by 8 each) to the next face
   - Issues `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)` to wait for SFPU completion
5. **Pack**: `pack_tile(0, c_2)` reads the result from DEST[0] and packs it into CB c_2.
6. **Release**: `tile_regs_release()` frees DEST for the next tile.

#### SFPU Configuration

- **Math fidelity**: `MathFidelity::HiFi4` (highest fidelity, though irrelevant for abs which is exact).
- **Math approx mode**: `false` for ABS. The `get_op_approx_mode` function returns `false` by default, and ABS has no special case. The `APPROXIMATION_MODE` template parameter in `calculate_abs` is unused.
- **Preprocessor defines**: `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE=1` (enables inclusion of `compute_kernel_api.h` which provides `abs_tile`/`abs_tile_init`). `SFPU_OP_CHAIN_0` is defined to expand to the init + function calls.
- **Data type defines**: One of `INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, or `INP_FLOAT` is set depending on input dtype.
- **fp32_dest_acc_en**: Configurable; when true, DEST accumulates in FP32 precision.
- **unpack_to_dest_mode**: Default for most dtypes; `UnpackToDestFp32` when `preserve_fp32_precision` is set.

#### Hardware Compatibility Notes

- **Wormhole B0 vs Blackhole**: The floating-point `calculate_abs()` is identical on both architectures. The int32 `calculate_abs_int32()` differs only in addressing mode parameters:
  - Wormhole: `TT_SFPLOAD(1, 4, 3, 0)` / `TTI_SFPSTORE(0, 4, 3, 0)` (addr_mod=3)
  - Blackhole: `TT_SFPLOAD(1, 12, ADDR_MOD_7, 0)` / `TTI_SFPSTORE(0, 12, ADDR_MOD_7, 0)` (addr_mod=ADDR_MOD_7, format=12)
  - These differences reflect different register file addressing conventions between the two architectures.
- The `SFPABS` instruction is available on both Wormhole and Blackhole and operates identically for floating-point absolute value.

## Implementation Notes

1. **Simplicity**: ABS is one of the simplest SFPU operations. The floating-point implementation is a single instruction (`SFPABS`) per element row, with no approximation, no LUT, and no iterative computation.

2. **Shared compute kernel**: ABS reuses the generic `eltwise_sfpu.cpp` kernel shared by many unary operations. The operation-specific logic is entirely injected via the `SFPU_OP_CHAIN_0` macro, which expands to `abs_tile_init(); abs_tile(0);`.

3. **No temporary CB**: Unlike operations such as HARDSHRINK or CBRT, ABS does not require the temporary circular buffer (CB c_1).

4. **No scalar parameters**: The runtime args `packed_scalar1` and `packed_scalar2` are always 0 for ABS, as it takes no parameters.

5. **Cached program reuse**: The `override_runtime_arguments` method only updates buffer addresses, enabling program reuse across calls with different tensor allocations but same shapes.

6. **Column-major core iteration**: Cores are assigned work in column-major order (`{i / num_cores_y, i % num_cores_y}`), which means work fills down columns first before moving to the next column.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary SFPU program factory work in ttnn? What kernels does it use (reader, compute, writer)? How are circular buffers configured and how is work distributed across cores for elementwise unary operations?"
   **Reason**: Needed to understand the overall architecture of the unary program factory before reading source code.
   **Key Findings**: Confirmed three-kernel architecture (reader/compute/writer), double-buffered CBs at c_0 and c_2, work split via `split_work_to_cores` into two core groups, and operation-specific compute kernel selection via `get_compute_kernel_path`.

2. **Query**: "How does the ABS (absolute value) SFPU operation work in tt-metal? What compute kernel and SFPU kernel files implement it?"
   **Reason**: Needed to locate the specific SFPU kernel files for ABS.
   **Key Findings**: Identified `ckernel_sfpu_abs.h` as the SFPU kernel, `abs_tile`/`abs_tile_init` as the compute API, and confirmed the float path uses `sfpi::abs()` while int32 uses raw `SFPABS` instructions.

3. **Query**: "How does sfpi::abs work on vFloat values? What SFPU instruction does it compile to? How does dst_reg work?"
   **Reason**: Needed precise understanding of the SFPU intrinsics used in the kernel.
   **Key Findings**: `sfpi::abs` compiles to `SFPABS` with `SFPABS_MOD1_FLOAT` modifier (clears sign bit). `dst_reg[0]` accesses current DEST register row, `dst_reg++` advances by stride 2.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to trace how ABS maps to its compute kernel and SFPU defines.
   **Key Information**: ABS maps to `eltwise_sfpu.cpp` (default case in `get_compute_kernel_path`), generates init/func pair `abs_tile_init()`/`abs_tile(0)`, uses `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` macro, and `get_op_approx_mode` returns false.

2. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Needed to understand the SFPU dispatch mechanism (`_llk_math_eltwise_unary_sfpu_params_`).
   **Key Information**: The function sets DEST write address, stalls until SFPU ready, loops over 4 faces in RC mode calling the SFPU function, then stalls until SFPU complete. Each face call invokes `calculate_abs` which processes 8 row iterations.
