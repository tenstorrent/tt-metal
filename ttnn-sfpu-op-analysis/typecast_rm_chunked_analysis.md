# TYPECAST (Row-Major Chunked) -- SFPU Operation Analysis

## Operation Overview

| Field | Value |
|-------|-------|
| **Operation Name** | `ttnn::typecast` (row-major chunked variant) |
| **Registered Name** | `ttnn::typecast` |
| **Namespace** | `ttnn::prim` (device operation), `ttnn::operations::copy` (public API) |
| **Category** | Copy / Data Type Conversion |
| **Program Factory** | `TypecastRowMajorChunkedProgramFactory` |
| **Selection Criteria** | Non-sharded, no sub_core_grids, `Layout::ROW_MAJOR` input |

### Purpose

The typecast operation converts tensor data between different numeric data types (e.g., Float16_b, Float32, Int32, UInt16, UInt32, Bfp8_b, Bfp4_b) entirely on-device. The "row-major chunked" variant handles tensors stored in ROW_MAJOR layout by splitting each row into fixed-size chunks (up to 1024 elements), processing them through the SFPU for type conversion, and writing the results back. This chunking strategy avoids requiring the entire row to fit in a single circular buffer page.

### Supported Type Conversions

The SFPU typecast kernel supports a large matrix of conversions. Some conversions require SFPU math; others are handled purely by the unpacker/packer hardware:

**SFPU-required conversions:**
- Float16_b / Float32 / Bfp8_b / Bfp4_b <-> UInt16
- Float16_b / Float32 / Bfp8_b / Bfp4_b <-> Int32
- Float16_b / Float32 / Bfp8_b / Bfp4_b <-> UInt32
- Float32 -> Float16_b (explicit rounding control)
- Int32 <-> Float32, Int32 <-> Float16_b
- UInt16 <-> UInt32, Int32 <-> UInt16
- UInt32 <-> Float16_b, UInt32 <-> Float32

**Packer/unpacker-only conversions (no SFPU kernel):**
- Float16_b <-> Float32 (fp16b->fp32 direction)
- Bfp8_b <-> Float16_b, Bfp8_b <-> Float32
- Bfp4_b <-> Float16_b, Bfp4_b <-> Bfp8_b, Bfp4_b <-> Float32

---

## Program Factory Structure

### File
`ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_rm_chunked_program_factory.cpp`

### Factory Selection Logic

The `TypecastDeviceOperation::select_program_factory` method (in `typecast_device_op.cpp`) selects this factory when:
1. Input tensor is NOT sharded
2. No `sub_core_grids` override is specified
3. Input tensor layout is `Layout::ROW_MAJOR`

Other factories handle sharded inputs (`TypecastShardedProgramFactory`), subgrid inputs (`TypecastSubgridProgramFactory`), and tiled inputs (`TypecastProgramFactory`).

### Operation Parameters

Defined in `TypecastParams` (`typecast_device_op_types.hpp`):

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dtype` | `DataType` | Source data type |
| `output_dtype` | `DataType` | Target data type |
| `output_memory_config` | `MemoryConfig` | Output memory configuration |
| `fp32_dest_acc_en` | `bool` | Enable 32-bit destination accumulator (required for 32-bit types) |
| `preserve_fp32_precision` | `bool` | Use `UnpackToDestFp32` mode for input CB |
| `bfp8_pack_precise` | `bool` | Enable precise BFP8 packing |
| `sub_core_grids` | `optional<CoreRangeSet>` | Not used by this factory (must be nullopt) |

### Shared Variables (Cached Program)

```cpp
struct shared_variables_t {
    tt::tt_metal::KernelHandle typecast_reader_kernel_id;
    tt::tt_metal::KernelHandle typecast_writer_kernel_id;
    uint32_t num_cores;
    uint32_t chunks_per_row;
    uint32_t input_chunk_size_bytes;
    uint32_t output_chunk_size_bytes;
};
```

---

## Work Distribution

### Chunking Strategy

Rows are split into chunks of at most **1024 elements**. This is a fixed upper bound chosen to keep circular buffer page sizes manageable.

```
ChunkSizeConfig:
  elements_per_full_chunk = min(1024, row_width_elements)
  full_chunks_per_row = row_width_elements / elements_per_full_chunk
  partial_chunks_per_row = (remainder > 0) ? 1 : 0
```

Each chunk is independently read, processed through the compute kernel, and written back. The input and output chunk sizes differ because input and output element sizes differ (e.g., 2 bytes for Float16_b vs 4 bytes for Float32).

### Core Assignment

Work is split **by rows** across cores using `split_work_to_cores`:
- Two core groups may have different row counts (group 1 gets `ceil`, group 2 gets `floor`)
- Each core processes complete rows (all full chunks + optional partial chunk per row)
- Runtime args: `{buffer_address, num_rows_for_core, start_row_id}`

---

## Circular Buffer Configuration

| CB Index | Name | Data Format | Page Size | Num Pages | Purpose |
|----------|------|-------------|-----------|-----------|---------|
| `c_0` | Input CB | Input dtype format | `input_full_chunk_size_bytes` | 2 (double buffer) | Holds input chunks read from DRAM |
| `c_2` | Output CB | Output dtype format | `output_full_chunk_size_bytes` | 2 (double buffer) | Holds output chunks to write to DRAM |

Both CBs use **double buffering** (2 pages) to overlap data movement with compute. The page size is set to the full chunk size; partial chunks use less than a full page but the CB allocation accommodates the maximum.

---

## Kernel Implementations

### Reader Kernel

#### File
`ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_typecast_rm_chunked.cpp`

#### Annotated Source

```cpp
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args: buffer address, number of rows this core handles, first row index
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_row_id = get_arg_val<uint32_t>(2);

    // Compile-time args define the chunking parameters
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t full_chunk_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t full_chunks_per_row = get_compile_time_arg_val(2);
    constexpr uint32_t partial_chunk_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t partial_chunks_per_row = get_compile_time_arg_val(4);  // 0 or 1, used for constexpr branching
    constexpr uint32_t row_page_size_bytes = get_compile_time_arg_val(5);
    // TensorAccessor args are appended starting at compile-time arg index 6
    constexpr auto src_args = TensorAccessorArgs<6>();

    constexpr uint32_t onepage = 1;

    // TensorAccessor maps logical row indices to physical NoC addresses
    // using the buffer's actual page size (row_page_size_bytes)
    const auto s = TensorAccessor(src_args, src_addr, row_page_size_bytes);

    const uint32_t end_row_id = start_row_id + num_rows;

    for (uint32_t row_id = start_row_id; row_id < end_row_id; ++row_id) {
        // Read all full-sized chunks for this row
        for (uint32_t chunk_idx = 0; chunk_idx < full_chunks_per_row; ++chunk_idx) {
            cb_reserve_back(cb_id_in, onepage);           // Wait for space in input CB
            const uint32_t l1_write_addr = get_write_ptr(cb_id_in);

            // Compute byte offset within the row for this chunk
            const uint32_t byte_offset = chunk_idx * full_chunk_size_bytes;
            // get_noc_addr with byte_offset reads a sub-page region
            const uint64_t chunk_noc_addr = s.get_noc_addr(row_id, byte_offset);
            noc_async_read(chunk_noc_addr, l1_write_addr, full_chunk_size_bytes);

            noc_async_read_barrier();                     // Wait for DMA completion
            cb_push_back(cb_id_in, onepage);              // Signal compute kernel
        }

        // If the row width is not evenly divisible by 1024, read the remainder
        if constexpr (partial_chunks_per_row > 0) {
            cb_reserve_back(cb_id_in, onepage);
            const uint32_t l1_write_addr = get_write_ptr(cb_id_in);

            const uint32_t byte_offset = full_chunks_per_row * full_chunk_size_bytes;
            const uint64_t chunk_noc_addr = s.get_noc_addr(row_id, byte_offset);
            noc_async_read(chunk_noc_addr, l1_write_addr, partial_chunk_size_bytes);

            noc_async_read_barrier();
            cb_push_back(cb_id_in, onepage);
        }
    }
}
```

### Writer Kernel

#### File
`ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_typecast_rm_chunked.cpp`

#### Annotated Source

```cpp
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_row_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t full_chunk_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t full_chunks_per_row = get_compile_time_arg_val(2);
    constexpr uint32_t partial_chunk_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t partial_chunks_per_row = get_compile_time_arg_val(4);
    constexpr uint32_t row_page_size_bytes = get_compile_time_arg_val(5);
    constexpr auto dst_args = TensorAccessorArgs<6>();

    constexpr uint32_t onepage = 1;

    // TensorAccessor for the output buffer uses the output row page size
    const auto s = TensorAccessor(dst_args, dst_addr, row_page_size_bytes);

    const uint32_t end_row_id = start_row_id + num_rows;

    for (uint32_t row_id = start_row_id; row_id < end_row_id; ++row_id) {
        // Write all full-sized output chunks for this row
        for (uint32_t chunk_idx = 0; chunk_idx < full_chunks_per_row; ++chunk_idx) {
            cb_wait_front(cb_id_out, onepage);            // Wait for compute to produce data
            const uint32_t l1_read_addr = get_read_ptr(cb_id_out);

            const uint32_t byte_offset = chunk_idx * full_chunk_size_bytes;
            const uint64_t chunk_noc_addr = s.get_noc_addr(row_id, byte_offset);
            noc_async_write(l1_read_addr, chunk_noc_addr, full_chunk_size_bytes);

            noc_async_writes_flushed();                   // Flush to NoC (non-blocking)
            cb_pop_front(cb_id_out, onepage);             // Free CB page for reuse
        }

        // Write partial chunk remainder
        if constexpr (partial_chunks_per_row > 0) {
            cb_wait_front(cb_id_out, onepage);
            const uint32_t l1_read_addr = get_read_ptr(cb_id_out);

            const uint32_t byte_offset = full_chunks_per_row * full_chunk_size_bytes;
            const uint64_t chunk_noc_addr = s.get_noc_addr(row_id, byte_offset);
            noc_async_write(l1_read_addr, chunk_noc_addr, partial_chunk_size_bytes);

            noc_async_writes_flushed();
            cb_pop_front(cb_id_out, onepage);
        }
    }
    noc_async_write_barrier();  // Final barrier ensures all writes complete before kernel exits
}
```

### Compute Kernel

#### File
`ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp`

#### Annotated Source

```cpp
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"  // Provides typecast_tile<IN,OUT> and typecast_tile_init<IN,OUT>

void kernel_main() {
    // per_core_block_cnt = total chunks this core processes (rows * chunks_per_row)
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    // per_core_block_dim = number of tiles per block; always 1 for this factory
    // because each chunk is treated as a single "tile" for the copy_tile/pack_tile flow
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr uint32_t input_cb = get_compile_time_arg_val(2);   // c_0
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);  // c_2

    // Initialize the SFPU for the specific typecast direction.
    // init_sfpu sets up unpack/pack configuration for the input/output CB pair.
    init_sfpu(input_cb, output_cb);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve output space for per_core_block_dim pages (always 1)
        cb_reserve_back(output_cb, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to destination registers for math operations
            tile_regs_acquire();

            // Wait for the reader to produce one input chunk
            cb_wait_front(input_cb, 1);

            // Copy the chunk from input CB to destination register index 0.
            // This triggers the unpacker to load data into DEST registers.
            copy_tile(input_cb, 0, 0);

            // TYPECAST_LLK_INIT and TYPECAST_LLK are preprocessor defines set by the host.
            // They expand to typecast_tile_init<IN_FMT, OUT_FMT>() and typecast_tile<IN_FMT, OUT_FMT>(0).
            // The init call configures SFPU instruction templates and macro registers.
            // The main call runs the SFPU typecast kernel on DEST register 0.
            TYPECAST_LLK_INIT();
            TYPECAST_LLK(0);

            // Release DEST registers to packer
            tile_regs_commit();

            // Wait for packer to be ready
            tile_regs_wait();

            // Pack the converted data from DEST register 0 into the output CB
            pack_tile(0, output_cb);

            // Free the input CB page so the reader can reuse it
            cb_pop_front(input_cb, 1);

            // Release DEST registers back to math unit
            tile_regs_release();
        }
        // Signal the writer that per_core_block_dim output pages are ready
        cb_push_back(output_cb, per_core_block_dim);
    }
}
```

#### Compute Kernel Defines

The program factory injects two preprocessor defines:

```cpp
unary_defines["TYPECAST_LLK_INIT"] = "typecast_tile_init<IN_FMT_INT, OUT_FMT_INT>"
unary_defines["TYPECAST_LLK"]      = "typecast_tile<IN_FMT_INT, OUT_FMT_INT>"
```

Where `IN_FMT_INT` and `OUT_FMT_INT` are the integer values of `tt::DataFormat` for the input and output types. This allows the same compute kernel source to handle all typecast directions through compile-time specialization.

#### Compute Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| `math_fidelity` | `MathFidelity::HiFi4` | Maximum precision for type conversion |
| `fp32_dest_acc_en` | From `TypecastParams` | Required for 32-bit source/dest types |
| `unpack_to_dest_mode` | `UnpackToDestFp32` if `preserve_fp32_precision` | Preserves full FP32 precision in DEST |
| `bfp8_pack_precise` | From `TypecastParams` | Enables precise rounding for BFP8 output |
| `math_approx_mode` | `false` | Always exact; no approximations for typecast |

---

### SFPU Kernel Implementation

This section provides a deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to via the `typecast_tile` / `typecast_tile_init` API.

#### SFPU Kernel API File
`tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h`

This thin wrapper calls into the LLK layer:

```cpp
template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE>
ALWI void typecast_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_typecast<APPROX, IN_DTYPE, OUT_DTYPE>(idst)));
}

template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE>
ALWI void typecast_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_typecast_init<APPROX, IN_DTYPE, OUT_DTYPE>()));
}
```

#### LLK Dispatch File
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h`

The `llk_math_eltwise_unary_sfpu_typecast` function uses `if constexpr` chains to select the appropriate SFPU kernel based on `(in_format, out_format)` pairs. Each branch calls `_llk_math_eltwise_unary_sfpu_params_` with the specific `calculate_typecast_*` function pointer and `ITERATIONS=8` (processing 8 DEST rows per invocation).

The `llk_math_eltwise_unary_sfpu_typecast_init` function similarly dispatches to the correct `init_typecast_*` function.

#### SFPU Kernel File (Blackhole)
`tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h`

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2025 Jason Davies <jason@jasondavies.com>
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// ============================================================================
// FP32 -> UINT16: Clamp negative to 0, then round to unsigned 16-bit integer
// Throughput: 2 cycles per DEST row via SFPLOADMACRO pipelining
// Pipeline stages: Load -> Simple(max(v,0)) -> Round(rnd to L16) -> Store(L16)
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint16_()
{
    // The SFPLOADMACRO instruction orchestrates a multi-stage pipeline.
    // Each iteration alternates between LREG0 and LREG1 (v = d & 1)
    // to allow overlapping of pipeline stages across consecutive rows.
    //
    // Stage breakdown per row:
    //   Load:   Read FP32 value from DEST into LREG[v]
    //   Simple: v = max(v, 0.0)  -- clamp negatives to zero
    //   Round:  Convert FP32 to UINT16 with rounding (SFPSTOCHRND)
    //   Store:  Write 16-bit result back to DEST
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, v >> 2);
        TTI_SFPNOP;  // Pipeline stall: Simple and Round cannot overlap at same time slot
    }
    // Drain pipeline: 3 NOPs to flush the last entries through Round and Store stages
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// UINT16 -> FP16B: Cast unsigned integer to bfloat16
// Throughput: 1 cycle per DEST row -- the simplest conversion
// Pipeline: Load(LO16) -> Simple(cast) -> Round(rnd to L16) -> Store(L16)
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp16b_()
{
    // LO16 load mode reads the lower 16 bits as unsigned integer.
    // SFPCAST converts integer representation to floating point.
    // The result is rounded to FP16B and stored.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// INT32 -> FP16B: Convert signed 32-bit integer to bfloat16
// Throughput: 4 cycles per DEST row
// Algorithm: abs(v) -> cast to float -> restore sign -> round to FP16B
// Uses LREG0=0.0, LREG1=-2^31 as constants for SFPMAD sign correction
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_()
{
    // LREG4 is used as a temporary register (t).
    // LREG0 = 0.0 and LREG1 = -2^31 are loaded as constants.
    // The sign bit of abs(v) is stored in LREG7 via SFPSHFT2 (shift right 31).
    // SFPMAD uses indirect VA addressing: if sign=0, VA=LREG0 (add 0); if sign=1, VA=LREG1 (add -2^31).
    // This corrects for the magnitude error introduced by abs() on negative numbers.
    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);       // LREG0 = 0.0
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00);  // LREG1 = -2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1);  // Alternate LREG2/LREG3 (LREG0,1 hold constants)
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPABS(0, v, t, 0);                                         // t = abs(v)
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5);           // LREG7 = t >> 31 (sign bit)
        TTI_SFPCAST(t, t, 0);                                          // t = float(t)
    }
    // 5 NOPs to drain the longer pipeline (Load+Simple+MAD+Round+Store)
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// FP32 -> INT32: Convert IEEE 754 float to two's complement signed integer
// Throughput: 1 iteration per loop body (no SFPLOADMACRO pipelining)
// Algorithm: Extract exponent and mantissa, shift mantissa, handle overflow/sign
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_int32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Load FP32 value from DEST row into LREG0
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        // Initialize result = 0 (for underflow case: exp < 0)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);

        // Extract debiased exponent; set condition code if sign bit set or exp < 0
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2,
            sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // Set result = INT_MIN (0x80000000) for overflow case
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);
        // exp -= 31; enables lanes where |value| < 2^31 (valid int32 range)
        TTI_SFPIADD(-31 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2,
            sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 8 (adjust shift: exman8 gives mantissa at bit 23, we want bit 31 max)
        TTI_SFPIADD(8, p_sfpu::LREG2, p_sfpu::LREG2,
            sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // Extract 8-bit mantissa with implicit leading 1 at bit 23
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        // Shift mantissa left by (exp - 23) to get integer magnitude
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // Re-enable all lanes
        TTI_SFPENCC(0, 0, 0, 0);

        // Check if input was negative
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // Negate result via two's complement for negative inputs
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1,
            sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // Re-enable all lanes
        TTI_SFPENCC(0, 0, 0, 0);

        // Store 32-bit integer result back to DEST
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}

// ============================================================================
// FP32 -> UINT32: Convert float to unsigned 32-bit integer
// Similar to FP32->INT32 but clamps negatives to 0, overflow to 0xFFFFFFFF
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);  // result = 0

        // Only enable lanes where input >= 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
        // Extract exponent with sign/exponent condition codes
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2,
            sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // result = 0xFFFFFFFF for overflow case
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
        // exp -= 32; enables lanes where value < 2^32
        TTI_SFPIADD(-32 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2,
            sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 9
        TTI_SFPIADD(9, p_sfpu::LREG2, p_sfpu::LREG2,
            sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // Extract mantissa and shift
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}

// ============================================================================
// FP32 -> FP16B: Precise rounding from 32-bit to 16-bit bfloat
// Throughput: 3 cycles per DEST row via SFPLOADMACRO with 2 macros
// Uses bit manipulation: extract LSB of truncated bits, add rounding bias
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_fp16b_()
{
    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1;
        // Macro 0 [a]: Load FP32, shift right 16 to get upper 16 bits
        TT_SFPLOADMACRO((0 << 2) | (a & 3), 0, ADDR_MOD_7, a >> 2);
        // Macro 1 [b]: Load same value, add 0x7FFF rounding bias
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), 0, ADDR_MOD_6, b >> 2);
        // Extract LSB (round-to-nearest-even bit)
        TT_SFPAND(0, p_sfpu::LREG12, a, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// UINT16 -> FP32: Cast unsigned 16-bit to 32-bit float
// Throughput: 1 cycle per DEST row -- simplest conversion
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp32_()
{
    constexpr int v = p_sfpu::LREG0;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // LO16 load reads 16-bit unsigned, SFPCAST converts to FP32
        TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// INT32 -> FP32: Convert signed 32-bit integer to IEEE 754 float
// Throughput: 4 cycles per DEST row
// Algorithm: abs -> store sign in LREG7 -> cast magnitude -> SFPMAD with sign correction
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp32_()
{
    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);       // 0.0
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00);  // -2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1);
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPABS(0, v, t, 0);                                         // t = |v|
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5);           // LREG7 = sign bit
        TTI_SFPCAST(t, t, 0);                                          // t = float(|v|)
        // SFPLOADMACRO pipeline: setsgn(t,v) restores sign, MAD adds -2^31 if needed
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// UINT32 -> FP16B: Convert unsigned 32-bit to bfloat16
// Throughput: 3 cycles per DEST row
// Algorithm: Extract sign bit to LREG7 -> clear sign -> cast -> SFPMAD adds 2^31 if needed -> round
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp16b_()
{
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);       // 0.0
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // +2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1);
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPSHFT2(v, p_sfpu::LREG12, p_sfpu::LREG7, 5);  // LREG7 = MSB (pseudo-sign for uint32)
        TT_SFPSETSGN(0, v, v, 1);                             // Clear sign bit
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// UINT32 -> FP32: Convert unsigned 32-bit integer to 32-bit float
// Throughput: 3 cycles per DEST row, uses 3 SFPLOADMACRO macros
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp32_()
{
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);       // 0.0
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // +2^31

    constexpr int a  = p_sfpu::LREG2;
    constexpr int b  = p_sfpu::LREG3;
    constexpr int L7 = p_sfpu::LREG7;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Three macros coordinate: [a] loads and clears sign, [b] casts and does MAD,
        // [L7] handles sign bit extraction and store
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_7, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_7, b >> 2);
        TTI_SFPLOADMACRO((2 << 2) | (L7 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, L7 >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// UINT16 -> UINT32: Zero-extend 16-bit unsigned to 32-bit
// Throughput: 1 cycle per DEST row
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_uint32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // LO16 load reads lower 16 bits, zero-extends to 32 bits, stores as INT32
        TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::LO16, ADDR_MOD_6, 0);
    }
    TTI_SFPNOP;
}

// ============================================================================
// UINT32 -> UINT16: Truncate to 16 bits with saturation
// Throughput: 2 cycles per DEST row, uses 2 macros
// Algorithm: [a] checks if value > 0 (sets to -1 i.e. 0xFFFF if true, else 0)
//            [b] OR with original lower 16 bits -> saturated 16-bit result
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_uint16_()
{
    constexpr int a = p_sfpu::LREG0;
    constexpr int b = p_sfpu::LREG1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::LO16, ADDR_MOD_7, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_6, b >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// INT32 -> UINT16: Clamp signed integer to [0, 65535]
// Throughput: 3 cycles per DEST row
// Algorithm: Load as INT32 -> SFPCAST to FP32 -> max(0, v) -> SFPSTOCHRND to UINT16
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_uint16_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_6, a >> 2);
        TT_SFPCAST(a, a, 0);   // Cast INT32 to FP32 representation
        TTI_SFPNOP;             // Pipeline stall between cast and max/round
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// INIT FUNCTIONS: Configure SFPLOADMACRO instruction templates and macros
// These run once before the main loop to program the SFPU pipeline scheduler
// ============================================================================

// ... (13 init functions configure SFPCONFIG registers for each conversion) ...

} // namespace sfpu
} // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOADMACRO` | Orchestrates a pipelined sequence of Load/Simple/MAD/Round/Store stages. Achieves high throughput by overlapping operations on different DEST rows. |
| `SFPLOAD` | Loads a single DEST row into an SFPU LREG register. Used in non-pipelined kernels (fp32_to_int32, fp32_to_uint32). |
| `SFPLOADI` | Loads an immediate constant into an LREG register (e.g., 0.0, -2^31, +2^31, 0x7FFF). |
| `SFPSTORE` | Stores an LREG value back to DEST. Used in non-pipelined kernels. |
| `SFPCAST` | Converts between integer and floating-point representations within LREG. |
| `SFPABS` | Computes absolute value of an LREG. Used in int32->float conversions. |
| `SFPEXEXP` | Extracts the debiased exponent from an IEEE 754 float. Sets condition codes for overflow/underflow detection. |
| `SFPEXMAN` | Extracts the mantissa (with implicit leading 1) from a float. |
| `SFPSHFT` | Shifts an LREG value left/right by an amount in another LREG. |
| `SFPSHFT2` | Shift operation variant used for extracting sign bits (shift right by 31). |
| `SFPIADD` | Integer add with immediate or register source. Also used for condition code manipulation. |
| `SFPSETCC` | Sets condition codes based on register comparison (e.g., LREG < 0, LREG >= 0). |
| `SFPENCC` | Enables all condition code lanes (re-enables all SIMD lanes). |
| `SFPSETSGN` | Sets or clears the sign bit of a floating-point value. |
| `SFPAND` | Bitwise AND between LREGs. Used in fp32_to_fp16b for round-to-nearest-even. |
| `SFPSWAP` | Min/max swap operation. Used to clamp values (e.g., max(0, v)). |
| `SFPSTOCHRND` (via `SFP_STOCH_RND`) | Stochastic/deterministic rounding for format conversion (FP32->FP16B, FP32->UINT16). |
| `SFPMAD` | Fused multiply-add. Used with indirect VA addressing for sign-dependent correction. |
| `SFPCONFIG` | Programs SFPLOADMACRO configuration registers (macro definitions, store modes). |
| `SFPNOP` | Pipeline NOP for draining SFPLOADMACRO pipeline stages. |
| `SFPGT` | Greater-than comparison. Used in uint32_to_uint16 for saturation. |
| `SFPOR` | Bitwise OR. Used in uint32_to_uint16 for combining saturation mask with value. |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `LREG0` | General purpose / constant 0.0 (in int32 conversions) / alternating data register |
| `LREG1` | General purpose / constant -2^31 or +2^31 (in int32 conversions) / alternating data register |
| `LREG2` | Temporary (exponent storage) / alternating data register / constant `b` in fp32_to_fp16b |
| `LREG3` | Alternating data register in int32/uint32 conversions |
| `LREG4` | Temporary (`t`) for intermediate values in int32 conversions |
| `LREG7` | Sign bit storage (extracted via SFPSHFT2 >> 31), used by SFPMAD indirect VA |
| `LREG12` | `vConstIntPrgm0` -- programmable constant register |
| `LREG13` | `vConstIntPrgm1` -- programmable constant register |
| `DEST` | Source/destination for tile data; rows are loaded/stored one at a time |

#### SFPU Execution Flow

1. **Initialization**: `init_sfpu(input_cb, output_cb)` configures the unpacker and packer for the input/output CB data formats.

2. **Per-chunk processing**:
   - `cb_wait_front(input_cb, 1)` -- wait for reader to produce a chunk
   - `copy_tile(input_cb, 0, 0)` -- unpacker loads chunk data into DEST register 0 (converting from CB data format to DEST format)
   - `TYPECAST_LLK_INIT()` -- calls `typecast_tile_init<IN,OUT>()` which:
     - Dispatches to the appropriate `init_typecast_*` function
     - Programs SFPCONFIG registers with instruction templates and macro definitions
     - Sets up constant registers (LREG0, LREG1, vConstIntPrgm0/1) as needed
   - `TYPECAST_LLK(0)` -- calls `typecast_tile<IN,OUT>(0)` which:
     - Dispatches to the appropriate `calculate_typecast_*` function
     - Iterates over 8 DEST rows (ITERATIONS=8), applying the type conversion via SFPU instructions
     - Most conversions use SFPLOADMACRO for pipelined high-throughput processing
     - Some conversions (fp32_to_int32, fp32_to_uint32) use explicit SFPLOAD/SFPSTORE for full algorithmic control
   - `pack_tile(0, output_cb)` -- packer reads DEST register 0 and writes to output CB in the output data format

3. **Synchronization**: `tile_regs_acquire/commit/wait/release` coordinate access to DEST registers between the math unit (SFPU) and the packer.

4. **CB flow**: Reader pushes chunks to input CB -> Compute reads, converts, pushes to output CB -> Writer pops from output CB and writes to DRAM.

#### SFPU Configuration

- **Math fidelity**: `HiFi4` (maximum precision, appropriate for exact type conversion)
- **Math approximation mode**: `false` (always `APPROX=false` for typecast)
- **FP32 dest accumulator**: Enabled when source or target is a 32-bit type
- **UnpackToDestMode**: `UnpackToDestFp32` when `preserve_fp32_precision` is true (maintains full FP32 in DEST)
- **BFP8 pack precise**: Optional precise rounding mode for BFP8 output
- **SFPLOADMACRO macros**: Each conversion direction defines 1-3 macros via SFPCONFIG, encoding which pipeline stages (Simple, MAD, Round, Store) are active and which instruction templates they use

#### Hardware Compatibility Notes

The Blackhole and Wormhole B0 implementations differ in several ways:

1. **Address modifier registers**: Blackhole uses `ADDR_MOD_6` / `ADDR_MOD_7` while Wormhole uses `ADDR_MOD_2` / `ADDR_MOD_7`. These control auto-increment behavior for DEST row addressing.

2. **FP32 -> INT32 implementation**: Wormhole uses high-level SFPI constructs (`sfpi::vFloat`, `sfpi::exexp()`, `sfpi::shft()`, `v_if` blocks) while Blackhole uses raw TTI instructions. The algorithm is identical but the code style differs significantly.

3. **Header includes**: Blackhole includes `ckernel_addrmod.h` and `ckernel_ops.h` separately; Wormhole includes `ckernel.h` which bundles these.

4. **Init functions**: The init functions that program SFPLOADMACRO are identical between architectures (they program the same SFPCONFIG bit patterns).

---

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Description |
|-------|------|-------------|
| 0 | `cb_id_in` | Input circular buffer index (`c_0`) |
| 1 | `full_chunk_size_bytes` | Size of a full chunk in input element bytes |
| 2 | `full_chunks_per_row` | Number of full-sized chunks per row |
| 3 | `partial_chunk_size_bytes` | Size of the remainder chunk (0 if evenly divisible) |
| 4 | `partial_chunks_per_row` | 0 or 1: whether a partial chunk exists |
| 5 | `row_page_size_bytes` | Buffer page size for TensorAccessor |
| 6+ | TensorAccessorArgs | Appended by `TensorAccessorArgs(*src_buffer)` |

### Writer Kernel

| Index | Name | Description |
|-------|------|-------------|
| 0 | `cb_id_out` | Output circular buffer index (`c_2`) |
| 1 | `full_chunk_size_bytes` | Size of a full chunk in output element bytes |
| 2 | `full_chunks_per_row` | Number of full-sized chunks per row |
| 3 | `partial_chunk_size_bytes` | Size of the remainder chunk in output bytes |
| 4 | `partial_chunks_per_row` | 0 or 1 |
| 5 | `row_page_size_bytes` | Buffer page size for TensorAccessor |
| 6+ | TensorAccessorArgs | Appended by `TensorAccessorArgs(*dst_buffer)` |

### Compute Kernel

| Index | Name | Description |
|-------|------|-------------|
| 0 | `per_core_block_cnt` | Total chunks per core: `num_rows * (full_chunks + partial_chunks)` |
| 1 | `per_core_block_dim` | Always 1 (one chunk processed at a time) |
| 2 | `input_cb` | Input CB index (`c_0`) |
| 3 | `output_cb` | Output CB index (`c_2`) |

---

## Runtime Arguments

### Reader Kernel

| Index | Name | Description |
|-------|------|-------------|
| 0 | `src_addr` | Source buffer DRAM address |
| 1 | `num_rows` | Number of rows assigned to this core |
| 2 | `start_row_id` | First row index for this core |

### Writer Kernel

| Index | Name | Description |
|-------|------|-------------|
| 0 | `dst_addr` | Destination buffer DRAM address |
| 1 | `num_rows` | Number of rows assigned to this core |
| 2 | `start_row_id` | First row index for this core |

---

## Program Caching and Runtime Override

The `override_runtime_arguments` method updates buffer addresses and row distributions when the cached program is reused with different tensors. It recalculates `num_rows_per_core` and redistributes rows across the same core count. The chunk configuration (sizes, counts) is baked into compile-time args and does not change between reuses -- only the buffer addresses and row assignments are updated.

The program hash includes the full padded shape (not just volume) for ROW_MAJOR layout, ensuring that tensors with different row widths get different cached programs.

---

## Validation Constraints

From `validate_on_program_cache_miss`:
- Input must be on device with a valid buffer
- For ROW_MAJOR layout: `padded_shape[-1] % 32 == 0` (last dimension must be 32-aligned)
- No `sub_core_grids` allowed with ROW_MAJOR
- Input and output must have the same memory layout (both INTERLEAVED for non-sharded)
- Preallocated output shape must match computed output shape
- Zero-volume tensors skip launch entirely

---

## Data Flow Summary

```
DRAM (input, dtype A)
  |
  v
[Reader Kernel] -- NoC read, chunk by chunk, into input CB (c_0)
  |
  v
[Input CB c_0] -- double-buffered, page_size = input_full_chunk_size_bytes
  |
  v
[Compute Kernel] -- copy_tile -> SFPU typecast -> pack_tile
  |    Unpacker: converts CB format -> DEST format
  |    SFPU: typecast_tile<A,B> applies type conversion in DEST registers
  |    Packer: converts DEST format -> output CB format
  |
  v
[Output CB c_2] -- double-buffered, page_size = output_full_chunk_size_bytes
  |
  v
[Writer Kernel] -- NoC write, chunk by chunk, to DRAM
  |
  v
DRAM (output, dtype B)
```

---

## File Inventory

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/copy/typecast/typecast.hpp` | Public TTNN API declaration |
| `ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp` | Public API implementation |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op.hpp` | Device operation struct |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op.cpp` | Factory selection, validation, output spec |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op_types.hpp` | `TypecastParams` and `TypecastInputs` |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_rm_chunked_program_factory.hpp` | Factory header |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_rm_chunked_program_factory.cpp` | **Factory implementation** (analyzed) |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_typecast_rm_chunked.cpp` | Reader kernel |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_typecast_rm_chunked.cpp` | Writer kernel |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp` | Compute kernel |
| `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h` | Compute API: `typecast_tile`, `typecast_tile_init` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h` | LLK dispatch layer |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h` | Architecture wrapper |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h` | **Core SFPU kernel implementations** |

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Typecast operation architecture, program factory selection, LLK API usage patterns
- `tenstorrent/tt-llk`: ckernel::sfpu namespace typecast functions, SFPLOADMACRO usage, template parameter conventions

### Confluence References
Not consulted for this analysis. The SFPU instruction details were sufficiently documented in the source code comments and DeepWiki.

### Glean References
Not consulted for this analysis.

---

## Design Rationale

### Why Chunking?

Row-major tensors can have arbitrarily wide rows (e.g., a 1x65536 tensor has rows of 65536 elements). Allocating a CB page large enough for an entire row would waste L1 memory and could exceed the 1.5MB SRAM budget. Chunking to 1024 elements keeps CB pages small (2-4KB per page depending on element size) while maintaining double-buffering efficiency.

### Why Not Tiling?

This factory handles ROW_MAJOR layout specifically because the standard typecast factory (`TypecastProgramFactory`) operates on TILE layout. Row-major data cannot be processed with tile-based APIs without first tilizing, which would add overhead. The row-major chunked approach processes data in its native layout.

### Why SFPLOADMACRO?

Most typecast conversions use `SFPLOADMACRO` rather than explicit `SFPLOAD`/`SFPSTORE` sequences. SFPLOADMACRO is a hardware instruction that orchestrates a pipelined sequence (Load -> Simple -> MAD -> Round -> Store) across multiple DEST rows, achieving throughputs of 1-4 cycles per row instead of the 5+ cycles needed with explicit instructions. The tradeoff is complexity in the init function (programming macro configurations via SFPCONFIG), but the performance gain is significant for element-wise operations.

### Why Are Some Conversions Handled Without SFPU?

Conversions between floating-point formats of different precision (e.g., Bfp8_b <-> Float16_b, Float16_b -> Float32) are handled entirely by the unpacker/packer hardware. The unpacker naturally converts input formats to the DEST format on load, and the packer converts from DEST format to the output format on store. No SFPU intervention is needed because these are "format conversions" rather than "type conversions" -- the data remains in a floating-point representation throughout.
