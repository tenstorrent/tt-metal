# SFPU Operation Analysis: ADD (binary_ng)

## Operation Overview

**Operation**: Element-wise addition (`BinaryOpType::ADD`)
**Variant**: `binary_ng` (Next Generation binary operation framework)
**Namespace**: `ttnn::operations::binary_ng`
**Program Factory**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

The ADD operation in the binary_ng framework computes `c = a + b` element-wise on tiled tensors. When both input tensors have matching data types of FLOAT32, INT32, UINT32, or UINT16, the operation is routed to the SFPU (Special Function Processing Unit) path. Otherwise, it uses the FPU path which leverages the hardware matrix engine's native add capability. The SFPU path is necessary because the FPU only supports BFLOAT16 natively for binary operations, while the SFPU can operate on FP32 and integer types directly.

### When is SFPU Used for ADD?

The `is_binary_sfpu_op` function (in `binary_ng_device_operation.cpp`) determines SFPU routing:

```
ADD: a == b && (a == FLOAT32 || a == INT32 || a == UINT32 || a == UINT16)
```

So ADD uses the SFPU when both inputs share the same dtype AND that dtype is one of {FLOAT32, INT32, UINT32, UINT16}. For BFLOAT16 inputs, the FPU path is used instead because the FPU natively supports BF16 add.

---

## Device Operation Structure

### File Organization

| File | Purpose |
|------|---------|
| `binary_ng_device_operation.hpp` | Operation attributes, tensor args, ProgramFactory declaration |
| `binary_ng_device_operation.cpp` | `is_binary_sfpu_op`, validation, output spec computation |
| `binary_ng_program_factory.cpp` | Program creation: CB setup, kernel registration, runtime args |
| `binary_ng_utils.hpp` | `OpConfig`, `KernelName` enum, `BinaryNgKernelConfig` |
| `binary_ng_utils.cpp` | `get_kernel_file_path`, `OpConfig` constructor, `get_sfpu_init_fn` |

### Operation Attributes

```cpp
struct operation_attributes_t {
    BinaryOpType binary_op_type;          // ADD
    SmallVector<EltwiseUnaryWithParam> lhs_activations;   // optional pre-processing on LHS
    SmallVector<EltwiseUnaryWithParam> rhs_activations;   // optional pre-processing on RHS
    SmallVector<EltwiseUnaryWithParam> post_activations;  // optional post-processing
    std::optional<ScalarVariant> scalar;   // for tensor-scalar operations
    MemoryConfig memory_config;
    DataType input_dtype;
    std::optional<DataType> dtype;
    CoreRangeSet worker_grid;
    SubtileBroadcastType subtile_broadcast_type;  // broadcast pattern
    bool is_sfpu;       // true when SFPU path selected
    bool is_quant_op;   // false for ADD
    bool is_where_op;   // false for ADD
};
```

### Cached Program Variables

```cpp
struct shared_variables_t {
    KernelHandle reader_kernel_id;
    KernelHandle writer_kernel_id;
    KernelHandle compute_kernel_id;
    CBHandle cb_src_a;
    CBHandle cb_src_b;
    CBHandle cb_src_c;
};
```

---

## OpConfig: Mapping ADD to SFPU Defines

When `is_sfpu` is true, the program factory constructs:

```cpp
const auto op_config = OpConfig(op_type, std::in_place_type<OpConfig::SfpuBinaryOp>, a_dtype);
```

The `OpConfig` constructor maps `BinaryOpType::ADD` to `SfpuBinaryOp::ADD` with no pre-processing or post-processing activations.

The `as_defines` method then calls `get_sfpu_init_fn(SfpuBinaryOp::ADD, dtype)` which returns:

| Data Type | `BINARY_SFPU_INIT` | `BINARY_SFPU_OP` |
|-----------|--------------------|-------------------|
| FLOAT32 | `add_binary_tile_init();` | `add_binary_tile` |
| INT32 | `add_int_tile_init();` | `add_int_tile<DataFormat::Int32>` |
| UINT32 | `add_int_tile_init();` | `add_int_tile<DataFormat::UInt32>` |
| UINT16 | `add_int_tile_init();` | `add_int_tile<DataFormat::UInt16>` |

These defines are injected into the compute kernel at compile time via the `ComputeConfig::defines` map.

---

## Circular Buffer Configuration

| CB Index | Name | Purpose | Tile Count | Data Format | Sharding |
|----------|------|---------|------------|-------------|----------|
| `c_0` | `cb_src_a` | Input tensor A | `a_num_tiles_per_shard` or 2 | `a_data_format` | Backed by A buffer if sharded |
| `c_1` | `cb_src_b` | Input tensor B (or scalar) | `b_num_tiles_per_shard` or 2 (1 for scalar) | `b_data_format` | Backed by B buffer if sharded |
| `c_2` | `cb_src_c` | Output tensor C | `c_num_tiles_per_shard` or 2 | `c_data_format` | Backed by C buffer if sharded |
| `c_3` | intermediate LHS | LHS after pre-activation | 1 | Same as A (for SFPU) | Never sharded |
| `c_4` | intermediate RHS | RHS after pre-activation | 1 | Same as B (for SFPU) | Never sharded |

CBs c_3 and c_4 are only created when `lhs_activations` or `rhs_activations` are non-empty, respectively. For a plain ADD with no activations, only c_0, c_1, c_2 are used.

### UnpackToDestMode

For SFPU operations (except POWER), all source CBs use `UnpackToDestMode::UnpackToDestFp32`. This ensures data is unpacked directly into the FP32 destination register file, bypassing the unpack-to-SRCA/SRCB path. This is critical because the SFPU reads operands from the destination registers, not from the source register banks.

### FP32 Dest Accumulation

`fp32_dest_acc_en` is set to `true` when:
- Output format is UInt32, Int32, or Float32
- Both input formats are Float32, Int32, or UInt32

For ADD with FLOAT32 inputs and FLOAT32 output, this is always true.

---

## Kernel Selection

### Kernel Variants by Broadcast Type

The `BinaryNgKernelConfig` selects kernels based on `SubtileBroadcastType`:

| SubtileBroadcastType | Compute Kernel | SFPU File |
|---------------------|----------------|-----------|
| `NONE` | `ComputeNoBcast` | `eltwise_binary_sfpu_no_bcast.cpp` |
| `SCALAR_A`, `COL_A`, `COL_B`, `ROW_B_COL_A`, `ROW_A_COL_B` | `ComputeBcast` | `eltwise_binary_sfpu.cpp` |
| `ROW_A`, `ROW_B` | `ComputeNoBcast` (with row bcast overwrite to `ComputeRowBcastNg`) | `eltwise_binary_sfpu_row_bcast.cpp` |
| `ROW_A_COL_B`, `ROW_B_COL_A` | (overwritten to `ComputeRowColBcastNg`) | `eltwise_binary_sfpu_row_col_bcast.cpp` |
| scalar (no tensor B) | `ComputeScalar` | `eltwise_binary_sfpu_scalar.cpp` |

### Reader Kernels

When tensor B is present, the reader is selected based on broadcast type:
- `NONE` -> `reader_interleaved_no_bcast.cpp` (kernels_ng)
- `ROW_A/ROW_B` -> `reader_interleaved_row_bcast.cpp` (kernels_ng)
- `COL_A/COL_B` -> `reader_interleaved_col_bcast.cpp` (kernels_ng)
- `SCALAR_A/SCALAR_B` -> `reader_interleaved_scalar_bcast.cpp` (kernels_ng)
- `ROW_B_COL_A/ROW_A_COL_B` -> `reader_interleaved_row_col_mixed_bcast.cpp` (kernels_ng)

### Writer Kernels

- Two-tensor: `writer_interleaved_no_bcast.cpp` (kernels_ng)
- Scalar: `writer_interleaved_scalar.cpp` (kernels)

---

## Kernel Implementations

### Compute Kernel

This section provides the primary compute kernel for the no-broadcast case, which is the most common path for ADD.

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// SFPU split includes provide access to individual SFPU op headers
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
// Provides unary_op_init_common and other unary utilities used for setup
#include "api/compute/eltwise_unary/eltwise_unary.h"

// Core SFPU binary operations: add_binary_tile, sub_binary_tile, mul_binary_tile, etc.
#include "api/compute/eltwise_binary_sfpu.h"
// Bitwise SFPU operations (AND, OR, XOR)
#include "api/compute/binary_bitwise_sfpu.h"
// Shift operations (left shift, right shift)
#include "api/compute/binary_shift.h"
// Integer addition SFPU
#include "api/compute/add_int_sfpu.h"
// Integer subtraction SFPU
#include "api/compute/sub_int_sfpu.h"
// Integer multiplication SFPU
#include "api/compute/mul_int_sfpu.h"
// Integer floor division
#include "api/compute/div_int32_floor.h"
// Integer division
#include "api/compute/div_int32_sfpu.h"
// Integer remainder
#include "api/compute/remainder_int32.h"
// Floating-point fmod
#include "api/compute/binary_fmod.h"
// Quantization operations
#include "api/compute/quantization.h"
// Binary max/min
#include "api/compute/binary_max_min.h"
// GCD
#include "api/compute/gcd.h"
// LCM
#include "api/compute/lcm.h"
// x * log(y)
#include "api/compute/xlogy.h"
// Binary comparison operations
#include "api/compute/binary_comp.h"

// Common macro infrastructure for activation preprocessing and compile-time branching
#include "eltwise_utils_common.hpp"
// SFPU-specific PREPROCESS macro that handles pack_reconfig for intermediate CBs
#include "eltwise_utils_sfpu.hpp"

void kernel_main() {
    // Runtime argument 0: total number of tiles this core must process
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Compile-time argument 0: how many output tiles to produce per read-compute-write cycle
    // For binary_ng this is always 1
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    // CB indices: c_0 = input A, c_1 = input B, c_2 = output
    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // If LHS/RHS activations exist, the preprocessed data goes through intermediate CBs (c_3/c_4)
    // Otherwise the post-CB aliases directly to the pre-CB (no intermediate step needed)
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    // Initialize unary op common state -- sets up unpacker and packer configurations
    // This is required even for binary SFPU ops because copy_tile uses the unary unpack path
    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    // If RELU post-activation is configured at compile time, enable it in the packer hardware
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // Initialize the SFPU binary operation once at the start if there are no activations
    // For ADD: expands to add_binary_tile_init() which calls
    // llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::ADD>()
    // For ADD, init is a no-op (no reciprocal or log tables needed)
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS(LHS, ...) is a no-op for plain ADD (no LHS activations)
        // If activations existed, it would: wait for input, copy to dest, apply activation,
        // pack to intermediate CB
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        // Wait for LHS tile to be available in the CB (written by reader kernel)
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);

        // Same for RHS
        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);

        // Reserve space in the output CB for the result tile
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Re-initialize SFPU if activations required re-purposing the math engine
#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        // Acquire destination registers -- blocks until DEST is available for writing
        // This is the double-buffered handshake between math and pack pipelines
        tile_regs_acquire();

        // Configure unpacker for LHS data format, then copy LHS tile from CB to DEST[0]
        // copy_tile_to_dst_init_short_with_dt reconfigures the unpacker's source data format
        // The second arg (cb_post_lhs) is the "old" format hint for reconfiguration
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // Copy tile from cb_post_lhs position i into DEST register at index i*2
            // Even indices (0, 2, 4...) hold LHS operands
            copy_tile(cb_post_lhs, i, i * 2);
        }

        // Reconfigure unpacker for RHS data format, then copy RHS tile to DEST[1]
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // Copy tile from cb_post_rhs position i into DEST register at index i*2+1
            // Odd indices (1, 3, 5...) hold RHS operands
            copy_tile(cb_post_rhs, i, i * 2 + 1);

            // Re-init SFPU per-tile if post-activations exist (they change math state)
#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT
#endif
            // Execute the SFPU binary ADD operation:
            //   BINARY_SFPU_OP(idst0=i*2, idst1=i*2+1, odst=i*2)
            // For ADD this expands to: add_binary_tile(i*2, i*2+1, i*2)
            // Result overwrites DEST[i*2] (the LHS slot), freeing DEST[i*2+1]
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);

            // Apply post-activations (e.g., RELU, GELU) -- empty for plain ADD
            PROCESS_POST_ACTIVATIONS(i * 2);
        }

        // Signal that math is done writing to DEST -- allows packer to start reading
        tile_regs_commit();

        // Wait for packer to be ready to consume DEST tiles
        tile_regs_wait();

        // Pack result tile from DEST[i*2] into the output CB
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);
        }

        // Release DEST registers -- allows math to write again (double-buffer swap)
        tile_regs_release();

        // Signal output tile is ready for the writer kernel
        cb_push_back(cb_out, num_tiles_per_cycle);
        // Signal we are done consuming the input tiles
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);
    }
}
```

### Compute Kernel Variants

#### Broadcast Variant (`eltwise_binary_sfpu.cpp`)

Used for SCALAR_A, SCALAR_B, COL_A, COL_B broadcast types. The key difference from the no-bcast variant is that it handles the broadcast operand being reused across multiple iterations. The broadcast operand is loaded once and kept in its CB while the non-broadcast operand cycles through tiles.

The `process_tile` function processes a set of tiles with the broadcast operand loaded once:
- The broadcast CB is waited-on once at the start
- An inner loop iterates over the non-broadcast operand tiles
- After all iterations, the broadcast CB is popped

Runtime arguments include `tile_freq` (how many tiles share the same broadcast tile) and `tile_start` (starting offset within the current broadcast cycle).

#### Scalar Variant (`eltwise_binary_sfpu_scalar.cpp`)

Used when one operand is a scalar (no tensor B). The scalar is filled into a tile by the writer kernel and loaded into CB c_1 exactly once. The compute kernel waits for the scalar tile once, then loops over all LHS tiles, applying the SFPU binary op against the same scalar tile.

### Reader Kernel

#### Reader Kernel File (no-broadcast, two-tensor, kernels_ng)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`

The reader kernel reads tiles from both input tensors A and B using `TensorAccessor` for address computation. It supports up to 6+ dimensional tensors by collapsing higher dimensions into an `nD` stride. Key runtime arguments:

| Index | Argument | Purpose |
|-------|----------|---------|
| 0 | `src_addr` | Base address of tensor A in DRAM |
| 1 | `start_tile_id` | Starting output tile index for this core |
| 2 | `src_num_tiles` | Number of A tiles (for sharded path) |
| 3 | `dst_num_tiles` | Number of output tiles for this core |
| 4 | `dst_shard_width` | Width of output shard in tiles |
| 5-8 | strides | nD/D/N/C strides for tensor A broadcasting |
| 9-14 | shape dims | D, N, C, Ht, Wt, cND for output tensor |
| 15 | `src_addr_b` | Base address of tensor B in DRAM |
| 16-19 | strides_b | nD/D/N/C strides for tensor B broadcasting |
| 20 | `src_num_tiles_b` | Number of B tiles (for sharded path) |

The reader uses a 6-level nested loop (nD, D, N, C, Ht, Wt) to iterate over all output tiles, computing the corresponding input tile offsets for both A and B using their respective strides. For sharded tensors, it simply does `cb_reserve_back` / `cb_push_back` to make the pre-existing L1 data available to the compute kernel.

### Writer Kernel

#### Writer Kernel File (no-broadcast, two-tensor, kernels_ng)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`

The writer kernel writes result tiles from CB c_2 to the output tensor. It uses the same 6-level nested loop structure as the reader. For sharded outputs, it is effectively a no-op (the output CB is already backed by the L1 output buffer).

---

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Kernel File
`tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`

(Identical implementation exists for Wormhole at `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`)

### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_exp.h"   // _sfpu_exp_ for POW operation
#include "ckernel_sfpu_log.h"   // _calculate_log_body_ for XLOGY
#include "ckernel_sfpu_recip.h" // _sfpu_reciprocal_ for DIV and POW
#include "sfpi.h"               // SFPI programming interface: vFloat, dst_reg, v_if, etc.

namespace ckernel {
namespace sfpu {

// The core SFPU binary operation template function.
// For ADD, this is instantiated as:
//   _calculate_sfpu_binary_<APPROX, BinaryOp::ADD, 8>(dst_in0, dst_in1, dst_out)
//
// Template parameters:
//   APPROXIMATION_MODE: not used for ADD (no approximation tables needed)
//   BINOP: selects the operation at compile time via constexpr if
//   ITERATIONS: number of face-rows to process per call (8 = one face of 8 groups of 4 rows)
//               The caller (_llk_math_eltwise_binary_sfpu_params_) calls this 4 times
//               for RC mode (4 faces x 8 iterations = 32 groups, covering all 32x32 elements)
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out)
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();

    // Process ITERATIONS groups of rows (each group = 4 rows processed in SIMD across 32 lanes)
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Each tile occupies 32 "rows" in the SFPI view of the destination register.
        // This constant converts a tile index to a base row offset:
        //   tile 0 starts at row 0, tile 1 starts at row 32, etc.
        // The actual hardware Dest register is 64 rows per tile (SFP_DESTREG_STRIDE=2),
        // but SFPI abstracts this to 32 logical rows.
        constexpr std::uint32_t dst_tile_size_sfpi = 32;

        // SFPLOAD: Load 4 rows (current dst_reg position) from tile at dst_index_in0
        // into SFPU local register LReg. The dst_reg[] access compiles to an SFPLOAD
        // instruction that reads from Dest[dst_index_in0 * 32 + current_row_offset].
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];

        // SFPLOAD: Load corresponding 4 rows from tile at dst_index_in1
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = 0.0f;

        // Compile-time dispatch: for ADD, only this branch is compiled
        if constexpr (BINOP == BinaryOp::ADD)
        {
            // SFPADD instruction: lanewise FP32 addition across all 32 lanes
            // vFloat operator+ maps to sfpi_int::fp_add which calls
            // __builtin_rvtt_sfpadd(in0, in1, 0)
            // This produces one SFPADD instruction per iteration
            result = in0 + in1;
        }
        else if constexpr (BINOP == BinaryOp::SUB)
        {
            result = in0 - in1;
        }
        else if constexpr (BINOP == BinaryOp::MUL)
        {
            result = in0 * in1;
        }
        else if constexpr (BINOP == BinaryOp::DIV)
        {
            result = in0 * _sfpu_reciprocal_<2>(in1);
        }
        else if constexpr (BINOP == BinaryOp::RSUB)
        {
            result = in1 - in0;
        }
        else if constexpr (BINOP == BinaryOp::POW)
        {
            result = _calculate_sfpu_binary_power_(in0, in1);
        }
        else if constexpr (BINOP == BinaryOp::XLOGY)
        {
            v_if ((in1 < 0.0f) || (in1 == nan))
            {
                result = nan;
            }
            v_else
            {
                sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = in1;
                _calculate_log_body_<false>(0, dst_index_out);
                result = sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] * in0;
            }
            v_endif;
        }

        // SFPSTORE: Write the result back to the output tile in Dest
        // Compiles to an SFPSTORE instruction that writes 4 rows to
        // Dest[dst_index_out * 32 + current_row_offset]
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;

        // Advance the internal row pointer by 1 group (4 rows)
        // This compiles to TTI_SETRWC to increment the SFPU read/write counter
        sfpi::dst_reg++;
    }
}

// Initialization function -- called once before processing tiles.
// For ADD, this is a no-op (no special tables or configurations needed).
template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        // Only DIV and POW need reciprocal lookup table initialization
        _init_sfpu_reciprocal_<false>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
    // ADD, SUB, MUL, RSUB: no initialization needed
}

} // namespace sfpu
} // namespace ckernel
```

### LLK Dispatch Layer

The compute kernel's `add_binary_tile(idst0, idst1, odst)` call goes through this chain:

```
add_binary_tile(idst0, idst1, odst)                    // api/compute/eltwise_binary_sfpu.h
  -> MATH(llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(idst0, idst1, odst))
    -> _llk_math_eltwise_binary_sfpu_params_<APPROX>(   // llk_math_eltwise_binary_sfpu_params.h
         calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8>,
         idst0, idst1, odst, VectorMode::RC)
      -> _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0)  // configure dest write addr
      -> for face in [0..3]:
           calculate_sfpu_binary<APPROX, ADD, 8>(idst0, idst1, odst)  // 8 iterations per face
           TTI_SETRWC(...)  // advance dest row counter by 16 rows (2 groups of 8)
      -> _llk_math_eltwise_binary_sfpu_done_()           // clear dest addr, wait for completion
```

The params layer (in `llk_math_eltwise_binary_sfpu_params.h`) handles the 32x32 tile's four 16x16 faces:
- **VectorMode::RC** (default): Processes all 4 faces sequentially. Each face calls the SFPU function once with ITERATIONS=8, processing 8 groups of 4 rows = 32 rows per face. Between faces, `TTI_SETRWC` advances the dest pointer by 16 rows (two increments of 8).
- Total: 4 faces x 8 iterations x 4 rows/iteration = 128 row-groups = 1024 elements = 32x32 tile.

### SFPU Instructions Used

| Instruction | Usage in ADD | Description |
|-------------|-------------|-------------|
| `SFPLOAD` | `dst_reg[idx]` read | Loads 4 consecutive rows (32 lanes each) from Dest register into SFPU local register (LReg). Performs data format conversion (e.g., FP16->FP32 or passthrough for FP32). |
| `SFPADD` | `in0 + in1` | Lanewise FP32 addition across all 32 SIMD lanes. Maps to `__builtin_rvtt_sfpadd(a, b, 0)`. No negation modifier needed for ADD (Mod1=0). |
| `SFPSTORE` | `dst_reg[idx] = result` | Stores 4 rows from LReg back to Dest register. Performs format conversion if needed (FP32->FP16_b, etc.). |
| `TTI_SETRWC` | `dst_reg++` and face advancement | Sets the read/write counter for Dest register addressing. Used to advance through rows within a tile and between faces. |

### SFPU Register Usage

| Register Type | Usage | Details |
|---------------|-------|---------|
| **Dest Registers** | Input and output storage | Tiles are unpacked from CBs into Dest before SFPU execution. For ADD with `num_tiles_per_cycle=1`: Dest[0] holds LHS tile, Dest[1] holds RHS tile. Result overwrites Dest[0]. |
| **LReg (Local Registers)** | SFPU computation | 4 LRegs (LReg0-3) available. `SFPLOAD` loads data into LRegs, arithmetic operates on LRegs, `SFPSTORE` writes LRegs back to Dest. For ADD: LReg0 = in0 (from SFPLOAD), LReg1 = in1 (from SFPLOAD), SFPADD writes result to LReg destination. |
| **Dest tile stride** | 32 rows per tile (SFPI view) | Each logical tile occupies 32 SFPI rows. The hardware has 64 physical rows per tile (SFP_DESTREG_STRIDE=2), but SFPI abstracts this. |

### SFPU Execution Flow

1. **Tile Acquisition**: The compute kernel calls `cb_wait_front` on both input CBs (c_0 and c_1) to wait for the reader kernel to supply input tiles.

2. **Dest Register Acquisition**: `tile_regs_acquire()` blocks until the destination register bank is available (double-buffered: while the packer reads one half, math writes the other).

3. **Unpack to Dest**: `copy_tile(cb_post_lhs, 0, 0)` unpacks the LHS tile from CB c_0 into Dest[0]. `copy_tile(cb_post_rhs, 0, 1)` unpacks the RHS tile from CB c_1 into Dest[1]. Because `UnpackToDestMode::UnpackToDestFp32` is set, data is unpacked directly into Dest in FP32 format, bypassing the traditional SRCA/SRCB path.

4. **SFPU Binary Operation**: `add_binary_tile(0, 1, 0)` dispatches to the SFPU. The params layer iterates over 4 faces of the 32x32 tile. Within each face, the `_calculate_sfpu_binary_` function iterates 8 times:
   - **SFPLOAD** from Dest[0 * 32] -> LReg (in0)
   - **SFPLOAD** from Dest[1 * 32] -> LReg (in1)
   - **SFPADD**: result = in0 + in1
   - **SFPSTORE** result -> Dest[0 * 32]
   - **TTI_SETRWC** to advance row pointer

5. **Pack to Output CB**: `tile_regs_commit()` signals math is done. `tile_regs_wait()` waits for the packer. `pack_tile(0, cb_out)` packs Dest[0] (containing the sum) into the output CB c_2. The packer converts from FP32 back to the output data format.

6. **Release**: `tile_regs_release()` frees the Dest bank. `cb_push_back(cb_out, 1)` makes the output tile available to the writer kernel. `cb_pop_front` frees the consumed input tiles.

### SFPU Configuration

| Configuration | Value for ADD | Purpose |
|---------------|---------------|---------|
| `APPROX` | Compile-time constant | Controls approximation mode. For ADD, this has no effect (no approximation tables used). |
| `UnpackToDestFp32` | Enabled for all source CBs | Forces unpacker to write FP32 data directly to Dest. Required because SFPU reads from Dest, not SRCA/SRCB. |
| `fp32_dest_acc_en` | `true` for FP32/INT32 inputs | Configures Dest register file for 32-bit accumulation mode. |
| `BINARY_SFPU_INIT` | `add_binary_tile_init()` | No-op for ADD -- no LUT or special state needed. |
| `BINARY_SFPU_OP` | `add_binary_tile` | Expands to the function pointer that the compute kernel calls with (idst0, idst1, odst). |
| `num_tiles_per_cycle` | 1 | Single tile per read-compute-write cycle. |

### Hardware Compatibility Notes

The `_calculate_sfpu_binary_` function for ADD is **identical** between Wormhole B0 and Blackhole architectures. Both use the same SFPI `vFloat operator+` which maps to the `SFPADD` instruction.

Key architectural differences that do NOT affect ADD but affect other binary ops:
- **Blackhole** supports `SFPMAD_MOD1_NEGATE_VA` and `SFPMAD_MOD1_NEGATE_VC` for operand negation in fused multiply-add, enabling more efficient SUB implementation.
- **Blackhole** has automatic stalling for Dest read-after-write hazards (4-cycle hazard window). Wormhole requires explicit scheduling to avoid the hazard.
- The `float32_to_bf16_rne` function (used by MUL and DIV but NOT by ADD) exists only in the Blackhole `ckernel_sfpu_binary.h` as an arch-local helper, since ADD does not need BF16 rounding.

For ADD specifically, there are no behavioral differences between Wormhole and Blackhole.

---

## Work Distribution

### Interleaved Mode

The program factory uses `split_work_to_cores` to divide output tiles evenly across available cores:

```cpp
std::tie(num_cores, all_cores, core_group_1, core_group_2,
         num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
    split_work_to_cores(compute_with_storage_grid, c_num_tiles, row_major);
```

This produces two core groups: group 1 gets `num_tiles_per_core_group_1` tiles each, group 2 gets one fewer tile. Cores not in either group are assigned zero tiles and skip all computation (their runtime args are zeroed).

### Sharded Mode

When inputs are sharded, each core processes the tiles in its local shard. The `ShardShapeGenerator` computes the actual shard dimensions per core, accounting for uneven sharding on edge cores. The reader/writer kernels operate on local L1 memory instead of issuing NoC reads/writes.

Native L1 sharding is only used when:
- Both inputs have the same shape and memory config
- Neither input is in DRAM
- No uneven sharding on any tensor
- Output grid matches input grids

Otherwise, the operation falls back to the interleaved (TensorAccessor) path.

---

## Runtime Arguments

### Compute Kernel Runtime Args

| Index | Argument | Description |
|-------|----------|-------------|
| 0 | `num_tiles` | Total tiles to process on this core |
| 1 | `freq` | Broadcast frequency (1 for no-broadcast) |
| 2 | `counter` | Starting counter within broadcast cycle (0 for no-broadcast) |
| 3 | `compute_scalar_value` | Quantization zero point (0 for non-quant ADD) |

### Reader Kernel Runtime Args (21 args, no-broadcast two-tensor)

| Index | Argument |
|-------|----------|
| 0 | `a.buffer()->address()` |
| 1 | `c_start_id` |
| 2 | `a_num_tiles` (shard) |
| 3 | `c_num_tiles` |
| 4 | `c_current_shard_width` |
| 5-8 | A strides: nD, D, N, C |
| 9-14 | Output shape: D, N, C, Ht, Wt, cND |
| 15 | `b.buffer()->address()` |
| 16-19 | B strides: nD, D, N, C |
| 20 | `b_num_tiles` (shard) |

### Writer Kernel Runtime Args (11 args, two-tensor)

| Index | Argument |
|-------|----------|
| 0 | `c.buffer()->address()` |
| 1 | `c_start_id` |
| 2 | `c_num_tiles` |
| 3 | `c_current_shard_width` |
| 4-10 | D, N, C, Ht, Wt, cND, 0 |

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: binary_ng operation architecture, kernel file paths, OpConfig class, get_kernel_file_path resolution
- `tenstorrent/tt-llk`: SFPU binary operation dispatch, `_calculate_sfpu_binary_` function, destination register management, `_llk_math_eltwise_binary_sfpu_params_` face iteration
- `tenstorrent/tt-isa-documentation`: SFPLOAD, SFPSTORE, SFPADD instruction details, Dest register addressing, Wormhole vs Blackhole differences
- `tenstorrent/sfpi`: vFloat operator+, dst_reg interface, __builtin_rvtt_sfpadd intrinsic mapping

### Confluence References
Not consulted -- DeepWiki provided sufficient detail for the ADD operation's SFPU instruction usage.

### Glean References
Not consulted -- the ADD operation uses standard SFPADD which is well-documented in open sources.

---

## Key Design Decisions

1. **Why SFPU for FP32 ADD instead of FPU?** The FPU's native binary operations only support BFLOAT16. For FLOAT32 precision, the SFPU path is required because it operates on FP32 values in the Dest register via SFPLOAD/SFPADD/SFPSTORE.

2. **Why UnpackToDestFp32?** The SFPU reads operands from the Dest register file, not from SRCA/SRCB. `UnpackToDestMode::UnpackToDestFp32` instructs the unpacker to bypass the source registers and write directly to Dest in FP32 format.

3. **Why copy_tile instead of unpack?** The `copy_tile` API (via `llk_unpack_A` under the hood) is the mechanism to move tile data from a CB into the Dest register. With UnpackToDestFp32 mode, this becomes a direct load-to-Dest operation.

4. **Why overwrite Dest[0] with the result (odst = idst0)?** This is a register allocation optimization. Since LHS is consumed, its Dest slot can be reused for the output, halving the Dest register pressure. The packer then reads from Dest[0] which now contains the result.

5. **Why 8 iterations x 4 faces?** A 32x32 tile is divided into four 16x16 faces. Each face has 16 rows, but SFPI processes 4 rows per lane-group, so 16/4 = 4 groups per face... except the loop does 8 iterations because the `dst_reg++` increment advances by 1 SFPI row (which is 2 physical rows due to SFP_DESTREG_STRIDE=2), so 8 iterations cover 8 * 2 = 16 physical rows per face.

6. **Why is ADD init a no-op?** Unlike DIV (which needs reciprocal tables) or XLOGY (which needs log tables), ADD uses the simple SFPADD instruction that requires no lookup table initialization.
