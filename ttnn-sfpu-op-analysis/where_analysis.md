# WHERE Operation Analysis

## Operation Overview

The WHERE operation implements element-wise conditional selection: `out = predicate ? value_true : value_false`. It is a ternary SFPU operation that evaluates a predicate tensor and selects corresponding elements from one of two value sources. The operation supports three input combinations (TTT, TTS, TST) with multiple broadcast patterns (NONE, COL_BCAST, ROW_BCAST, OUTER_BCAST, SCALAR_BCAST, SCALAR_A_BCAST, SCALAR_B_BCAST), and it supports interleaved, height-sharded, width-sharded, and block-sharded memory layouts.

**Program Factory**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp`

---

## Operation Attributes

```cpp
struct operation_attributes_t {
    TernaryOpType ternary_op_type;            // WHERE for this operation
    TernaryVariant ternary_variant;           // TTT, TTS, TST, or TSS
    TernaryBroadcastType broadcast_type;      // NONE, OUTER_BCAST, COL_BCAST, ROW_BCAST, SCALAR_BCAST, etc.
    tt::tt_metal::MemoryConfig memory_config;
    DataType input_dtype;
    const CoreRangeSet worker_grid;
    std::optional<DataType> dtype;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config;
    std::optional<CoreRangeSet> sub_core_grids;
    std::optional<float> scalar_input_a;      // For TST variant (true value is scalar)
    std::optional<float> scalar_input_b;      // For TTS variant (false value is scalar)
};
```

### Tensor Args

```cpp
struct tensor_args_t {
    const Tensor& input_tensor_a;             // Predicate tensor (always a tensor)
    std::optional<Tensor> input_tensor_b;     // True value tensor (absent in TST)
    std::optional<Tensor> input_tensor_c;     // False value tensor (absent in TTS)
    std::optional<Tensor> optional_output_tensor;
};
```

### Ternary Variants

| Variant | Predicate | True Value | False Value | Description |
|---------|-----------|------------|-------------|-------------|
| TTT     | Tensor    | Tensor     | Tensor      | All three inputs are tensors |
| TTS     | Tensor    | Tensor     | Scalar      | False value is a scalar |
| TST     | Tensor    | Scalar     | Tensor      | True value is a scalar |
| TSS     | Tensor    | Scalar     | Scalar      | Both values are scalars (not yet supported) |

### Broadcast Types

| Type | Description |
|------|-------------|
| NONE | All tensors have identical shapes |
| OUTER_BCAST | Same H,W dimensions; broadcast in outer dims (-5,-4,-3) |
| COL_BCAST | One or more tensors have W=1, broadcast along width |
| ROW_BCAST | One or more tensors have H=1, broadcast along height |
| SCALAR_BCAST | TTT: one or more tensors have H=1,W=1 (scalar-like) |
| SCALAR_A_BCAST | TTS/TST: predicate tensor is (1,1) |
| SCALAR_B_BCAST | TTS/TST: value tensor is (1,1) |

---

## Circular Buffer Configuration

| CB Index | Name | Role | Data Format | Num Pages |
|----------|------|------|-------------|-----------|
| c_0 | predicate_tensor_cb | Predicate input (always present) | Matches predicate tensor dtype | 2 (or shard volume if sharded) |
| c_1 | value_true/false_tensor_cb | True tensor (TTT/TTS) or False tensor (TST) | Matches respective tensor dtype | 2 (or shard volume if sharded) |
| c_2 | value_false_tensor_cb | False tensor (TTT only) | Matches false tensor dtype | 2 (or shard volume if sharded) |
| c_3 | output_tensor_cb | Output | Matches output dtype | 2 (or shard volume if sharded) |
| c_4 | cb_bcast_a | Row broadcast scratch for predicate (ROW_BCAST TTT only) | Matches predicate dtype | 2 |
| c_5 | cb_bcast_b | Row broadcast scratch for true tensor (ROW_BCAST TTT only) | Matches true tensor dtype | 2 |
| c_6 | cb_bcast_c | Row broadcast scratch for false tensor (ROW_BCAST TTT only) | Matches false tensor dtype | 2 |

### CB Assignment by Variant

- **TTT**: c_0=predicate, c_1=true_value, c_2=false_value, c_3=output. For ROW_BCAST, c_4/c_5/c_6 are additional scratch buffers for LLK unary_bcast results.
- **TTS**: c_0=predicate, c_1=true_value (tensor), c_3=output. False value is a scalar passed as a runtime arg.
- **TST**: c_0=predicate, c_1=false_value (tensor), c_3=output. True value is a scalar passed as a runtime arg.

The design decision to always use c_3 as the output CB (rather than c_2 which is standard for binary operations) allows the TTT variant to use c_2 for the third tensor input without conflict.

---

## Kernel Selection

The program factory uses a hash map (`kernel_config_map`) with composite keys `{TernaryOpType, TernaryVariant, TernaryBroadcastType}` to select the appropriate reader, compute, and writer kernel names. The `get_kernel_file_path()` function maps `KernelName` enums to file paths.

### Kernel Configuration for WHERE

| Variant | Broadcast | Reader Kernel | Compute Kernel | Writer Kernel |
|---------|-----------|---------------|----------------|---------------|
| TTT | NONE | `ternary_reader_nosubtilebcast_ttt.cpp` | `ternary_sfpu_no_bcast_ttt.cpp` | `ternary_writer_nobcast.cpp` |
| TTT | OUTER_BCAST | `ternary_reader_nosubtilebcast_ttt.cpp` | `ternary_sfpu_no_bcast_ttt.cpp` | `ternary_writer_nobcast.cpp` |
| TTT | COL_BCAST | `ternary_reader_colbcast_ttt.cpp` | `ternary_sfpu_col_scalar_bcast_ttt.cpp` | `ternary_writer_nobcast.cpp` |
| TTT | ROW_BCAST | `ternary_reader_rowbcast_ttt.cpp` | `ternary_sfpu_no_bcast_ttt.cpp` (or `ternary_sfpu_row_bcast_ttt.cpp` for bf16) | `ternary_writer_nobcast.cpp` |
| TTT | SCALAR_BCAST | `ternary_reader_scalar_ttt.cpp` | `ternary_sfpu_col_scalar_bcast_ttt.cpp` | `ternary_writer_nobcast.cpp` |
| TTS | NONE | `ternary_reader_nobcast_tst_tts.cpp` | `ternary_sfpu_no_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| TTS | COL_BCAST | `tts_tst_reader_col_bcast.cpp` | `ternary_sfpu_col_scalar_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| TTS | ROW_BCAST | `tts_tst_reader_row_bcast.cpp` | `ternary_sfpu_no_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| TTS | OUTER_BCAST | `tst_tts_reader_outer_bcast.cpp` | `ternary_sfpu_no_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| TTS | SCALAR_A_BCAST | `tst_tts_reader_scalar_bcast.cpp` | `ternary_sfpu_col_scalar_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| TTS | SCALAR_B_BCAST | `tst_tts_reader_scalar_bcast.cpp` | `ternary_sfpu_col_scalar_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| TST | NONE | `ternary_reader_nobcast_tst_tts.cpp` | `ternary_sfpu_no_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| TST | COL_BCAST | `tts_tst_reader_col_bcast.cpp` | `ternary_sfpu_col_scalar_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| TST | ROW_BCAST | `tts_tst_reader_row_bcast.cpp` | `ternary_sfpu_no_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| TST | OUTER_BCAST | `tst_tts_reader_outer_bcast.cpp` | `ternary_sfpu_no_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| TST | SCALAR_A_BCAST | `tst_tts_reader_scalar_bcast.cpp` | `ternary_sfpu_col_scalar_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| TST | SCALAR_B_BCAST | `tst_tts_reader_scalar_bcast.cpp` | `ternary_sfpu_col_scalar_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |

### Special ROW_BCAST Handling for bf16

When all three TTT inputs are BFLOAT16 and the broadcast type is ROW_BCAST, the operation switches to `ComputeRowBcastTTT` (`ternary_sfpu_row_bcast_ttt.cpp`) which uses LLK `unary_bcast<BroadcastType::ROW>` to replicate the single-row tile across the full tile height before performing the SFPU where operation. This is because the SFPU where kernel operates on full tiles and does not natively handle row broadcasting. For non-bf16 types with ROW_BCAST, the reader kernel performs the broadcast by filling tiles.

---

## Compile-Time Defines

### Compute Kernel Defines (set by `get_compute_defines`)

| Define | Value for WHERE |
|--------|----------------|
| `TERNARY_SFPU_OP_INIT` | `where_tile_init` |
| `TERNARY_SFPU_OP_FUNC` | `where_tile<DataFormat::Float16_b>` (bf16), `where_tile<DataFormat::Float32>` (f32), `where_tile<DataFormat::Int32>` (int32) |

### Broadcast Defines (set on compute kernels for bcast variants)

| Define | Meaning |
|--------|---------|
| `BCAST_A` | Predicate tensor is broadcast (single tile reused across iterations) |
| `BCAST_B` | True tensor is broadcast |
| `BCAST_C` | False tensor is broadcast |

### Fill Defines (for TTS/TST scalar handling)

| Define | Meaning |
|--------|---------|
| `FILL_LLK` | `fill_tile` (float), `fill_tile_int<DataFormat::Int32>` (int32), `fill_tile_uint<DataFormat::UInt32>` (uint32) |
| `FILL_WITH_VALUE_FLOAT` | Set to `"1"` for float types |
| `FILL_WITH_VALUE_INT` | Set to `"1"` for int/uint types |

### Compile-Time Args for Compute Kernels

| Index | Value | Description |
|-------|-------|-------------|
| 0 | `num_tiles_per_cycle` (always 1) | Number of output tiles produced per iteration |
| 1 | `scalar_is_true` (0 or 1) | TTS=0, TST=1 (only used by TTS/TST variants) |

### Reader Defines

| Define | Meaning |
|--------|---------|
| `SRC_BCAST_A` | `"1"` if predicate is broadcast in reader |
| `SRC_BCAST_B` | `"1"` if true tensor is broadcast in reader |
| `SRC_BCAST_C` | `"1"` if false tensor is broadcast in reader |
| `SRC_SHARDED_A` | `"1"` if predicate is sharded |
| `SRC_SHARDED_B` | `"1"` if true tensor is sharded |
| `SRC_SHARDED_C` | `"1"` if false tensor is sharded |
| `BCAST_LLK` | `"1"` if row broadcast uses LLK bcast (bf16 only), `"0"` otherwise |
| `FILL_TILE_WITH_FIRST_COLUMN` / `FILL_TILE_WITH_FIRST_ROW` / etc. | Dataflow broadcast tile fill functions, type-dependent |

---

## Runtime Arguments

### Reader Runtime Args (27 args)

| Index | Name | Description |
|-------|------|-------------|
| 0 | `src0_addr` | Predicate tensor buffer address |
| 1 | `src1_addr` | True tensor address (TTT/TTS) or False tensor address (TST) |
| 2 | `src2_addr` | False tensor address (TTT) or 0 (TTS/TST) |
| 3 | `num_tiles` | Number of tiles per core |
| 4 | `start_id` | Starting tile ID for this core |
| 5-8 | `nD_stride, d_stride, n_stride, c_stride` | Predicate tensor strides for broadcast |
| 9-14 | `D, N, C, Ht, Wt, cND` | Output dimensions |
| 15-19 | `b_nD_stride, b_d_stride, b_n_stride, b_c_stride, b_num_tiles` | True/tensor operand strides |
| 20-24 | `c_nD_stride, c_d_stride, c_n_stride, c_c_stride, c_num_tiles` | False/scalar operand strides (0 for scalars) |
| 25 | `dst_shard_width` | Shard width in tiles (sharding only) |
| 26 | `src_num_tiles` | Predicate tile count (sharding only) |

### Writer Runtime Args (11 args)

| Index | Name | Description |
|-------|------|-------------|
| 0 | `dst_addr` | Output buffer address |
| 1 | `num_tiles` | Tiles per core |
| 2 | `start_id` | Starting tile ID |
| 3 | `dst_shard_width` | Shard width in tiles |
| 4-9 | `D, N, C, Ht, Wt, cND` | Output dimensions |
| 10 | padding | Reserved (0) |

### Compute Runtime Args (4 args)

| Index | Name | Description |
|-------|------|-------------|
| 0 | `num_tiles` | Tiles per core |
| 1 | `freq` | Broadcast frequency (Wt for COL_BCAST, Ht*Wt for SCALAR_BCAST, 0 for NONE/OUTER/ROW) |
| 2 | `counter` | Starting tile offset within broadcast cycle |
| 3 | `scalar_arg` | Packed scalar value (for TTS/TST variants) |

---

## Kernel Implementations

### Compute Kernel: `ternary_sfpu_no_bcast_ttt.cpp` (Primary TTT Kernel)

This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_no_bcast_ttt.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
// where.h provides where_tile<DataFormat> and where_tile_init()
#include "api/compute/eltwise_unary/where.h"
// lerp.h included for shared kernel support (LERP also uses this kernel file)
#include "api/compute/eltwise_unary/lerp.h"

void kernel_main() {
    // Runtime arg 0: number of tiles assigned to this core
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Compile-time arg 0: always 1 -- we process one tile per iteration
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    // Circular buffer indices:
    // c_0 = predicate, c_1 = true value, c_2 = false value, c_3 = output
    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_pre_in3 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    // Initialize unary operation -- sets up pack/unpack format for SFPU path
    unary_op_init_common(cb_pre_in1, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Wait for all three input tiles to arrive from the reader
        cb_wait_front(cb_pre_in1, num_tiles_per_cycle);
        cb_wait_front(cb_pre_in2, num_tiles_per_cycle);
        cb_wait_front(cb_pre_in3, num_tiles_per_cycle);

        // Reserve space in output CB
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Acquire exclusive access to destination registers
        tile_regs_acquire();

        // Copy predicate tile into DST register 0
        copy_tile_to_dst_init_short(cb_pre_in1);
        copy_tile(cb_pre_in1, 0, 0);

        // Copy true-value tile into DST register 1
        copy_tile_to_dst_init_short(cb_pre_in2);
        copy_tile(cb_pre_in2, 0, 1);

        // Copy false-value tile into DST register 2
        copy_tile_to_dst_init_short(cb_pre_in3);
        copy_tile(cb_pre_in3, 0, 2);

        // Initialize the SFPU for the ternary operation (where_tile_init)
        TERNARY_SFPU_OP_INIT();
        // Execute: where_tile<DataFormat>(0, 1, 2, 0)
        // Reads predicate from DST[0], true from DST[1], false from DST[2]
        // Writes result to DST[0] (overwrites predicate)
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        // Signal that DST registers are written and ready for pack
        tile_regs_commit();
        tile_regs_wait();

        // Pack DST[0] (result) into the output CB
        pack_tile(0, cb_out);

        // Release DST registers for next iteration
        tile_regs_release();

        // Push output tile and pop all three input tiles
        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in1, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in2, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in3, num_tiles_per_cycle);
    }
}
```

### Compute Kernel: `ternary_sfpu_col_scalar_bcast_ttt.cpp` (TTT Column/Scalar Broadcast)

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_col_scalar_bcast_ttt.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"

// process_tile handles the inner loop for column/scalar broadcast.
// Broadcast CBs are waited/popped outside the inner loop (reused across iterations),
// while non-broadcast CBs are waited/popped inside the loop (one tile per iteration).
ALWI void process_tile(
    tt::CBIndex predicate_cb,
    tt::CBIndex true_cb,
    tt::CBIndex false_cb,
    tt::CBIndex cb_out,
    uint32_t freq,         // broadcast frequency (Wt for COL, Ht*Wt for SCALAR)
    uint32_t tile_start,   // starting offset within broadcast cycle
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

    // Wait for broadcast CBs once (they contain a single tile reused for all iterations in the cycle)
#if BCAST_A
    cb_wait_front(predicate_cb, num_tiles_per_cycle);
#endif
#if BCAST_B
    cb_wait_front(true_cb, num_tiles_per_cycle);
#endif
#if BCAST_C
    cb_wait_front(false_cb, num_tiles_per_cycle);
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Wait for non-broadcast CBs per iteration
#if !BCAST_A
        cb_wait_front(predicate_cb, num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_wait_front(true_cb, num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_wait_front(false_cb, num_tiles_per_cycle);
#endif

        cb_reserve_back(cb_out, num_tiles_per_cycle);
        tile_regs_acquire();

        // Load all 3 inputs to DST registers 0, 1, 2
        copy_tile_init(predicate_cb);
        copy_tile(predicate_cb, 0, 0);
        copy_tile_init(true_cb);
        copy_tile(true_cb, 0, 1);
        copy_tile_init(false_cb);
        copy_tile(false_cb, 0, 2);

        // Execute SFPU ternary operation
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);

        // Pop non-broadcast CBs per iteration
#if !BCAST_A
        cb_pop_front(predicate_cb, num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_pop_front(true_cb, num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_pop_front(false_cb, num_tiles_per_cycle);
#endif
    }

    // Pop broadcast CBs once after all iterations in the cycle
#if BCAST_A
    cb_pop_front(predicate_cb, num_tiles_per_cycle);
#endif
#if BCAST_B
    cb_pop_front(true_cb, num_tiles_per_cycle);
#endif
#if BCAST_C
    cb_pop_front(false_cb, num_tiles_per_cycle);
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);   // broadcast cycle length
    uint32_t tile_start = get_arg_val<uint32_t>(2);   // offset into first cycle

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto predicate_cb = tt::CBIndex::c_0;
    constexpr auto true_cb = tt::CBIndex::c_1;
    constexpr auto false_cb = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(predicate_cb, cb_out);

    // Split work into complete broadcast cycles and a remainder
    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(predicate_cb, true_cb, false_cb, cb_out, tile_freq, tile_start, num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile(predicate_cb, true_cb, false_cb, cb_out, remaining_iterations, tile_start, num_tiles_per_cycle);
    }
}
```

### Compute Kernel: `ternary_sfpu_no_bcast_tts_tst.cpp` (TTS/TST No-Broadcast)

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_no_bcast_tts_tst.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
// fill.h provides fill_tile / fill_tile_int / fill_tile_uint for scalar injection
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    // Runtime arg 3: packed scalar value (float or int bit pattern)
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    // Compile-time arg 1: distinguishes TST (scalar=true, value=1) from TTS (scalar=false, value=0)
    constexpr bool scalar_is_true = get_compile_time_arg_val(1);

    // Only two CBs for TTS/TST: predicate (c_0) and the one tensor operand (c_1)
    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(cb_pre_in1, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_pre_in1, num_tiles_per_cycle);
        cb_wait_front(cb_pre_in2, num_tiles_per_cycle);
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        tile_regs_acquire();

        // Predicate always goes to DST[0]
        copy_tile_to_dst_init_short(cb_pre_in1);
        copy_tile(cb_pre_in1, 0, 0);

        // The tensor operand goes to DST[1] (if TTS, it's the true value)
        // or DST[2] (if TST, it's the false value)
        copy_tile_to_dst_init_short(cb_pre_in2);
        if constexpr (scalar_is_true) {
            // TST: tensor is the false value -> DST[2]
            copy_tile(cb_pre_in2, 0, 2);
        } else {
            // TTS: tensor is the true value -> DST[1]
            copy_tile(cb_pre_in2, 0, 1);
        }

        // Fill the scalar value into the remaining DST register using SFPU fill
        fill_tile_init();
        const auto scalar_val = reinterpret_cast<const float*>(&scalar_value);
        if constexpr (scalar_is_true) {
            // TST: scalar is the true value -> fill DST[1]
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(1, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(1, scalar_value);
#endif
        } else {
            // TTS: scalar is the false value -> fill DST[2]
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(2, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(2, scalar_value);
#endif
        }

        // Execute ternary SFPU with all three values in DST[0], DST[1], DST[2]
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in1, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in2, num_tiles_per_cycle);
    }
}
```

### Compute Kernel: `ternary_sfpu_row_bcast_ttt.cpp` (TTT Row Broadcast via LLK)

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_row_bcast_ttt.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Ternary SFPU compute kernel with optional ROW broadcast via LLK unary_bcast.
// For tensors that need row broadcast (H=1), this kernel first expands them
// using unary_bcast<BroadcastType::ROW> into scratch CBs (c_4/c_5/c_6),
// then performs the ternary SFPU operation on the expanded tiles.

#include <cstdint>

#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    // Pre-CBs: raw inputs from reader
    constexpr auto cb_pre_a = tt::CBIndex::c_0;
    constexpr auto cb_pre_b = tt::CBIndex::c_1;
    constexpr auto cb_pre_c = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    // Post-bcast CBs: expanded tiles after LLK row broadcast
    constexpr auto cb_bcast_a = tt::CBIndex::c_4;
    constexpr auto cb_bcast_b = tt::CBIndex::c_5;
    constexpr auto cb_bcast_c = tt::CBIndex::c_6;

    // Select effective CB based on whether each input needs row broadcast
#if BCAST_A
    constexpr auto cb_eff_a = cb_bcast_a;  // Use broadcast-expanded version
#else
    constexpr auto cb_eff_a = cb_pre_a;    // Use raw input directly
#endif
#if BCAST_B
    constexpr auto cb_eff_b = cb_bcast_b;
#else
    constexpr auto cb_eff_b = cb_pre_b;
#endif
#if BCAST_C
    constexpr auto cb_eff_c = cb_bcast_c;
#else
    constexpr auto cb_eff_c = cb_pre_c;
#endif

    unary_op_init_common(cb_eff_a, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Phase 1: For each broadcast input, expand single-row tile to full tile
        // using LLK unary_bcast<ROW>. This copies the first row across all 32 rows.
#if BCAST_A
        {
            cb_wait_front(cb_pre_a, num_tiles_per_cycle);
            cb_reserve_back(cb_bcast_a, num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_a, cb_bcast_a);
            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_a, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_bcast_a);
            cb_push_back(cb_bcast_a, num_tiles_per_cycle);
            tile_regs_release();
            cb_pop_front(cb_pre_a, num_tiles_per_cycle);
        }
#endif

#if BCAST_B
        {
            cb_wait_front(cb_pre_b, num_tiles_per_cycle);
            cb_reserve_back(cb_bcast_b, num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_b, cb_bcast_b);
            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_b, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_bcast_b);
            cb_push_back(cb_bcast_b, num_tiles_per_cycle);
            tile_regs_release();
            cb_pop_front(cb_pre_b, num_tiles_per_cycle);
        }
#endif

#if BCAST_C
        {
            cb_wait_front(cb_pre_c, num_tiles_per_cycle);
            cb_reserve_back(cb_bcast_c, num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_c, cb_bcast_c);
            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_c, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_bcast_c);
            cb_push_back(cb_bcast_c, num_tiles_per_cycle);
            tile_regs_release();
            cb_pop_front(cb_pre_c, num_tiles_per_cycle);
        }
#endif

        // Phase 2: Execute ternary SFPU on the (possibly broadcast-expanded) tiles
        cb_reserve_back(cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_eff_a, num_tiles_per_cycle);
        cb_wait_front(cb_eff_b, num_tiles_per_cycle);
        cb_wait_front(cb_eff_c, num_tiles_per_cycle);

        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_eff_a);
        copy_tile(cb_eff_a, 0, 0);  // predicate -> DST[0]
        copy_tile_to_dst_init_short(cb_eff_b);
        copy_tile(cb_eff_b, 0, 1);  // true value -> DST[1]
        copy_tile_to_dst_init_short(cb_eff_c);
        copy_tile(cb_eff_c, 0, 2);  // false value -> DST[2]

        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_eff_a, num_tiles_per_cycle);
        cb_pop_front(cb_eff_b, num_tiles_per_cycle);
        cb_pop_front(cb_eff_c, num_tiles_per_cycle);
    }
}
```

### Reader Kernel: `ternary_reader_nosubtilebcast_ttt.cpp`

#### Reader Kernel File
`ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_reader_nosubtilebcast_ttt.cpp`

This reader handles TTT NONE and OUTER_BCAST cases. It uses `TensorAccessor` to read tiles from DRAM for each of the three input tensors (predicate, true, false). For sharded inputs, it simply reserves and pushes the shard into the CB. For interleaved inputs, it performs `noc_async_read_page` for each tile, iterating through all dimensions (ND, D, N, C, Ht, Wt) and applying per-tensor stride offsets for outer broadcast support.

### Writer Kernel: `ternary_writer_nobcast.cpp`

#### Writer Kernel File
`ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_writer_nobcast.cpp`

All WHERE variants use the same writer kernel. For sharded output (`DST_SHARDED=1`), it is a no-op (the output CB is already backed by sharded L1 memory). For interleaved output, it iterates through output tiles, waits for each to appear in the output CB, writes it to DRAM via `noc_async_write_page`, and pops the CB.

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_where.h`
`tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h`

Both Wormhole B0 and Blackhole implementations are identical in structure and logic.

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"
#include "lltt.h"   // lltt::record / lltt::replay for instruction replay buffer
#include "sfpi.h"   // SFPI programming interface

namespace ckernel::sfpu
{

// _calculate_where_: The core SFPU kernel for conditional selection.
// Template parameters:
//   APPROXIMATION_MODE: unused for where (no approximation needed for conditional select)
//   data_format: determines load/store instruction modifier (LO16 for bf16, INT32 for fp32/int)
//   ITERATIONS: number of replay iterations (8 = 8 rows of 32 elements per face = 256 elements)
template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void _calculate_where_(
    const std::uint32_t dst_index_in0,  // predicate tile index in DST
    const std::uint32_t dst_index_in1,  // true-value tile index in DST
    const std::uint32_t dst_index_in2,  // false-value tile index in DST
    const std::uint32_t dst_index_out)  // output tile index in DST
{
    // Static assertion: only Float32, Float16_b, Int32, and UInt32 are supported
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b ||
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32,
        "Unsupported data format for _calculate_where_(). Only Float32, Int32, UInt32, and Float16_b are allowed.");

    // Calculate byte offsets into the DST register file.
    // Each tile index corresponds to 32 rows; <<1 converts to 16-bit word offset.
    int offset0 = (dst_index_in0 * 32) << 1;  // predicate offset
    int offset1 = (dst_index_in1 * 32) << 1;  // true-value offset
    int offset2 = (dst_index_in2 * 32) << 1;  // false-value offset

    // Select load/store modifier: LO16 for bf16 (16-bit), INT32 for 32-bit formats
    constexpr std::uint32_t mod0 = data_format == DataFormat::Float16_b
        ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;

    // --- DISABLE_SFPLOADMACRO path (fallback without macro coissue optimization) ---
#ifdef DISABLE_SFPLOADMACRO
    int offset3 = (dst_index_out * 32) << 1;

    // Record 6 instructions into replay buffer slot 0
    lltt::record(0, 6);
    // Load predicate value from DST[offset0] into LREG0
    TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, offset0);
    // Load true-value from DST[offset1] into LREG1
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset1);
    // Set lane flags: LaneEnabled = (LREG0 == 0), i.e. where predicate is zero
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    // Load false-value from DST[offset2] into LREG1, overwriting true-value
    // ONLY in lanes where LaneEnabled is true (predicate was zero)
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset2);
    // Disable conditional execution (all lanes active again)
    TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1);
    // Store LREG1 (which now has true-value where predicate!=0, false-value where predicate==0)
    TT_SFPSTORE(p_sfpu::LREG1, mod0, ADDR_MOD_6, offset3);

    // Replay the 6-instruction sequence ITERATIONS times (once per row of 32 elements)
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        lltt::replay(0, 6);
    }

    // --- SFPLOADMACRO path (optimized with coissue macros) ---
#else
    if (dst_index_out == dst_index_in0)
    {
        // Optimized case: output overwrites the predicate register (saves 1 cycle).
        // Uses macros 0 and 2 to achieve 3 cycles per row of 32 values:
        //
        // Cycle | Load Unit               | Simple Unit                      | Store Unit
        // ------+-------------------------+----------------------------------+---------------------------
        //   1   | SFPLOAD L0=Dst[offset0] |                                  |
        //   2   | SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L0 EQ 0)   |
        //   3   | SFPLOAD L0=Dst[offset2] | SFPENCC (LaneEnabled=true)       |
        //  (4)  | (next iteration SFPLOAD) |                                 | SFPSTORE Dst[offset0]=L0

        lltt::record(0, 3);
        // Macro 0: triggers SFPSETCC on the Simple Unit when this load completes
        TT_SFPLOADMACRO((0 << 2), mod0, ADDR_MOD_3, offset0);
        // Macro 2: triggers SFPENCC on the Simple Unit when this load completes
        TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_3, offset1);
        // Regular load + store via ADDR_MOD_2 (increments dest by 2 for next iteration)
        TT_SFPLOAD(0, mod0, ADDR_MOD_2, offset2);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 3);
        }
    }
    else
    {
        // General case: output goes to a different register than the predicate.
        // Uses macros 1 and 2 to achieve 4 cycles per row of 32 values:
        //
        // Cycle | Load Unit               | Simple Unit                      | Store Unit
        // ------+-------------------------+----------------------------------+---------------------------
        //   1   | SFPLOAD L0=Dst[offset0] |                                  |
        //   2   | SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L0 EQ 0)   |
        //   3   | SFPLOAD L0=Dst[offset2] | SFPENCC (LaneEnabled=true)       |
        //   4   |                         |                                  | SFPSTORE Dst[offset3]=L0
        //  (5)  | (next iteration SFPLOAD) |                                 |

        int offset3 = (dst_index_out * 32) << 1;

        lltt::record(0, 4);
        TT_SFPLOADMACRO((1 << 2), mod0, ADDR_MOD_3, offset0);
        TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_3, offset1);
        TT_SFPLOAD(0, mod0, ADDR_MOD_3, offset2);
        TT_SFPSTORE(0, mod0, ADDR_MOD_2, offset3);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 4);
        }
    }
#endif
}

// _init_where_: Initializes SFPU instruction templates and macros for the where operation.
// This is called once before processing tiles. It programs:
// 1. Instruction templates for SFPSETCC and SFPENCC (used by SFPLOADMACRO coissue)
// 2. Three macros (0, 1, 2) that define which simple/store operations to coissue with loads
// 3. Misc configuration for store address mode and wait-for-elapsed behavior
template <bool APPROXIMATION_MODE>
inline void _init_where_()
{
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]: SFPSETCC with LREG_EQ0 mode
    // This tests if the loaded predicate value equals zero and sets per-lane flags
    TTI_SFPSETCC(0, 0, 12, 6); // SFPSETCC_MOD1_LREG_EQ0

    // InstructionTemplate[1]: SFPENCC to disable conditional execution
    // Restores all lanes to active after the conditional load
    TTI_SFPENCC(0, 0, 13, 0);

    // Macro 0: Used for the dst_index_out == dst_index_in0 optimization case
    // Coissues: simple_bits -> template 4 (SFPSETCC), store_bits -> template 3 (store with ADDR_MOD_2)
    {
        // simple_bits: (0 << 3) | 4 means use InstructionTemplate[0] (SFPSETCC)
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4;
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        // store_bits: (2 << 3) | 3 means use ADDR_MOD_2 for store, template 3
        constexpr std::uint32_t store_bits  = 0x00 | 0x00 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        // SFPCONFIG stores the macro definition; 4+0 = macro slot 0 (full definition)
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1: Used for the general case (dst_index_out != dst_index_in0)
    // Coissues: simple_bits -> template 4 (SFPSETCC), no store in this macro
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4;
        constexpr std::uint32_t mad_bits    = 0;

        // SFPCONFIG with mode=1 means "simple+mad only" definition
        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1);
    }

    // Macro 2: Coissues SFPENCC (InstructionTemplate[1]) on Simple Unit
    {
        // simple_bits: (0 << 3) | 5 means use InstructionTemplate[1] (SFPENCC)
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 5;
        constexpr std::uint32_t mad_bits    = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1);
    }

    // Misc configuration: {UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1} for all macros
    // 0x770 sets bits for all three macros (0, 1, 2)
    TTI_SFPCONFIG(0x770, 8, 1);
#endif
}

} // namespace ckernel::sfpu
```

#### SFPU Instructions Used

| Instruction | Description | Usage in WHERE |
|-------------|-------------|----------------|
| `SFPLOAD` | Load 32 values from DST register file into SFPU local register (LREG) | Loads predicate, true-value, and false-value tiles row by row |
| `SFPSETCC` | Set per-lane condition flags based on a comparison | `SFPSETCC_MOD1_LREG_EQ0`: sets LaneEnabled=true for lanes where predicate==0 |
| `SFPENCC` | Enable/disable per-lane conditional execution | `SFPENCC_MOD1_EU_R1`: disables conditional execution, restoring all lanes to active |
| `SFPSTORE` | Store 32 values from SFPU local register back to DST register file | Writes the selected result (true or false value) to the output tile |
| `SFPLOADMACRO` | Load with coissue -- triggers a macro (simple/mad/store) alongside the load | Overlaps SFPSETCC and SFPENCC with loads for higher throughput |
| `SFPLOADI` | Load immediate value into SFPU register | Used during init to program macro definitions |
| `SFPCONFIG` | Configure SFPU macro slots and misc settings | Programs macros 0, 1, 2 during initialization |
| `SETRWC` | Set read/write counters for DST register address advancement | Advances DST address between faces in the ternary params template |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Predicate value (in DISABLE_SFPLOADMACRO path) |
| **LREG1** | Initially holds true-value; after conditional load, holds the selected result |
| **L0** (SFPLOADMACRO path) | Single register used for all three loads due to coissue scheduling |
| **LaneFlags** | Per-lane boolean flags set by SFPSETCC, used to mask SFPLOAD of false-value |
| **DST[offset0]** | Predicate tile data in destination register file |
| **DST[offset1]** | True-value tile data in destination register file |
| **DST[offset2]** | False-value tile data in destination register file |
| **DST[offset3]** | Output tile location in destination register file (may alias offset0) |
| **ADDR_MOD_7** | Zero increment for all registers (no auto-advance) |
| **ADDR_MOD_6** | Destination increment by 2 (advances to next row pair after store) |
| **ADDR_MOD_3** | Used by SFPLOADMACRO path for load operations |
| **ADDR_MOD_2** | Used by SFPLOADMACRO path for store with auto-advance |

#### SFPU Execution Flow

1. **Initialization** (`_init_where_` called once per tile batch):
   - Programs two instruction templates: `SFPSETCC` (test predicate==0) and `SFPENCC` (disable conditional execution).
   - Programs three SFPLOADMACRO macros that coissue these simple-unit instructions alongside load-unit operations.
   - Configures store address mode and wait-for-elapsed behavior.

2. **Per-face execution** (`_llk_math_eltwise_ternary_sfpu_params_` handles 4 faces):
   - Sets DST write address to face base via `math::set_dst_write_addr`.
   - Stalls until SFPU is ready (`TTI_STALLWAIT`).
   - Calls `_calculate_where_` for each of the 4 faces (16x32 elements each).
   - Between faces, advances DST address by 16 rows using two `TTI_SETRWC` instructions (8 rows each).
   - After all faces, calls `_llk_math_eltwise_ternary_sfpu_done_` to clear DST address and wait for SFPU completion.

3. **Per-row execution** (`_calculate_where_` inner loop, 8 iterations per face):
   - **SFPLOADMACRO optimized path** (default, when `SFPLOADMACRO` is enabled):
     - If output aliases predicate (offset_out == offset_in0): **3 cycles per row**
       - Cycle 1: Load predicate into L0 (macro 0 triggers SFPSETCC on Simple Unit with 1-cycle delay)
       - Cycle 2: Load true-value into L0 (macro 2 triggers SFPENCC on Simple Unit; SFPSETCC sets LaneFlags from previous predicate load)
       - Cycle 3: Load false-value into L0 (SFPENCC disables conditional; false-value load is masked by LaneFlags -- only lanes where predicate==0 get the false-value; Store of L0 to output happens on Store Unit)
     - If output is separate: **4 cycles per row** (extra cycle for store)
   - **Fallback path** (`DISABLE_SFPLOADMACRO`): **6 instructions per row**
     - Explicit SFPLOAD, SFPSETCC, SFPLOAD, SFPENCC, SFPSTORE sequence.

4. **Key insight -- the conditional selection mechanism**:
   - `SFPSETCC` with `LREG_EQ0` mode sets `LaneFlags[lane] = (predicate[lane] == 0)` for each of 32 lanes.
   - The subsequent `SFPLOAD` of the false-value into LREG1 is **predicated**: it only overwrites lanes where `LaneFlags == true` (i.e., where predicate was zero).
   - Lanes where predicate was non-zero retain their true-value from the earlier load.
   - `SFPENCC` then disables conditional execution so the `SFPSTORE` writes all 32 lanes unconditionally.
   - This achieves `out[lane] = (predicate[lane] != 0) ? true_value[lane] : false_value[lane]` in minimal cycles.

#### SFPU Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| `APPROXIMATION_MODE` | Unused (passed but irrelevant for where) | WHERE is exact; no approximation needed |
| `data_format` | `Float16_b`, `Float32`, `Int32`, or `UInt32` | Determines `InstrModLoadStore` (LO16 vs INT32) |
| `ITERATIONS` | 8 | 8 rows per face (each row = 32 elements, 8x32 = 256 elements per face) |
| `VectorMode` | `RC` (default) | Process all 4 faces of the 32x32 tile |
| `ADDR_MOD_7` | `{srca.incr=0, srcb.incr=0, dest.incr=0}` | No auto-increment (manual addressing) |
| `ADDR_MOD_6` | `{srca.incr=0, srcb.incr=0, dest.incr=2}` | Auto-increment dest by 2 after store |
| Instruction Templates | Template[0]=SFPSETCC, Template[1]=SFPENCC | Pre-programmed for macro coissue |
| Macros | 0=SFPSETCC+store, 1=SFPSETCC only, 2=SFPENCC | Three macros for different coissue patterns |

#### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations of `ckernel_sfpu_where.h` are **structurally identical**. The only difference is in how `load_replay_buf` is defined -- Blackhole uses a lambda-based `load_replay_buf` helper whereas Wormhole uses the direct `lltt::record`/`TT_SFPLOADMACRO` sequence. The SFPU instructions (`SFPLOAD`, `SFPSETCC`, `SFPENCC`, `SFPSTORE`, `SFPLOADMACRO`) are available on both architectures.

The address modifiers differ slightly:
- **Wormhole**: Uses `ADDR_MOD_7` (zero increment) and `ADDR_MOD_6` (dest increment by 2) as configured by `eltwise_ternary_sfpu_configure_addrmod<SfpuType::where>()`.
- **Blackhole**: Same address modifier configuration. The `load_replay_buf` wrapper is a Blackhole-specific convenience that internally calls `lltt::record`.

Both architectures support the `SFPLOADMACRO` optimization. The `DISABLE_SFPLOADMACRO` fallback path exists for testing or hardware revisions that may not support the macro coissue feature.

---

## LLK Layer: Ternary SFPU Params Template

### File
`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_ternary_sfpu_params.h`

### Annotated Source

```cpp
template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_params_(
    Callable&& sfpu_func,           // The SFPU calculation function (e.g., _calculate_where_)
    std::uint32_t dst_index_in0,    // Predicate DST index
    std::uint32_t dst_index_in1,    // True-value DST index
    std::uint32_t dst_index_in2,    // False-value DST index
    std::uint32_t dst_index_out,    // Output DST index
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    // Validate all DST indices are within bounds
    LLK_ASSERT((dst_index_in0 < get_dest_max_tiles<...>()), ...);
    LLK_ASSERT((dst_index_in1 < get_dest_max_tiles<...>()), ...);
    LLK_ASSERT((dst_index_in2 < get_dest_max_tiles<...>()), ...);
    LLK_ASSERT((dst_index_out < get_dest_max_tiles<...>()), ...);

    // Set DST write address and stall until SFPU is ready
    _llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(0);

    if (vector_mode == VectorMode::RC)
    {
        // Process all 4 faces of the 32x32 tile
        for (int face = 0; face < 4; face++)
        {
            sfpu_func(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, args...);
            // Advance DST address by 16 rows (two increments of 8)
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }
    // ... (VectorMode::R and VectorMode::C also supported)

    // Clear DST address and wait for SFPU completion
    _llk_math_eltwise_ternary_sfpu_done_();
}
```

The params template processes all 4 faces of a 32x32 tile. Each face is 16x32 elements (16 rows, 32 columns). The `_calculate_where_` function processes 8 iterations per call, with each iteration handling one row of 32 elements. So each face call processes 8 rows, and `SETRWC` advances the DST pointer by 8+8=16 rows to the next face. After 4 faces, the full 32x32 tile (actually 4x16x32 = 2048 elements) is processed.

---

## Work Distribution

### Interleaved Memory

Work is split across cores using `split_work_to_cores`, which divides the total number of output tiles evenly across the available cores. If the division is uneven, `core_group_1` gets `num_tiles_per_core_group_1` tiles and `core_group_2` gets one fewer tile. Cores not in either group receive zero-initialized runtime args and skip execution.

### Sharded Memory

For native L1 sharding (all tensors identically sharded in L1), each core processes its assigned shard. The `ShardShapeGenerator` computes per-core shard shapes accounting for edge cores that may have fewer tiles. Shard specifications are adjusted between tensors using `adjust_to_shape` when broadcast shapes differ.

Native L1 sharding is only supported for TTT variant when:
- All three input tensors have identical logical shapes
- All three input tensors have identical memory configs
- All tensors are in L1 (not DRAM)
- No uneven sharding (tensor volume divides evenly into shard shape)

---

## Program Caching

The program factory supports caching via `override_runtime_arguments`. On cache hits, only runtime arguments (buffer addresses, tile counts, start IDs, strides) are updated -- the kernel binaries, circular buffers, and compile-time configuration remain unchanged. The common runtime args for TensorAccessor are also updated in-place.

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Located ternary operation structure, program factory patterns, kernel file paths
- `tenstorrent/tt-llk`: Understood ckernel ternary SFPU params template, address modifier configuration, face iteration pattern
- `tenstorrent/tt-isa-documentation`: Identified SFPU conditional selection mechanism (SFPSETCC, SFPENCC, SFPLOADMACRO, LaneFlags predication)
- `tenstorrent/sfpi`: Understood SFPI condition code stack, v_if/v_else constructs, and how they map to hardware instructions

### Confluence References
Not consulted for this analysis -- the SFPU instructions used (SFPSETCC, SFPENCC, SFPLOAD, SFPSTORE, SFPLOADMACRO) were sufficiently documented by DeepWiki and the source code annotations.

### Glean References
Not consulted for this analysis -- the Wormhole and Blackhole implementations are identical in the source code, and no confidential hardware specification details were needed beyond what is available in the open-source tt_llk repository.
