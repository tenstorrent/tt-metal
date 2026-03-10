# LERP Operation Analysis

## Overview

**Operation**: LERP (Linear Interpolation)
**Formula**: `out = input + weight * (end - input)`
**Category**: Ternary elementwise SFPU operation
**Program Factory**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp`

LERP computes elementwise linear interpolation between two tensors (`input`/start and `end`) using a third tensor or scalar (`weight`). It shares the ternary program factory infrastructure with WHERE, ADDCMUL, and ADDCDIV operations, differentiated only by compile-time defines that select the SFPU kernel function.

---

## Operation Attributes

Defined in `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_device_operation.hpp`:

```cpp
struct operation_attributes_t {
    TernaryOpType ternary_op_type;        // TernaryOpType::LERP
    TernaryVariant ternary_variant;       // TTT, TTS (no TST for LERP)
    TernaryBroadcastType broadcast_type;  // NONE, COL_BCAST, ROW_BCAST, OUTER_BCAST, SCALAR_BCAST, etc.
    tt::tt_metal::MemoryConfig memory_config;
    DataType input_dtype;
    const CoreRangeSet worker_grid;
    std::optional<DataType> dtype;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config;
    std::optional<CoreRangeSet> sub_core_grids;
    std::optional<ScalarVariant> scalar_input_a;  // used for TST variant scalar
    std::optional<ScalarVariant> scalar_input_b;  // used for TTS variant scalar (weight)
};
```

### Tensor Arguments

```cpp
struct tensor_args_t {
    const Tensor& input_tensor_a;              // input/start tensor (CB0)
    std::optional<Tensor> input_tensor_b;      // end tensor (CB1)
    std::optional<Tensor> input_tensor_c;      // weight tensor (CB2, for TTT)
    std::optional<Tensor> optional_output_tensor;
};
```

For LERP, the ternary framework maps:
- `input_tensor_a` = `input` (start value) -> CB0 (predicate position in ternary framework)
- `input_tensor_b` = `end` value -> CB1 (true value position)
- `input_tensor_c` = `weight` -> CB2 (false value position, for TTT variant)

### Supported Variants

| Variant | Description | Inputs |
|---------|------------|--------|
| TTT | tensor-tensor-tensor | `input` (tensor), `end` (tensor), `weight` (tensor) |
| TTS | tensor-tensor-scalar | `input` (tensor), `end` (tensor), `weight` (scalar float) |

### Supported Broadcast Types

For **TTT** variant:
- `NONE` -- all tensors have identical shapes
- `COL_BCAST` -- one or more tensors have width=1, broadcast along width
- `ROW_BCAST` -- one or more tensors have height=1, broadcast along height
- `OUTER_BCAST` -- same H,W but different batch/channel dims
- `SCALAR_BCAST` -- one or more tensors have (1,1) in H,W dimensions

For **TTS** variant:
- `NONE`, `COL_BCAST`, `ROW_BCAST`, `OUTER_BCAST`
- `SCALAR_A_BCAST` -- input tensor has (1,1) H,W
- `SCALAR_B_BCAST` -- end tensor has (1,1) H,W

### Supported Data Types

- `Float32` (with `fp32_dest_acc_en = true`)
- `Float16_b` / `BFloat16`

**Not supported**: `INT32`, `UINT32` (static_assert in the SFPU kernel enforces Float32 or Float16_b only).

---

## Circular Buffer Configuration

| CB Index | Role | Size (tiles) | Data Format | Notes |
|----------|------|-------------|-------------|-------|
| c_0 | Input/start tensor | 2 (or shard volume) | Input dtype | Sharded when input is sharded |
| c_1 | End tensor | 2 (or shard volume) | End tensor dtype | Sharded when end tensor is sharded |
| c_2 | Weight tensor (TTT only) | 2 (or shard volume) | Weight dtype | Not used for TTS variant |
| c_3 | Output tensor | 2 (or shard volume) | Output dtype | Sharded when output is sharded |
| c_4 | Row-bcast scratch for A | 2 | Input dtype | Only for ROW_BCAST + TTT |
| c_5 | Row-bcast scratch for B | 2 | End dtype | Only for ROW_BCAST + TTT |
| c_6 | Row-bcast scratch for C | 2 | Weight dtype | Only for ROW_BCAST + TTT |

The CBs c_4, c_5, c_6 are only created when `variant == TTT && broadcast_type == ROW_BCAST`. They serve as intermediate buffers for the LLK `unary_bcast<ROW>` operation that replicates a single row across the full tile height.

---

## Kernel Configuration

### Kernel Selection via `kernel_config_map`

The kernel selection is a hash-map lookup keyed by `(TernaryOpType, TernaryVariant, TernaryBroadcastType)`. LERP entries in the map (from `ternary_op_utils.cpp`):

**TTT configurations:**

| Broadcast Type | Reader Kernel | Compute Kernel | Writer Kernel |
|---------------|--------------|---------------|--------------|
| NONE | `ternary_reader_nosubtilebcast_ttt.cpp` | `ternary_sfpu_no_bcast_ttt.cpp` | `ternary_writer_nobcast.cpp` |
| COL_BCAST | `ternary_reader_colbcast_ttt.cpp` | `ternary_sfpu_col_scalar_bcast_ttt.cpp` | `ternary_writer_nobcast.cpp` |
| ROW_BCAST | `ternary_reader_rowbcast_ttt.cpp` | `ternary_sfpu_no_bcast_ttt.cpp` (or `ternary_sfpu_row_bcast_ttt.cpp` for bfloat16-only) | `ternary_writer_nobcast.cpp` |
| OUTER_BCAST | `ternary_reader_nosubtilebcast_ttt.cpp` | `ternary_sfpu_no_bcast_ttt.cpp` | `ternary_writer_nobcast.cpp` |
| SCALAR_BCAST | `ternary_reader_scalar_ttt.cpp` | `ternary_sfpu_col_scalar_bcast_ttt.cpp` | `ternary_writer_nobcast.cpp` |

**TTS configurations:**

| Broadcast Type | Reader Kernel | Compute Kernel | Writer Kernel |
|---------------|--------------|---------------|--------------|
| NONE | `ternary_reader_nobcast_tst_tts.cpp` | `ternary_sfpu_no_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| COL_BCAST | `tts_tst_reader_col_bcast.cpp` | `ternary_sfpu_col_scalar_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| ROW_BCAST | `tts_tst_reader_row_bcast.cpp` | `ternary_sfpu_no_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| OUTER_BCAST | `tst_tts_reader_outer_bcast.cpp` | `ternary_sfpu_no_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| SCALAR_A_BCAST | `tst_tts_reader_scalar_bcast.cpp` | `ternary_sfpu_col_scalar_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |
| SCALAR_B_BCAST | `tst_tts_reader_scalar_bcast.cpp` | `ternary_sfpu_col_scalar_bcast_tts_tst.cpp` | `ternary_writer_nobcast.cpp` |

### Compute Defines for LERP

Set by `get_compute_defines()` in `ternary_op_utils.cpp`:

```cpp
case TernaryOpType::LERP:
    defines["TERNARY_SFPU_OP_INIT"] = "lerp_tile_init";
    defines["TERNARY_SFPU_OP_FUNC"] =
        (dtype == DataType::FLOAT32) ? "lerp_tile<DataFormat::Float32>" : "lerp_tile<DataFormat::Float16_b>";
    break;
```

These macros are expanded at compile time in the compute kernels to dispatch to the appropriate SFPU function.

---

## Kernel Implementations

### Compute Kernel

The LERP operation uses four compute kernel variants depending on the combination of variant and broadcast type. All share the same SFPU dispatch pattern via the `TERNARY_SFPU_OP_INIT()` / `TERNARY_SFPU_OP_FUNC()` macros.

#### Compute Kernel File: `ternary_sfpu_no_bcast_ttt.cpp` (TTT, no broadcast / outer / row broadcast)

```
ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_no_bcast_ttt.cpp
```

#### Annotated Compute Kernel Source (TTT no-bcast)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
// lerp.h provides lerp_tile<DataFormat> and lerp_tile_init() in the ckernel namespace
#include "api/compute/eltwise_unary/lerp.h"

void kernel_main() {
    // Runtime arg 0: number of output tiles this core must process
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Compile-time arg 0: tiles processed per read-compute-write cycle (always 1)
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    // CB indices: c_0=input(start), c_1=end, c_2=weight, c_3=output
    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_pre_in3 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    // Initialize unpack and pack hardware for SFPU-style ternary operation
    unary_op_init_common(cb_pre_in1, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Wait for reader to deliver one tile from each of the 3 input CBs
        cb_wait_front(cb_pre_in1, num_tiles_per_cycle);  // input/start tile ready
        cb_wait_front(cb_pre_in2, num_tiles_per_cycle);  // end tile ready
        cb_wait_front(cb_pre_in3, num_tiles_per_cycle);  // weight tile ready

        // Reserve space in output CB for one tile
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Acquire destination register file -- prevents pack from reading while we write
        tile_regs_acquire();

        // Unpack input/start tile from CB0 into DST register 0
        copy_tile_to_dst_init_short(cb_pre_in1);
        copy_tile(cb_pre_in1, 0, 0);

        // Unpack end tile from CB1 into DST register 1
        copy_tile_to_dst_init_short(cb_pre_in2);
        copy_tile(cb_pre_in2, 0, 1);

        // Unpack weight tile from CB2 into DST register 2
        copy_tile_to_dst_init_short(cb_pre_in3);
        copy_tile(cb_pre_in3, 0, 2);

        // Initialize SFPU for lerp operation (configures SFPU pipeline)
        // Expands to: lerp_tile_init()
        TERNARY_SFPU_OP_INIT();

        // Execute lerp: DST[0] = DST[0] + DST[2] * (DST[1] - DST[0])
        // Args: (idst0=input, idst1=end, idst2=weight, odst=output_dst_reg)
        // Result overwrites DST[0]
        // Expands to: lerp_tile<DataFormat::Float32>(0, 1, 2, 0) or lerp_tile<DataFormat::Float16_b>(0, 1, 2, 0)
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        // Signal that DST registers are written and ready for pack
        tile_regs_commit();
        // Wait for pack to be ready to consume
        tile_regs_wait();

        // Pack DST register 0 (containing lerp result) to output CB
        pack_tile(0, cb_out);

        // Release DST registers for next iteration
        tile_regs_release();

        // Notify writer that one output tile is ready
        cb_push_back(cb_out, num_tiles_per_cycle);
        // Free consumed input tiles from all 3 CBs
        cb_pop_front(cb_pre_in1, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in2, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in3, num_tiles_per_cycle);
    }
}
```

#### Compute Kernel File: `ternary_sfpu_col_scalar_bcast_ttt.cpp` (TTT, column / scalar broadcast)

```
ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_col_scalar_bcast_ttt.cpp
```

#### Annotated Compute Kernel Source (TTT col/scalar bcast)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"

// Process a group of tiles with broadcast-aware CB synchronization.
// Broadcast CBs (BCAST_A/B/C=1) are waited on ONCE before the loop and popped ONCE after,
// while non-broadcast CBs are waited/popped per iteration.
ALWI void process_tile(
    tt::CBIndex predicate_cb,    // CB for input/start (A)
    tt::CBIndex true_cb,         // CB for end (B)
    tt::CBIndex false_cb,        // CB for weight (C)
    tt::CBIndex cb_out,
    uint32_t freq,               // number of tiles before broadcast CB must be refreshed
    uint32_t tile_start,         // starting offset within the frequency group
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

    // Wait for broadcast CBs outside the loop (they persist across multiple output tiles)
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
        // Wait for non-broadcast CBs inside the loop (new tile each iteration)
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

        // Copy all 3 inputs to destination registers
        copy_tile_init(predicate_cb);
        copy_tile(predicate_cb, 0, 0);   // input/start -> DST[0]

        copy_tile_init(true_cb);
        copy_tile(true_cb, 0, 1);        // end -> DST[1]

        copy_tile_init(false_cb);
        copy_tile(false_cb, 0, 2);       // weight -> DST[2]

        // Execute the SFPU lerp operation
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);

        // Pop non-broadcast CBs inside loop
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

    // Pop broadcast CBs outside loop (consumed after all tiles in the group)
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
    uint32_t tile_freq = get_arg_val<uint32_t>(1);     // broadcast refresh frequency (e.g., Wt for COL_BCAST)
    uint32_t tile_start = get_arg_val<uint32_t>(2);     // offset within first frequency group

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto predicate_cb = tt::CBIndex::c_0;
    constexpr auto true_cb = tt::CBIndex::c_1;
    constexpr auto false_cb = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(predicate_cb, cb_out);

    // Divide work into complete frequency groups plus a remainder
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

#### Compute Kernel File: `ternary_sfpu_no_bcast_tts_tst.cpp` (TTS/TST, no broadcast / outer / row broadcast)

```
ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_no_bcast_tts_tst.cpp
```

#### Annotated Compute Kernel Source (TTS/TST no-bcast)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/fill.h"  // provides fill_tile for scalar filling

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    // Runtime arg 3: packed scalar value (for LERP TTS, this is the weight scalar)
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    // Compile-time arg 1: 1 if TST (scalar replaces true/end), 0 if TTS (scalar replaces false/weight)
    constexpr bool scalar_is_true = get_compile_time_arg_val(1);

    // For TTS: cb_pre_in1=input(start), cb_pre_in2=end(tensor), scalar=weight
    // For TST: cb_pre_in1=input(start), cb_pre_in2=false(tensor), scalar=true/end
    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(cb_pre_in1, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_pre_in1, num_tiles_per_cycle);
        cb_wait_front(cb_pre_in2, num_tiles_per_cycle);
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        tile_regs_acquire();

        // Always copy input/start to DST register 0
        copy_tile_to_dst_init_short(cb_pre_in1);
        copy_tile(cb_pre_in1, 0, 0);

        // Copy tensor operand to the correct DST register based on variant
        copy_tile_to_dst_init_short(cb_pre_in2);
        if constexpr (scalar_is_true) {
            // TST: tensor is false/weight value -> DST[2]
            copy_tile(cb_pre_in2, 0, 2);
        } else {
            // TTS: tensor is end value -> DST[1]
            copy_tile(cb_pre_in2, 0, 1);
        }

        // Fill the scalar value into the remaining DST register
        fill_tile_init();
        const auto scalar_val = reinterpret_cast<const float*>(&scalar_value);
        if constexpr (scalar_is_true) {
            // TST: fill scalar into DST[1] (the end/true position)
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(1, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(1, scalar_value);
#endif
        } else {
            // TTS: fill scalar into DST[2] (the weight/false position)
            // For LERP TTS, this means the weight is a scalar broadcast to all elements
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(2, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(2, scalar_value);
#endif
        }

        // Execute SFPU lerp: DST[0] = DST[0] + DST[2] * (DST[1] - DST[0])
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

#### Compute Kernel File: `ternary_sfpu_row_bcast_ttt.cpp` (TTT, row broadcast with LLK unary_bcast)

```
ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_row_bcast_ttt.cpp
```

#### Annotated Compute Kernel Source (TTT row bcast)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Ternary SFPU compute kernel with optional ROW broadcast via LLK unary_bcast.
// Used when all inputs are bfloat16 and one or more inputs need row broadcasting.
// Pre-CBs (c_0/c_1/c_2) hold single-row tiles; bcast CBs (c_4/c_5/c_6) hold
// the fully replicated tiles after unary_bcast<ROW>.

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

    // Pre-CBs for inputs A (input/start), B (end), C (weight) and output
    constexpr auto cb_pre_a = tt::CBIndex::c_0;
    constexpr auto cb_pre_b = tt::CBIndex::c_1;
    constexpr auto cb_pre_c = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    // Intermediate CBs for LLK row-broadcast results
    constexpr auto cb_bcast_a = tt::CBIndex::c_4;
    constexpr auto cb_bcast_b = tt::CBIndex::c_5;
    constexpr auto cb_bcast_c = tt::CBIndex::c_6;

    // Compile-time CB selection: if BCAST_X is set, use the broadcast CB instead of the pre-CB
    // This allows the SFPU kernel code to be agnostic to whether broadcasting happened
#if BCAST_A
    constexpr auto cb_eff_a = cb_bcast_a;
#else
    constexpr auto cb_eff_a = cb_pre_a;
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
        // Phase 1: Perform LLK row broadcast for any input that needs it.
        // unary_bcast<ROW> replicates the single row of each face across all 32 rows.
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

        // Phase 2: Execute the ternary SFPU operation on the effective inputs
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        cb_wait_front(cb_eff_a, num_tiles_per_cycle);
        cb_wait_front(cb_eff_b, num_tiles_per_cycle);
        cb_wait_front(cb_eff_c, num_tiles_per_cycle);

        tile_regs_acquire();

        // Load all three operands into DST registers
        copy_tile_to_dst_init_short(cb_eff_a);
        copy_tile(cb_eff_a, 0, 0);    // input/start -> DST[0]

        copy_tile_to_dst_init_short(cb_eff_b);
        copy_tile(cb_eff_b, 0, 1);    // end -> DST[1]

        copy_tile_to_dst_init_short(cb_eff_c);
        copy_tile(cb_eff_c, 0, 2);    // weight -> DST[2]

        // Execute lerp SFPU
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

### Reader Kernel (TTT no-bcast example)

#### Reader Kernel File

```
ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_reader_nobcast_ttt.cpp
```

#### Annotated Reader Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args: DRAM addresses and tile range
    uint32_t src0_addr = get_arg_val<uint32_t>(0);   // input/start tensor address
    uint32_t src1_addr = get_arg_val<uint32_t>(1);   // end tensor address
    uint32_t src2_addr = get_arg_val<uint32_t>(2);   // weight tensor address
    uint32_t num_tiles = get_arg_val<uint32_t>(3);   // tiles to process on this core
    uint32_t start_id = get_arg_val<uint32_t>(4);    // global starting tile ID

    // Compile-time args: CB indices for the three input tensors
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);  // c_0
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);  // c_1
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(2);  // c_2

    // TensorAccessor setup for DRAM-to-L1 tile reads
    constexpr auto src0_args = TensorAccessorArgs<3, 0>();
    constexpr auto src1_args =
        TensorAccessorArgs<src0_args.next_compile_time_args_offset(), src0_args.next_common_runtime_args_offset()>();
    constexpr auto src2_args =
        TensorAccessorArgs<src1_args.next_compile_time_args_offset(), src1_args.next_common_runtime_args_offset()>();
    const auto s0 = TensorAccessor(src0_args, src0_addr, get_tile_size(cb_id_in0));
    const auto s1 = TensorAccessor(src1_args, src1_addr, get_tile_size(cb_id_in1));
    const auto s2 = TensorAccessor(src2_args, src2_addr, get_tile_size(cb_id_in2));
    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;
    uint32_t l1_write_addr_in2;
    constexpr uint32_t onetile = 1;

    // Read tiles from DRAM into the three input CBs in lockstep
    for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
        // Reserve CB space and issue async NoC reads for all 3 inputs
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_page(tile_id, s0, l1_write_addr_in0);

        cb_reserve_back(cb_id_in1, onetile);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_page(tile_id, s1, l1_write_addr_in1);

        cb_reserve_back(cb_id_in2, onetile);
        l1_write_addr_in2 = get_write_ptr(cb_id_in2);
        noc_async_read_page(tile_id, s2, l1_write_addr_in2);

        // Wait for all 3 async reads to complete
        noc_async_read_barrier();

        // Push tiles to make them visible to the compute kernel
        cb_push_back(cb_id_in0, onetile);
        cb_push_back(cb_id_in1, onetile);
        cb_push_back(cb_id_in2, onetile);
    }
}
```

### Writer Kernel

#### Writer Kernel File

```
ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_writer_nobcast.cpp
```

#### Annotated Writer Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);        // output DRAM address
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(1);   // tiles to write
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);   // global starting tile ID
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(3); // shard width (for width-sharded)
    const uint32_t D = get_arg_val<uint32_t>(4);               // batch dim D
    const uint32_t N = get_arg_val<uint32_t>(5);               // batch dim N
    const uint32_t C = get_arg_val<uint32_t>(6);               // channel dim
    const uint32_t Ht = get_arg_val<uint32_t>(7);              // height in tiles
    const uint32_t Wt = get_arg_val<uint32_t>(8);              // width in tiles
    const uint32_t cND = get_arg_val<uint32_t>(9);             // collapsed ND dims

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1, 0>();

    // DST_SHARDED=0: write tiles from output CB to DRAM via NoC
    // DST_SHARDED=1: output CB is directly backed by sharded L1 memory, no write needed
#if !DST_SHARDED
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    constexpr bool has_sharding = get_compile_time_arg_val(dst_args.next_compile_time_args_offset()) == 1;
    const uint32_t HtWt = Ht * Wt;

    // Decompose start_tile_id into multi-dimensional coordinates for strided writes
    const uint32_t tiles_per_n = C * HtWt;
    const uint32_t tiles_per_d = N * tiles_per_n;
    const uint32_t tiles_per_nd = D * tiles_per_d;
    // ... (multi-dimensional loop to write tiles accounting for width sharding offsets)

    uint32_t num_tiles_written = 0;
    uint32_t dst_tile_offset = start_tile_id;

    for (uint32_t nd = start_nd; nd < cND && num_tiles_written < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_written < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_written < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_written < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_written < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_written < dst_num_tiles;
                             ++tw, ++num_tiles_written) {
                            cb_wait_front(cb_id_out, onetile);
                            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
                            noc_async_write_page(dst_tile_offset + num_tiles_written, s, l1_read_addr);
                            noc_async_write_barrier();
                            cb_pop_front(cb_id_out, onetile);
                        }
                        if constexpr (has_sharding) {
                            dst_tile_offset += (Wt - dst_shard_width);
                        } else {
                            start_tw = 0;
                        }
                    }
                }
            }
        }
    }
#endif
}
```

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File

The SFPU kernel is identical for both Wormhole B0 and Blackhole architectures:

- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lerp.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_lerp.h`

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "sfpi.h"                  // SFPI programming interface: vFloat, dst_reg, etc.
#include "ckernel_sfpu_binary.h"   // shared binary SFPU utilities (float32_to_bf16_rne)

namespace ckernel::sfpu {

// calculate_lerp: Elementwise linear interpolation on the SFPU.
//
// Template parameters:
//   APPROXIMATION_MODE: whether to use approximate math (not used by lerp, passed through for API consistency)
//   is_fp32_dest_acc_en: true if destination accumulator is FP32 (avoids bf16 rounding)
//   data_format: Float32 or Float16_b (static_assert enforces this)
//   ITERATIONS: number of 32-element row groups to process (default 8 = 256 elements = one 32x32 tile face)
//
// DST register layout:
//   Each tile occupies dst_tile_size_sfpi (=32) rows in the SFPU's view of the DST register file.
//   The SFPU processes one row (32 elements) per loop iteration via SIMD vector operations.
//   8 iterations * 32 elements = 256 elements = one 16x16 face; 4 faces per tile = 1024 elements.
//   The outer dispatcher (_llk_math_eltwise_ternary_sfpu_params_) handles the face iteration.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_lerp(
    const uint dst_index_in0,  // DST register index for input/start tensor
    const uint dst_index_in1,  // DST register index for end tensor
    const uint dst_index_in2,  // DST register index for weight tensor
    const uint dst_index_out) {
    // Enforce that only floating-point formats are supported
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_lerp(). Supported data formats are: Float32, Float16_b.");

    // Each tile in DST occupies 32 rows from the SFPU's perspective.
    // This is because the full 64-row DST stride is halved by the SFP_DESTREG_STRIDE factor.
    constexpr uint dst_tile_size_sfpi = 32;

    // lerp formula: out = input + weight * (end - input)
    // This is mathematically equivalent to: out = (1 - weight) * input + weight * end
    // The chosen form minimizes register pressure: only 3 loads, one sub, one mul, one add.

    // Process ITERATIONS rows (each row = 32 elements in SIMD).
    // The pragma requests full unrolling for performance (avoids branch overhead).
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Load one row (32 floats) from each of the 3 input tiles in DST
        // dst_reg[index * 32] reads the current row of the tile at DST register 'index'
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];  // input/start
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];  // end
        sfpi::vFloat in2 = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];  // weight

        // Compute lerp using SFPI vector operations:
        //   (in1 - in0) compiles to sfpadd with negated in0 (subtraction via FP add with sign flip)
        //   in2 * (in1 - in0) compiles to sfpmul
        //   in0 + result compiles to sfpadd
        // The compiler may fuse the multiply + add into sfpmad if profitable.
        sfpi::vFloat result = in0 + in2 * (in1 - in0);

        // If destination accumulator is not FP32, round the result to BFloat16
        // using round-nearest-even to match hardware pack behavior
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }

        // Store the result row back to DST at the output tile's position
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;

        // Advance the DST row pointer to the next row (shared across all tile accesses)
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

#### SFPU Instructions Used

| SFPI Operation | Hardware Instruction | Description |
|---------------|---------------------|-------------|
| `sfpi::vFloat in0 = sfpi::dst_reg[idx]` | `SFPLOADI` / DST register read | Load 32-element vector from DST register file |
| `in1 - in0` | `SFPADD` (with sign flip) | Subtraction via floating-point addition with negated operand |
| `in2 * (in1 - in0)` | `SFPMUL` | 32-wide SIMD floating-point multiplication |
| `in0 + product` | `SFPADD` | 32-wide SIMD floating-point addition |
| `sfpi::dst_reg[idx] = result` | `SFPSTOREI` / DST register write | Store 32-element vector back to DST register file |
| `float32_to_bf16_rne(result)` | `SFPSTOCHRND` or software rounding | Round FP32 to BFloat16 with round-nearest-even |

The compiler may optimize the multiply-then-add sequence into a single `SFPMAD` (fused multiply-add) instruction: `result = in0 + in2 * (in1 - in0)` could become `SFPMAD(in2, diff, in0)` where `diff = in1 - in0`.

#### SFPU Register Usage

- **DST Registers**: 4 tile slots used (indices 0, 1, 2 for inputs; 0 for output, overwriting input/start)
  - `DST[0]` = input/start tensor tile (also receives output)
  - `DST[1]` = end tensor tile
  - `DST[2]` = weight tensor tile
- **SFPU Internal Registers**: Implicit `vFloat` vector registers used as temporaries during the computation. The SFPI programming model abstracts these as C++ local variables.
- **DST Row Pointer** (`dst_reg++`): Auto-incremented each iteration to process the next 32-element row within the tile face.

#### SFPU Execution Flow

1. **Tile Acquisition**: The compute kernel waits on circular buffers c_0, c_1, c_2 for one tile each (`cb_wait_front`).
2. **DST Register Loading**: Three `copy_tile()` calls unpack tiles from the CBs into DST registers 0, 1, 2. The unpack hardware converts from the CB's data format to the DST register format (FP32 or BF16 depending on `fp32_dest_acc_en`).
3. **SFPU Initialization**: `lerp_tile_init()` calls `_llk_math_eltwise_ternary_sfpu_init_<SfpuType::lerp>()` which configures the SFPU pipeline for the lerp operation type.
4. **SFPU Math Execution**: The `_llk_math_eltwise_ternary_sfpu_params_` dispatcher iterates over the 4 faces of a 32x32 tile (each face is 16x16 = 256 elements). For each face, it calls `calculate_lerp` with `ITERATIONS=8`, processing 8 rows of 32 elements = 256 elements per face.
   - For each row: load `in0`, `in1`, `in2` from DST; compute `in0 + in2 * (in1 - in0)`; optionally round to BF16; store result to DST[0].
5. **Result Pack**: `pack_tile(0, cb_out)` packs DST register 0 into the output CB, converting from the DST format back to the output tensor's data format.
6. **CB Cleanup**: Output CB is pushed (`cb_push_back`) to notify the writer; input CBs are popped (`cb_pop_front`) to free space for the next tile.

#### SFPU Configuration

- **`fp32_dest_acc_en`**: Enabled when output dtype is Float32, Int32, or UInt32. Controls whether DST registers hold FP32 or BF16 values, and whether the BF16 rounding step in `calculate_lerp` is skipped.
- **`UnpackToDestMode`**: Set per-CB based on input tensor dtype. When an input is Float32, `UnpackToDestFp32` is used to unpack directly to FP32 DST registers.
- **`APPROX` template parameter**: Passed through but not used by the lerp calculation (no transcendental approximation needed for linear interpolation).
- **`TERNARY_SFPU_OP_INIT` / `TERNARY_SFPU_OP_FUNC` macros**: Set by the program factory via `get_compute_defines()`. For LERP: `lerp_tile_init` and `lerp_tile<DataFormat::Float32>` or `lerp_tile<DataFormat::Float16_b>`.
- **`FILL_LLK` macro**: Used in TTS/TST variants to fill a scalar value into a DST register. Maps to `fill_tile` (float) or `fill_tile_int`/`fill_tile_uint` depending on dtype.

#### Hardware Compatibility Notes

The SFPU lerp kernel (`ckernel_sfpu_lerp.h`) is **identical** for both Wormhole B0 and Blackhole architectures. The source files at both paths contain the same code, meaning:

- The SFPI instruction set used (SFPADD, SFPMUL, DST register access) is compatible across both architectures.
- The `float32_to_bf16_rne` function is available on both platforms.
- The DST register tile size (32 rows per tile in SFPU view) is the same.

The only architectural difference relevant to LERP is in the **row broadcast path**: the `ternary_sfpu_row_bcast_ttt.cpp` kernel is only used when `is_llk_bcast()` returns true, which requires all three inputs to be BFloat16 and `ROW_BCAST` type. This is because the LLK `unary_bcast<ROW>` operation has format-specific hardware support.

---

## LLK API Call Chain

The full dispatch chain from compute kernel to SFPU hardware:

```
lerp_tile<DataFormat>(idst0, idst1, idst2, odst)              // api/compute/eltwise_unary/lerp.h
  -> MATH(llk_math_eltwise_ternary_sfpu_lerp<...>(...))       // guarded: only runs on TRISC_MATH processor
    -> _llk_math_eltwise_ternary_sfpu_params_<APPROX>(        // ternary dispatcher (in tt_llk submodule)
         sfpu::calculate_lerp<...>,                            // function pointer to the SFPU kernel
         dst_index0, dst_index1, dst_index2, odst,
         VectorMode::RC)
      -> [iterates over 4 tile faces]
        -> calculate_lerp<APPROX, fp32, DataFormat, 8>(...)   // ckernel_sfpu_lerp.h
          -> [8 iterations of SFPI vector ops per face]
            -> SFPADD, SFPMUL, SFPADD (or SFPMAD)             // hardware SFPU instructions
```

The `MATH()` macro ensures the lerp computation only executes on the math RISC-V processor (TRISC_MATH), not on the unpack or pack processors that share the same kernel binary.

---

## Program Factory Details

### Work Distribution

The program factory distributes output tiles across cores using `split_work_to_cores()`:

- For **interleaved** tensors: tiles are split evenly across available cores, with two core groups to handle remainder tiles.
- For **sharded** tensors: each core processes its own shard. The `ShardShapeGenerator` class computes per-core shard dimensions, accounting for edge cores that may have smaller shards.

### Runtime Arguments

**Reader Kernel** (27 args per core for broadcast variants):
- Args 0-4: tensor addresses, tile count, start tile ID
- Args 5-14: stride information for multi-dimensional broadcast (nD/D/N/C strides, output dimensions)
- Args 15-24: strides for secondary and tertiary tensors
- Args 25-26: shard width and predicate tile count (for sharding)

**Writer Kernel** (11 args per core):
- Args 0-2: output address, tile count, start tile ID
- Args 3-9: shard width and dimension info (D, N, C, Ht, Wt, ND)

**Compute Kernel** (4 args per core):
- Arg 0: number of tiles to process
- Arg 1: broadcast frequency (e.g., Wt for COL_BCAST; 0 for no-bcast)
- Arg 2: broadcast counter/start offset
- Arg 3: packed scalar value (for TTS/TST variants only)

### Compile-Time Arguments

**Reader Kernel**: CB indices (c_0, c_1, [c_2 for TTT]), TensorAccessor args, sharding flag
**Compute Kernel**: `{num_tiles_per_cycle=1, scalar_is_true_value}` where `scalar_is_true_value` is 1 for TST, 0 for TTS
**Writer Kernel**: output CB index (c_3), TensorAccessor args, sharding flag

### Compile-Time Defines

| Define | Value | Description |
|--------|-------|-------------|
| `TERNARY_SFPU_OP_INIT` | `lerp_tile_init` | SFPU initialization function |
| `TERNARY_SFPU_OP_FUNC` | `lerp_tile<DataFormat::Float32>` or `lerp_tile<DataFormat::Float16_b>` | SFPU execution function |
| `BCAST_A` | `0` or `1` | Whether input/start tensor is broadcast |
| `BCAST_B` | `0` or `1` | Whether end tensor is broadcast |
| `BCAST_C` | `0` or `1` | Whether weight tensor is broadcast |
| `FILL_LLK` | `fill_tile` or `fill_tile_int<...>` | Scalar fill function for TTS/TST |
| `FILL_WITH_VALUE_FLOAT` | `1` | Indicates float scalar fill (for non-int dtypes) |
| `BCAST_LLK` | `0` or `1` | Whether LLK-level row broadcast is used (TTT only) |
| `SRC_BCAST_A/B/C` | `0` or `1` | Reader-side broadcast flags |
| `SRC_SHARDED_A/B/C` | `0` or `1` | Reader-side sharding flags |
| `DST_SHARDED` | `0` or `1` | Writer-side sharding flag |

---

## External Knowledge Sources

### DeepWiki References

- **tenstorrent/tt-metal**: Program factory structure for ternary operations, kernel selection via `kernel_config_map`, LERP compute defines, composite fallback implementation
- **tenstorrent/tt-llk**: Ternary SFPU dispatch mechanism via `_llk_math_eltwise_ternary_sfpu_params_`, ckernel namespace organization
- **tenstorrent/sfpi**: SFPI programming interface -- vFloat type, dst_reg access, SFPADD/SFPMUL instructions, potential SFPMAD fusion for lerp

### Confluence References

Not consulted for this analysis -- the SFPU instructions used by LERP (basic arithmetic: add, multiply, subtract) are well-documented in DeepWiki and the SFPI repo.

### Glean References

Not consulted for this analysis -- no confidential hardware specifications were needed beyond what the open-source codebase provides.

---

## Key Design Decisions

1. **Shared Ternary Infrastructure**: LERP reuses the same program factory, reader kernels, writer kernels, and compute kernel templates as WHERE. The only difference is the compile-time `TERNARY_SFPU_OP_INIT` / `TERNARY_SFPU_OP_FUNC` defines. This design minimizes code duplication and ensures broadcast/sharding support is consistent across all ternary operations.

2. **Output Overwrites Input Register**: The SFPU kernel writes the result to `DST[0]`, the same register that held the input/start value. This is safe because by the time the result is written, `in0` has already been consumed in the computation. It avoids needing a 4th DST register.

3. **BF16 Rounding Guard**: The `if constexpr (!is_fp32_dest_acc_en)` block in `calculate_lerp` explicitly rounds to BF16 after each element. This is necessary because the SFPU operates internally in FP32 even for BF16 data, and without explicit rounding, intermediate precision could leak into the output.

4. **Row Broadcast via Separate Kernel**: Rather than performing row broadcast within the SFPU math loop, the `ternary_sfpu_row_bcast_ttt.cpp` kernel uses the LLK `unary_bcast<ROW>` hardware operation as a pre-processing step. This is because the SFPU's per-row iteration model does not natively support row replication -- the row broadcast must happen at the tile level before the SFPU processes the data.

5. **No INT32/UINT32 Support**: The `static_assert` in `calculate_lerp` restricts LERP to floating-point formats. This is a deliberate design choice because linear interpolation on integer types is mathematically ambiguous (the weight multiplication produces non-integer results).

6. **Scalar Weight via `fill_tile`**: In the TTS variant, the scalar weight is broadcast to all elements of DST register 2 using the `fill_tile` LLK function, which writes the same float value to every element position. This is simpler than using a separate CB for a scalar tensor.
