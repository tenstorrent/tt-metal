# MAXIMUM (binary_ng) -- SFPU Operation Analysis

## Operation Overview

**Operation**: `MAXIMUM` (elementwise binary maximum)
**Variant**: `binary_ng` (next-generation binary framework)
**Mathematical Definition**: `C[i] = max(A[i], B[i])` for each element
**SFPU-Only**: Yes -- MAXIMUM always routes through the SFPU path; there is no FPU fallback. The function `is_binary_sfpu_op()` unconditionally returns `true` for `BinaryOpType::MAXIMUM`.

**Supported Data Types**:
- Float (bfloat16, float32) -- uses `binary_max_tile` / `binary_max_tile_init`
- INT32 -- uses `binary_max_int32_tile` / `binary_max_int32_tile_init`
- UINT32 -- uses `binary_max_uint32_tile` / `binary_max_uint32_tile_init`

**Namespace**: `ttnn::operations::binary_ng`

---

## Program Factory

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

### Factory Structure

The `BinaryNgDeviceOperation::ProgramFactory::create` function creates a complete TT-Metal program with three kernel types (reader, compute, writer) and associated circular buffers. For MAXIMUM, the key decision points are:

1. **SFPU detection**: `operation_attributes.is_sfpu` is `true` (set by `is_binary_sfpu_op()` in the device operation layer).
2. **OpConfig construction**: `OpConfig(op_type, std::in_place_type<OpConfig::SfpuBinaryOp>, a_dtype)` maps `BinaryOpType::MAXIMUM` to `SfpuBinaryOp::MAXIMUM`.
3. **Kernel define generation**: `OpConfig::as_defines()` calls `get_sfpu_init_fn(SfpuBinaryOp::MAXIMUM, dtype)` which produces:
   - `BINARY_SFPU_INIT` = `"binary_max_tile_init();"` (or int32/uint32 variant)
   - `BINARY_SFPU_OP` = `"binary_max_tile"` (or int32/uint32 variant)
4. **Kernel file selection**: `get_kernel_file_path(kernel_name, is_sfpu=true, is_where_op=false)` selects `eltwise_binary_sfpu_*.cpp` kernels.

### Circular Buffers

| CB Index | Name | Purpose | Size (tiles) |
|----------|------|---------|--------------|
| c_0 | cb_src_a | Input tensor A | `a_num_tiles_per_shard` or 2 (double buffer) |
| c_1 | cb_src_b | Input tensor B (or scalar tile) | `b_num_tiles_per_shard` or 2 (1 if scalar) |
| c_2 | cb_out | Output tensor C | `c_num_tiles_per_shard` or 2 (double buffer) |
| c_3 | cb_lhs_intermediate | LHS post-activation intermediate | 1 tile (only if LHS activations present) |
| c_4 | cb_rhs_intermediate | RHS post-activation intermediate | 1 tile (only if RHS activations present) |
| c_5 | cb_row_bcast_a | Row broadcast buffer for A | 2 tiles (only for ROW_A/ROW_A_COL_B broadcast) |
| c_6 | cb_row_bcast_b | Row broadcast buffer for B | 2 tiles (only for ROW_B/ROW_B_COL_A broadcast) |

### SFPU-Specific Configuration

For SFPU operations (including MAXIMUM), the program factory configures:

- **`fp32_dest_acc_en`**: Enabled when output is UInt32/Int32/Float32, or when both inputs are Float32, Int32, or UInt32.
- **`UnpackToDestMode`**: For all SFPU ops except POWER, all source CBs (c_0, c_1, c_3, c_4) use `UnpackToDestFp32`. This is because the SFPU operates on data in the DST register in FP32 format.
- **No pre/post activations by default**: MAXIMUM has no `process_lhs`, `process_rhs`, or `postprocess` in its `OpConfig`, so `PROCESS_LHS_ACTIVATIONS`, `PROCESS_RHS_ACTIVATIONS`, and `PROCESS_POST_ACTIVATIONS` macros expand to empty.

### Broadcast Modes

The `SubtileBroadcastType` determines which compute kernel variant is used:

| Broadcast Type | Compute Kernel | Description |
|----------------|---------------|-------------|
| NONE | `eltwise_binary_sfpu_no_bcast.cpp` | Both tensors have equal tile dimensions |
| SCALAR_A, COL_A, COL_B, ROW_B_COL_A, ROW_A_COL_B | `eltwise_binary_sfpu.cpp` | One operand is broadcast (col or scalar) |
| SCALAR_B (b is scalar value) | `eltwise_binary_sfpu_scalar.cpp` | B is a scalar, filled into a single tile |
| ROW_A, ROW_B | `eltwise_binary_sfpu_row_bcast.cpp` (ng variant) | Row broadcast with LLK bcast support |
| ROW_A_COL_B, ROW_B_COL_A | `eltwise_binary_sfpu_row_col_bcast.cpp` (ng variant) | Mixed row/col broadcast |

### Work Distribution

Work is distributed across cores based on the output tile count:
- For **interleaved** memory: `split_work_to_cores()` divides output tiles evenly, creating two core groups (group 1 with `num_tiles_per_core_group_1` tiles, group 2 with potentially fewer).
- For **sharded** memory: Each core processes its own shard. The `ShardShapeGenerator` handles edge cases where the last core in a dimension may have a smaller shard.
- Runtime arguments include `{c_num_tiles, freq, counter, 0}` for compute, where `freq` and `counter` handle broadcast cycling.

---

## Kernel Implementations

### Reader Kernel

**Two-tensor path** (both A and B are tensors): `kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`

The reader kernel reads tiles from both input tensor A (into CB c_0) and input tensor B (into CB c_1). It iterates through the output tile space using a 6-nested loop over dimensions (ND, D, N, C, Ht, Wt), computing independent tile offsets for A and B using per-dimension strides. This stride-based addressing enables implicit broadcasting: if a dimension has size 1 in A or B, its stride is 0, so the same data is re-read for every iteration of that dimension.

**Scalar path** (B is a scalar): `kernels/dataflow/writer_interleaved_scalar.cpp` handles filling a single tile with the scalar value into CB c_1. The reader kernel `kernels/dataflow/reader_interleaved_no_bcast.cpp` reads only tensor A into CB c_0.

Both kernels support sharded and interleaved memory layouts via compile-time `SRC_SHARDED` / `SRC_SHARDED_B` defines.

### Writer Kernel

**Two-tensor path**: `kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`

The writer kernel reads computed tiles from CB c_2 and writes them to the output buffer using the same dimension-nested loop pattern. For sharded outputs (`DST_SHARDED` defined), the CB is pre-allocated in the shard region and no explicit writes are needed.

**Scalar path**: `kernels/dataflow/writer_interleaved_scalar.cpp` -- doubles as both the scalar fill kernel and the output writer.

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File (no-broadcast variant)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

#### Annotated Compute Kernel Source (no-broadcast)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// SFPU unary split includes for activation support
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

// SFPU binary operation APIs -- each header provides tile-level init + compute functions
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/remainder_int32.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"   // Provides binary_max_tile, binary_max_tile_init, etc.
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"

// Macro infrastructure for preprocessing activations and broadcast handling
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"

void kernel_main() {
    // Runtime argument 0: total number of tiles this core must process
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Compile-time argument 0: number of tiles produced per read-compute-write cycle (always 1)
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    // Circular buffer assignments:
    // c_0 = input A (pre-activation), c_1 = input B (pre-activation), c_2 = output
    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // If LHS/RHS activations are defined (via defines), use intermediate CBs c_3/c_4;
    // otherwise, read directly from c_0/c_1. For MAXIMUM, no activations are defined,
    // so cb_post_lhs == cb_pre_lhs and cb_post_rhs == cb_pre_rhs.
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    // Initialize the unary op common state (sets up pack/unpack config)
    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    // Optional ReLU fusing in the pack stage -- not used for MAXIMUM
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // For MAXIMUM with no pre/post activations, BINARY_SFPU_INIT is called once here.
    // This expands to binary_max_tile_init() which configures:
    //   - SFPU address modifiers for the max/min swap pattern
    //   - SFPLOADMACRO instruction templates and macro sequences
    //   - SFPCONFIG misc register for store behavior
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT  // Expands to: binary_max_tile_init();
#endif

    // Main tile processing loop: one tile per iteration
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS macros expand to nothing for MAXIMUM (no pre-activations)
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        // Wait for the reader kernel to push a tile of A into CB c_0
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        // Wait for the reader kernel to push a tile of B into CB c_1
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);

        // Reserve space in the output CB for one tile
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Re-init SFPU if activations present (not the case for MAXIMUM)
#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        // Acquire DST register bank -- blocks until DST is available
        tile_regs_acquire();

        // Copy tile A from CB c_0 into DST register at index 0 (i*2 = 0)
        // The init_short_with_dt call reconfigures the unpacker for A's data format
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_lhs, i, i * 2);  // A -> DST[0]
        }

        // Copy tile B from CB c_1 into DST register at index 1 (i*2+1 = 1)
        // Reconfigure unpacker for B's data format
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_rhs, i, i * 2 + 1);  // B -> DST[1]

            // Re-init per-tile if post-activations present (not for MAXIMUM)
#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT
#endif
            // Execute the SFPU max operation:
            // BINARY_SFPU_OP(0, 1, 0) expands to binary_max_tile(0, 1, 0)
            // This reads DST[0] (A) and DST[1] (B), computes max, writes to DST[0]
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);

            // Post-activation processing (empty for MAXIMUM)
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        // Signal that DST registers are ready for packing
        tile_regs_commit();

        // Wait for commit to complete, then pack result from DST[0] into output CB
        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);  // DST[0] -> CB c_2
        }
        // Release DST registers for next iteration
        tile_regs_release();

        // Signal to writer that output tile is ready
        cb_push_back(cb_out, num_tiles_per_cycle);
        // Signal to reader that input tiles have been consumed
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);
    }
}
```

#### Compute Kernel File (broadcast variant)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp`

The broadcast variant adds a `process_tile` function that handles the case where one operand is broadcast (e.g., column broadcast). The broadcast operand's CB is held open (waited once, popped once at the end), while the non-broadcast operand cycles through `freq` tiles per broadcast tile. The core SFPU dispatch is identical: `copy_tile` both operands into DST, call `BINARY_SFPU_OP`, pack result.

#### Compute Kernel File (scalar variant)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp`

The scalar variant pre-loads the scalar tile from CB c_1 once (waited at start), then iterates over all tiles of A, applying `BINARY_SFPU_OP(0, 1, 0)` for each. The scalar tile remains resident in CB c_1 and is popped only after all tiles are processed.

---

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Kernel File (Blackhole)
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h`

### SFPU Kernel File (Wormhole B0)
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h`

Both architectures share the same source code (identical implementation). The implementations are shown below.

### Annotated SFPU Kernel Source (Float path -- `calculate_binary_max_min`)

```cpp
// Template parameters:
//   IS_MAX_OP: true for max, false for min (controls which register is stored)
//   ITERATIONS: 8 = number of 4-element rows in a 32x32 tile face (32 rows / 4 faces = 8 rows per face)
template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(
    const uint dst_index_in0,   // DST register index for input A
    const uint dst_index_in1,   // DST register index for input B
    const uint dst_index_out) { // DST register index for output (typically same as in0)

    // Compute byte offsets into the DST register file.
    // Each tile occupies 32 rows, each row is 2 bytes wide (16-bit in DST address space),
    // hence the << 1 shift.
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
    // Fallback path: simple sequential load-swap-store pattern.
    // Throughput: ~4 cycles per row (1 load + 1 load + 1 swap + 1 store).
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load row from input A into LREG0
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0);
        // Load row from input B into LREG1
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1);
        // SFPSWAP with VEC_MIN_MAX mode: after execution,
        //   LREG1 = max(LREG0, LREG1) per element
        //   LREG0 = min(LREG0, LREG1) per element
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        // Store the appropriate register: LREG1 for max, LREG0 for min.
        // ADDR_MOD_6 auto-increments the offset for the next iteration.
        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0,
                     InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset2);
    }
#else
    // Optimized path using SFPLOADMACRO for instruction-level parallelism.
    // Achieves 3 cycles per input row by overlapping load, swap, round, and store
    // across the SFPU's 5 sub-units (load, simple, MAD, round, store).
    //
    // Pipeline schedule (per row):
    //   Cycle 0: SFPLOADMACRO loads A into LREG[a] (alternating LREG0/LREG1)
    //   Cycle 1: Regular SFPLOAD loads B into LREG2
    //   Cycle 2: SFPLOADMACRO triggers:
    //            - Simple sub-unit: SFPSWAP(min_max) on [a] and LREG2
    //            - Round sub-unit: copies swap result to L16 register
    //            - Store sub-unit: writes L16 to output DST offset
    //
    // The alternation between LREG0 and LREG1 for 'a' avoids read-after-write
    // hazards by ensuring the register being written is not read in the same cycle.

    constexpr int b = p_sfpu::LREG2;  // Always use LREG2 for input B
    constexpr int c = p_sfpu::LREG3;  // Always use LREG3 for store staging

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        int a = i & 1;  // Alternate: even iterations use LREG0, odd use LREG1
        // Macro 0: Load A, schedule swap+round+store via SFPLOADMACRO
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7,
                         offset0 | (a >> 2));
        // Regular load: B into LREG2
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1);
        // Macro 1: triggers store pipeline, using LREG3 as staging
        TT_SFPLOADMACRO((1 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6,
                         offset2 | (c >> 2));
    }

    // Drain the pipeline: 3 NOPs to let the last scheduled instructions complete
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}
```

### Annotated SFPU Kernel Source (INT32/UINT32 path -- `calculate_binary_max_min_int32`)

```cpp
// Template parameters:
//   IS_MAX_OP: true for max, false for min
//   IS_UNSIGNED: true for uint32 comparison, false for int32
//   ITERATIONS: 8 rows per tile face
template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {

    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
    // Fallback path for INT32 max/min.
    // The SFPSWAP VEC_MIN_MAX instruction operates on signed-magnitude floats,
    // which gives incorrect results for 2's complement integers in certain cases.
    // A corrective conditional swap is needed after the initial SFPSWAP.
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load A and B as INT32 into LREG0 and LREG1
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, offset1);

        // Initial min/max swap. For unsigned, mod1=9 reverses the comparison
        // direction because SFPSWAP treats values as signed-magnitude.
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0,
                     IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

        // Corrective phase: SFPSWAP may get the wrong result for negative
        // integers (signed) or large values (unsigned) because it compares
        // in signed-magnitude format. The correction uses condition codes:
        //   - For signed (IS_UNSIGNED=false): check if LREG < 0 (SFPSETCC_MOD1_LREG_LT0)
        //   - For unsigned (IS_UNSIGNED=true): check if LREG >= 0 (SFPSETCC_MOD1_LREG_GTE0)
        // If the condition is met, a conditional SWAP corrects the order.
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0,
                      IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0
                                  : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0,
                      IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0
                                  : sfpi::SFPSETCC_MOD1_LREG_LT0);
        // Conditional swap: only swaps lanes where the condition code is set
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP);
        // Clear condition codes
        TTI_SFPENCC(0, 0, 0, 0);

        // Store the correct result
        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0,
                     InstrModLoadStore::INT32, ADDR_MOD_6, offset2);
    }
#else
    // Optimized SFPLOADMACRO path for INT32 max/min.
    // Achieves 5 cycles per row by pipelining load, swap, setcc, encc, and store.
    // Uses double-buffering with (a0, b0) and (a1, b1) register pairs.
    // A replay buffer records 10 instructions (2 iterations) and replays them.

    constexpr int a0 = p_sfpu::LREG0;
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int a1 = p_sfpu::LREG2;
    constexpr int b1 = p_sfpu::LREG3;
    constexpr int c = p_sfpu::LREG7;  // Store staging register

    // Record 10 instructions into replay buffer slot 0
    // (Wormhole uses load_replay_buf, Blackhole uses lltt::record<lltt::NoExec>)
    // The replay buffer stores the instruction sequence for 2 iterations,
    // then replays it ITERATIONS/2 times for the remaining rows.

    // First iteration: process with a0, b0
    TT_SFPLOADMACRO((0 << 2) | (a0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7,
                     offset0 | (a0 >> 2));           // Load A -> a0
    TT_SFPLOADMACRO((2 << 2) | (b0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7,
                     offset1 | (b0 >> 2));           // Load B -> b0, triggers swap
    TTI_SFPSETCC(0, a1, 0,                           // Set CC on a1 (from prev iter)
                 IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0
                             : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPENCC(0, 0, 0, 0);                         // Clear CC
    TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_6,
                     offset2 | (c >> 2));            // Store result

    // Second iteration: process with a1, b1 (double-buffered)
    TT_SFPLOADMACRO((1 << 2) | (a1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7,
                     offset0 | (a1 >> 2));
    TT_SFPLOADMACRO((2 << 2) | (b1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7,
                     offset1 | (b1 >> 2));
    TTI_SFPSETCC(0, a0, 0,
                 IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0
                             : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPENCC(0, 0, 0, 0);
    TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_6,
                     offset2 | (c >> 2));

    // Replay the 10-instruction sequence for all remaining row pairs
#pragma GCC unroll 4
    for (int i = 0; i < ITERATIONS / 2; ++i) {
        lltt::replay(0, 10);
    }

    // Handle odd iteration count and drain the pipeline
    if constexpr (ITERATIONS & 1) {
        lltt::replay(0, 5);   // Replay first half only
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(5 + 2, 2);  // Replay the setcc+encc correction
    } else {
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(2, 2);  // Final correction for last iteration
    }
    TTI_SFPNOP;  // Final drain
#endif
}
```

### Annotated SFPU Init Source (Float path -- `binary_max_min_init`)

```cpp
// Configures SFPLOADMACRO instruction templates and macro sequences for the
// pipelined max/min operation.
template <bool IS_MAX_OP = true>
inline void binary_max_min_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b = p_sfpu::LREG2;

    // InstructionTemplate[0]: The SFPSWAP instruction to be scheduled by SFPLOADMACRO.
    // mod1=9 when IS_MAX_OP=true: sets VD=max, VC=min (swapped from default).
    // mod1=SFPSWAP_MOD1_VEC_MIN_MAX (=1) when IS_MAX_OP=false: VD=min, VC=max.
    // Register 12 is the template slot for the "simple" sub-unit instruction.
    TTI_SFPSWAP(0, b, 12, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // InstructionTemplate[1]: SFPSHFT2 for the "round" sub-unit -- copies result to L16.
    // Register 13 is the template slot. mod1=6 is SFPSHFT2_MOD1_SHFT_IMM.
    TTI_SFPSHFT2(0, 0, 13, 6);

    // Macro 0: Defines the scheduling for the first SFPLOADMACRO call per iteration.
    // Bit fields encode which sub-unit instructions fire and their timing:
    //   simple_bits: 0x80 (enable) | (1 << 3) (template index 1??) | 4 (delay)
    //   round_bits: 0x80 (enable) | 0x40 (use L16) | (3 << 3) | 5
    // These are loaded into SFPCONFIG macro slot 0.
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (3 << 3) | 5;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);  // Write to macro slot 0
    }

    // Macro 1: Defines the scheduling for the second SFPLOADMACRO call (store phase).
    // Only the store sub-unit fires, writing the L16 value to DST.
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);  // Write to macro slot 1
    }

    // Misc configuration:
    //   StoreMod0: DEFAULT (store uses default load modifier)
    //   UsesLoadMod0ForStore: {1,1} -- both macros use load mod 0 for store
    //   UnitDelayKind: {1,1} -- WaitForElapsedInstructions=1
    TTI_SFPCONFIG(0x330, 8, 1);
#endif
}
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| **SFPLOAD** | Loads a 4-element vector from DST register file into an SFPU local register (LREG). Supports DEFAULT (float) and INT32 load modes. |
| **SFPSTORE** | Stores a 4-element vector from an LREG back to the DST register file. |
| **SFPSWAP** | With `VEC_MIN_MAX` mode (mod1=1): simultaneously computes `VD = min(VC, VD)` and `VC = max(VC, VD)` per element. With mod1=9 (inverted): `VD = max(VC, VD)` and `VC = min(VC, VD)`. With `SWAP` mode: conditional swap based on condition codes. Latency: 2 cycles. |
| **SFPLOADMACRO** | Combines an SFPLOAD with up to 4 scheduled sub-unit instructions (simple, MAD, round, store). Enables instruction-level parallelism by overlapping load, compute, and store across the SFPU's 5 sub-units. |
| **SFPSETCC** | Sets per-lane condition codes based on register comparison. Used in INT32 path to detect when SFPSWAP gave incorrect results (due to signed-magnitude vs 2's complement mismatch). |
| **SFPENCC** | Clears (disables) condition codes. |
| **SFPNOP** | No-operation; used to drain the pipeline after SFPLOADMACRO sequences. |
| **SFPLOADI** | Loads immediate values into SFPU configuration registers. Used to program SFPLOADMACRO macro bit fields. |
| **SFPCONFIG** | Writes to SFPU configuration registers (macro slots, misc settings). |
| **SFPSHFT2** | Shift instruction used as a template instruction for the "round" sub-unit in SFPLOADMACRO -- copies data to L16 staging register. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Input A row (alternates with LREG1 in optimized path) |
| **LREG1** | Input A row (alternates with LREG0 in optimized path); also used for B in fallback path |
| **LREG2** | Input B row (float path); b0 register (int32 path, first iteration pair) |
| **LREG3** | Store staging register 'c' (float path); b1 register (int32 path, second iteration pair) |
| **LREG7** | Store staging register 'c' (int32 path only) |
| **L16** | Internal staging register used by the "round" sub-unit to pass data to the "store" sub-unit |
| **DST[idst0]** | Source tile A (typically DST index 0) |
| **DST[idst1]** | Source tile B (typically DST index 1) |
| **DST[odst]** | Output tile (typically DST index 0, overwriting A) |

### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` to claim the DST register bank.

2. **Unpack to DST**: Two `copy_tile()` calls unpack tiles from CB c_0 and CB c_1 into DST[0] and DST[1] respectively. For SFPU operations, `UnpackToDestFp32` mode is used, ensuring data arrives in FP32 format in DST (except for POWER op).

3. **SFPU initialization** (`binary_max_tile_init`): Called once (or per-tile if activations are present). Configures:
   - Instruction templates for SFPSWAP and SFPSHFT2
   - Macro 0 and Macro 1 bit fields (which sub-units fire and when)
   - Misc configuration (store behavior, delay kinds)

4. **SFPU computation** (`binary_max_tile` -> `calculate_binary_max_min<true>`):
   - Computes byte offsets into DST for both inputs and output
   - Iterates 8 times (one per row in a tile face; the LLK framework calls this function once per face)
   - **Float path (optimized)**: Uses SFPLOADMACRO to pipeline load-swap-store at 3 cycles/row:
     - SFPLOADMACRO loads A into alternating LREG0/LREG1 and schedules the SFPSWAP
     - Regular SFPLOAD loads B into LREG2
     - Second SFPLOADMACRO triggers the store pipeline
   - **INT32 path (optimized)**: Uses SFPLOADMACRO with replay buffers at 5 cycles/row:
     - Adds corrective SFPSETCC + conditional SFPSWAP after initial VEC_MIN_MAX because SFPSWAP compares in signed-magnitude format, which is incorrect for 2's complement integers
     - Double-buffers with (a0,b0) and (a1,b1) register pairs
     - Records 10 instructions into a replay buffer and replays for each pair of rows
   - 3 SFPNOP instructions drain the pipeline at the end

5. **Pack to output CB**: `tile_regs_commit()` signals that DST is ready. `pack_tile(0, cb_out)` packs DST[0] into CB c_2 in the output data format. `tile_regs_release()` frees DST.

6. **CB synchronization**: `cb_push_back(cb_out, 1)` signals the writer kernel. `cb_pop_front()` on both input CBs signals the reader kernel.

### SFPU Configuration

- **Compile-time defines**:
  - `BINARY_SFPU_INIT` = `binary_max_tile_init();` (float), `binary_max_int32_tile_init();` (int32), `binary_max_uint32_tile_init();` (uint32)
  - `BINARY_SFPU_OP` = `binary_max_tile` / `binary_max_int32_tile` / `binary_max_uint32_tile`
  - `APPROX` = compile-time template parameter from `ComputeConfig`; not behaviorally significant for max/min (no approximation involved)

- **Math fidelity**: Not applicable -- MAXIMUM uses SFPSWAP which is an exact comparison, not a polynomial approximation. The `APPROX` template parameter is passed through but has no effect on the max/min logic.

- **fp32_dest_acc_en**: Enabled based on input/output data types. Controls whether DST operates in full FP32 or reduced precision.

- **UnpackToDestMode**: `UnpackToDestFp32` for all source CBs (except POWER op). Ensures tiles are in FP32 in DST before SFPU processes them.

- **`DISABLE_SFPLOADMACRO`**: A compile-time escape hatch that disables the optimized SFPLOADMACRO pipeline and falls back to sequential load-swap-store. This may be needed for debugging or on platforms that do not support SFPLOADMACRO.

### Hardware Compatibility Notes

- **Wormhole B0 vs Blackhole**: The SFPU kernel source code is **identical** for both architectures. The only difference is in address modifier constants used by SFPLOAD/SFPSTORE (`ADDR_MOD_7` and `ADDR_MOD_6` on Blackhole; `ADDR_MOD_3` and `ADDR_MOD_2` on Wormhole). These are architecture-specific register addressing schemes that control auto-increment behavior during tile row iteration.

- **Replay buffer API**: Wormhole uses `load_replay_buf(slot, count, lambda)` while Blackhole uses `lltt::record<lltt::NoExec>(slot, count)` followed by inline instructions. The `lltt::replay(slot, count)` call is used on both to execute the recorded sequence.

- **SFPSWAP signed-magnitude issue**: Both architectures share the same limitation where SFPSWAP's VEC_MIN_MAX mode operates on signed-magnitude representation. For INT32 data (2's complement), a corrective conditional swap is required. This is an RTL-level hardware behavior, not a software bug.

---

## External Knowledge Sources

### DeepWiki References

- **tenstorrent/tt-metal**: Binary_ng program factory architecture, kernel dispatch logic, SFPU op selection via `OpConfig` and `get_sfpu_init_fn`.
- **tenstorrent/tt-llk**: LLK binary SFPU API (`_llk_math_eltwise_binary_sfpu_init_`, `_llk_math_eltwise_binary_sfpu_params_`), SFPU type enumeration, address modifier configuration.
- **tenstorrent/tt-isa-documentation**: SFPSWAP instruction semantics (VEC_MIN_MAX mode, 2-cycle latency, signed-magnitude comparison), SFPLOADMACRO instruction-level parallelism across 5 sub-units, LaneConfig bits for min/max control.
- **tenstorrent/sfpi**: SFPSWAP_MOD1_VEC_MIN_MAX constant (value = 1), SFPSWAP_MOD1_SWAP for conditional swap, SFPSETCC modes, SFPCONFIG macro register layout.

### Confluence References

Not consulted for this analysis. The DeepWiki and source code provided sufficient detail on the SFPSWAP and SFPLOADMACRO instructions used by the MAXIMUM operation.

### Glean References

Not consulted for this analysis. The Wormhole/Blackhole SFPU kernel implementations were directly readable from the codebase and the signed-magnitude correction logic was self-documented in the source.

---

## File Index

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp` | Program factory -- creates kernels, CBs, runtime args |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp` | OpConfig, KernelName enum, utility declarations |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` | OpConfig::as_defines, get_sfpu_init_fn, kernel path resolution |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp` | Device operation struct, SubtileBroadcastType enum |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp` | is_binary_sfpu_op, validate, compute_output_specs |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` | Compute kernel (no broadcast, SFPU) |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp` | Compute kernel (with broadcast, SFPU) |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp` | Compute kernel (scalar B, SFPU) |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp` | PREPROCESS macro for activation preprocessing |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp` | HAS_ACTIVATIONS, BCAST_OP macros |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` | Reader kernel (two-tensor, ng variant) |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` | Writer kernel (ng variant) |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp` | Reader kernel (single-tensor + scalar) |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp` | Writer kernel (scalar path) |
| `tt_metal/hw/inc/api/compute/binary_max_min.h` | Compute API: binary_max_tile, binary_max_tile_init |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h` | LLK wrapper (Blackhole) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h` | LLK wrapper (Wormhole B0) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h` | SFPU kernel impl (Blackhole) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h` | SFPU kernel impl (Wormhole B0) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_init.h` | LLK binary SFPU init framework |
