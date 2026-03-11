// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 6 (affine): Full layer normalization with optional gamma/beta.
// Pass 1: tilize+reduce->mean. Pass 2: tilize+sub+square+reduce->var+eps+rsqrt->inv_std.
// Pass 3: tilize+sub+mul_inv_std+[mul_gamma]+[add_beta]->output, untilize->RM.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

constexpr uint32_t c_0 = 0;    // RM input staging for tilize
constexpr uint32_t c_1 = 1;    // Tilized input tiles
constexpr uint32_t c_2 = 2;    // Reduce scaler 1/W
constexpr uint32_t c_3 = 3;    // Epsilon scalar
constexpr uint32_t c_4 = 4;    // Gamma tilized (optional)
constexpr uint32_t c_5 = 5;    // Beta tilized (optional)
constexpr uint32_t c_16 = 16;  // Final tiles before untilize
constexpr uint32_t c_24 = 24;  // Row-wise mean (1 tile)
constexpr uint32_t c_25 = 25;  // Intermediate tiles
constexpr uint32_t c_26 = 26;  // Row-wise variance (1 tile)
constexpr uint32_t c_27 = 27;  // Inv_std (1 tile)
constexpr uint32_t c_17 = 17;  // Untilized RM output

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);

// Determine the CB that holds the final result before untilize.
// Both gamma+beta: mul(c_16->c_25), add(c_25->c_16) => c_16
// Gamma only: mul(c_16->c_25) => c_25
// Beta only: add(c_16->c_25) => c_25
// Neither: c_16
constexpr uint32_t c_pre_untilize = (has_gamma && has_beta) ? c_16 : (has_gamma || has_beta) ? c_25 : c_16;

void kernel_main() {
    // Runtime args
    uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    // Hardware init - must come first
    compute_kernel_hw_startup(c_0, c_2, c_1);

    // Gamma/beta tilize at program start (once, before main loop)
    if constexpr (has_gamma) {
        compute_kernel_lib::tilize<
            c_0,
            c_4,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(Wt, 1);
    }
    if constexpr (has_beta) {
        compute_kernel_lib::tilize<
            c_0,
            c_5,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(Wt, 1);
    }

    // Wait for epsilon CB once before loop (persistent, never popped during loop)
    cb_wait_front(c_3, 1);

    // Wait for gamma/beta CBs once before loop (persistent for all tile-rows)
    if constexpr (has_gamma) {
        cb_wait_front(c_4, Wt);
    }
    if constexpr (has_beta) {
        cb_wait_front(c_5, Wt);
    }

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        // === Pass 1: Tilize + Reduce -> Mean ===

        // Phase 1: Tilize c_0 -> c_1
        compute_kernel_lib::tilize<
            c_0,
            c_1,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(Wt, 1);

        // Phase 2: Reduce row -> mean (c_1 -> c_24)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            c_1, c_2, c_24, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // === Pass 2: Tilize + Subtract + Square + Reduce -> Variance ===

        // Phase 3: Tilize c_0 -> c_1 (pass 2)
        compute_kernel_lib::tilize<
            c_0,
            c_1,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(Wt, 1);

        // Phase 4: Sub mean -> centered: sub<COL>(c_1, c_24) -> c_25
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            c_1, c_24, c_25, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5: Square centered: square(c_25) -> c_1
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            c_25, c_1, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 6: Reduce row -> variance: reduce<SUM,REDUCE_ROW>(c_1) -> c_26
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            c_1, c_2, c_26, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 7: Add epsilon + rsqrt -> inv_std: add<SCALAR>(c_26, c_3) -> c_27 with rsqrt post_op
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            c_26, c_3, c_27, compute_kernel_lib::BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // === Pass 3: Tilize + Subtract Mean + Multiply inv_std + Affine -> Output ===

        // Phase 8: Tilize c_0 -> c_1 (pass 3)
        compute_kernel_lib::tilize<
            c_0,
            c_1,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(Wt, 1);

        // Phase 9: Sub mean (again): sub<COL>(c_1, c_24) -> c_25
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            c_1, c_24, c_25, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 10: Multiply by inv_std: mul<COL>(c_25, c_27) -> c_16
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            c_25, c_27, c_16, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 11: Multiply by gamma (conditional): mul<NONE>(c_16, c_4) -> c_25
        if constexpr (has_gamma) {
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                compute_kernel_lib::BinaryOutputPolicy::PerTile,
                compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                c_16, c_4, c_25, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        }

        // Phase 12: Add beta (conditional)
        if constexpr (has_beta) {
            constexpr uint32_t src_cb = has_gamma ? c_25 : c_16;
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                compute_kernel_lib::BinaryOutputPolicy::PerTile,
                compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                src_cb, c_5, c_16, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        }

        // Phase 13: Untilize -> RM output
        compute_kernel_lib::untilize<
            Wt,
            c_pre_untilize,
            c_17,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);

        // End of tile-row: pop persistent per-tile-row CBs
        cb_pop_front(c_24, 1);  // mean
        cb_pop_front(c_27, 1);  // inv_std
    }

    // End of program: pop persistent program CBs
    if constexpr (has_gamma) {
        cb_pop_front(c_4, Wt);
    }
    if constexpr (has_beta) {
        cb_pop_front(c_5, Wt);
    }
}
