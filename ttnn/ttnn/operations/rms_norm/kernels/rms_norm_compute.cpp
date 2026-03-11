// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Compute Kernel
// Stage 4: tilize, square, reduce, add_eps+rsqrt, normalize_mul(COL), gamma_mul(ROW), untilize

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_tilized = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_sq = 3;
constexpr uint32_t cb_rms = 4;
constexpr uint32_t cb_eps = 5;
constexpr uint32_t cb_rms_inv = 6;
constexpr uint32_t cb_gamma = 7;
constexpr uint32_t cb_norm_temp = 8;  // Temp CB for normalize output when gamma active
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_untilized = 17;

// Compile-time args
constexpr uint32_t Ht_max = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t input_is_rm = get_compile_time_arg_val(2);
constexpr uint32_t has_gamma = get_compile_time_arg_val(3);

// When has_gamma: Phase 5 writes to cb_norm_temp, gamma mul reads from cb_norm_temp -> cb_out
// When no gamma: Phase 5 writes directly to cb_out
constexpr uint32_t cb_normalize_dst = has_gamma ? cb_norm_temp : cb_out;

void kernel_main() {
    // Runtime arg: actual Ht for this core
    uint32_t Ht = get_arg_val<uint32_t>(0);
    if (Ht == 0) {
        return;
    }

    // Hardware startup: srcA=cb_tilized, srcB=cb_scaler, ocb=cb_out
    compute_kernel_hw_startup(cb_tilized, cb_scaler, cb_out);

    // Pre-loop: Tilize gamma from c_0 -> c_7
    // Gamma is always RM (1,1,1,W). Reader loaded replicated sticks into c_0 (Wt tile-pages).
    if constexpr (has_gamma) {
        compute_kernel_lib::tilize<
            cb_input_rm,
            cb_gamma,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(Wt, 1);
    }

    for (uint32_t row = 0; row < Ht; ++row) {
        // Phase 1: Tilize (RM path only) c_0 -> c_1
        if constexpr (input_is_rm) {
            compute_kernel_lib::tilize<
                cb_input_rm,
                cb_tilized,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(Wt, 1);
        }

        // Phase 2: Square c_1 -> c_3
        // WaitUpfrontNoPop: persist c_1 for Phase 5 (normalize multiply)
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_tilized, cb_sq, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 3: Reduce SUM REDUCE_ROW c_3 -> c_4 with c_2 scaler
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_sq, cb_scaler, cb_rms, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1));

        // Phase 4: Add epsilon + rsqrt: c_4 + c_5 -> c_6
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_rms, cb_eps, cb_rms_inv, compute_kernel_lib::BinaryInputBlockShape::of(1, 1), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 5: Normalize multiply x * rms_inv (COL broadcast)
        // c_1 already waited from Phase 2 (WaitUpfrontNoPop), use NoWaitNoPop
        // c_6 has 1 tile (rms_inv), COL broadcast replicates across Wt columns
        // Output goes to cb_normalize_dst: cb_norm_temp (c_8) if gamma, cb_out (c_16) if no gamma
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_tilized, cb_rms_inv, cb_normalize_dst, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Manual pop c_1 after Phase 5 (was persisted since Phase 2)
        cb_pop_front(cb_tilized, Wt);

        // Phase 6: Gamma multiply (optional, ROW broadcast with persisted gamma)
        // cb_norm_temp (c_8) has Wt tiles (normalized output from Phase 5)
        // cb_gamma (c_7) has Wt tiles (gamma, tilized before main loop)
        // ROW broadcast: gamma row 0 is broadcast across all rows of A
        // Output to cb_out (c_16) -- same CB writer always reads
        if constexpr (has_gamma) {
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
                compute_kernel_lib::BinaryOutputPolicy::PerTile,
                compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                cb_norm_temp, cb_gamma, cb_out, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        }

        // Phase 7: Untilize (RM path only)
        // Source is always cb_out (c_16) -> cb_untilized (c_17)
        if constexpr (input_is_rm) {
            compute_kernel_lib::untilize<
                Wt,
                cb_out,
                cb_untilized,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
        }
    }

    // Pop gamma tiles from c_7 at the end (if gamma was loaded)
    if constexpr (has_gamma) {
        cb_pop_front(cb_gamma, Wt);
    }
}
