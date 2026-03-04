// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Compute Kernel (two-pass)
//   Pass 1: tilize -> mean -> subtract -> square -> variance -> rsqrt(var+eps)
//   Pass 2: tilize -> subtract mean -> multiply rsqrt -> untilize

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

constexpr uint32_t cb_in = tt::CBIndex::c_0;
constexpr uint32_t cb_tilized = tt::CBIndex::c_1;
constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_2;
constexpr uint32_t cb_mean = tt::CBIndex::c_3;         // 1 tile: mean value
constexpr uint32_t cb_centered = tt::CBIndex::c_4;     // Wt tiles: centered values
constexpr uint32_t cb_var = tt::CBIndex::c_5;          // 1 tile: variance/rsqrt
constexpr uint32_t cb_gamma = tt::CBIndex::c_6;        // Wt tiles: gamma (persistent)
constexpr uint32_t cb_beta = tt::CBIndex::c_7;         // Wt tiles: beta (persistent)
constexpr uint32_t cb_eps = tt::CBIndex::c_8;          // 1 tile: epsilon (persistent)
constexpr uint32_t cb_normalized = tt::CBIndex::c_16;  // Wt tiles: temp for squared
constexpr uint32_t cb_out = tt::CBIndex::c_17;

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t nblocks_per_core = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);

void kernel_main() {
    compute_kernel_hw_startup(cb_in, cb_reduce_scaler, cb_out);

    // Wait for persistent gamma/beta tiles (pushed once by reader, used every iteration)
    if constexpr (has_gamma) {
        cb_wait_front(cb_gamma, Wt);
    }
    if constexpr (has_beta) {
        cb_wait_front(cb_beta, Wt);
    }

    for (uint32_t block = 0; block < nblocks_per_core; ++block) {
        // ===== PASS 1: Compute statistics =====

        // Phase 1a: Tilize first read (cb_in -> cb_tilized)
        compute_kernel_lib::tilize<cb_in, cb_tilized, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(
            Wt, 1);

        // Phase 2: Mean (cb_tilized -> cb_mean)
        // WaitUpfrontNoPop: tiles persist for Phase 3
        compute_kernel_lib::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract Mean (cb_tilized - cb_mean -> cb_centered)
        // NoPop on both: cb_tilized freed manually, cb_mean kept for Pass 2
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        cb_pop_front(cb_tilized, Wt);

        // Phase 4: Square (cb_centered -> cb_normalized)
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_centered, cb_normalized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 5: Variance (reduce squared -> cb_var)
        compute_kernel_lib::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_normalized, cb_reduce_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 6: Add eps + rsqrt (cb_var -> cb_normalized[0])
        rsqrt_tile_init();
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_var,
            cb_eps,
            cb_normalized,
            compute_kernel_lib::BinaryInputBlockShape::single(),
            {},
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) { rsqrt_tile(dst_idx); });

        // ===== PASS 2: Normalize =====

        // Phase 1b: Tilize second read (cb_in -> cb_tilized)
        compute_kernel_lib::tilize<cb_in, cb_tilized, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(
            Wt, 1);

        // Phase 3b: Subtract Mean (cb_tilized - cb_mean -> cb_centered)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryOutputPolicy::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 7: Normalize (cb_centered * rsqrt -> cb_tilized)
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
                compute_kernel_lib::BinaryOutputPolicy::PerTile,
                compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                cb_centered, cb_normalized, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::single());
        }
        cb_pop_front(cb_normalized, 1);

        // Phase 8: Scale by Gamma (conditional)
        // cb_tilized has Wt tiles from Phase 7. Gamma is persistent in cb_gamma (Wt tiles).
        // In-place: cb_tilized -> cb_tilized (WaitUpfrontPopAtEnd reads all, then pops, then Bulk writes)
        if constexpr (has_gamma) {
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                compute_kernel_lib::BinaryOutputPolicy::Bulk,
                compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                cb_tilized, cb_gamma, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        // Phase 9: Shift by Beta (conditional)
        // Same in-place pattern as Phase 8
        if constexpr (has_beta) {
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                compute_kernel_lib::BinaryOutputPolicy::Bulk,
                compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                cb_tilized, cb_beta, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        // Phase 10: Untilize (cb_tilized -> cb_out)
        compute_kernel_lib::untilize<
            Wt,
            cb_tilized,
            cb_out,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::NoWait>(1);
    }
}
