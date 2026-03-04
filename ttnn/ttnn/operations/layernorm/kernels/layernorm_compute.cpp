// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Compute Kernel
// Stage 2: tilize + reduce mean + subtract mean + untilize

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

constexpr uint32_t cb_in = tt::CBIndex::c_0;
constexpr uint32_t cb_tilized = tt::CBIndex::c_1;
constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_2;
constexpr uint32_t cb_mean = tt::CBIndex::c_3;
constexpr uint32_t cb_centered = tt::CBIndex::c_4;
constexpr uint32_t cb_out = tt::CBIndex::c_17;

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t nblocks_per_core = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);

void kernel_main() {
    // Hardware init: srcA=cb_in, srcB=cb_reduce_scaler, output=cb_out
    compute_kernel_hw_startup(cb_in, cb_reduce_scaler, cb_out);

    for (uint32_t block = 0; block < nblocks_per_core; ++block) {
        // Phase 1: Tilize (cb_in -> cb_tilized)
        compute_kernel_lib::tilize<cb_in, cb_tilized, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(
            Wt, 1);

        // Phase 2: Compute Mean via reduce (cb_tilized -> cb_mean)
        // WaitUpfrontNoPop: tiles persist in cb_tilized for Phase 3
        compute_kernel_lib::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract Mean (cb_tilized - cb_mean -> cb_centered)
        // A: cb_tilized already waited from Phase 2 (NoWaitNoPop)
        // B: cb_mean freshly pushed (WaitUpfrontPopAtEnd - consumed)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryOutputPolicy::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Manual pop of cb_tilized after Phase 3 (was held by WaitUpfrontNoPop)
        cb_pop_front(cb_tilized, Wt);

        // Phase 10: Untilize (cb_centered -> cb_out)
        compute_kernel_lib::
            untilize<Wt, cb_centered, cb_out, compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit>(1);
    }
}
