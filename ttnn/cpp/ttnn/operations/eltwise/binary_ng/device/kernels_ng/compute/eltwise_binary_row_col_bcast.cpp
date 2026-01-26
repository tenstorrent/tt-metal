// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#include "compute_kernel_api/bcast.h"

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils.hpp"

namespace NAMESPACE {

ALWI void process_tile(
    tt::CBIndex cb_pre_lhs,
    tt::CBIndex cb_post_lhs,
    tt::CBIndex cb_pre_rhs,
    tt::CBIndex cb_post_rhs,
    tt::CBIndex cb_out,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

#if BCAST_INPUT  // ROW_A_COL_B
#define CB_PRE_BCAST cb_pre_rhs
#define CB_POST_BCAST cb_post_rhs
#define CB_PRE_OTHER cb_pre_lhs
#define CB_POST_OTHER cb_post_lhs
    constexpr auto cb_llk_post = tt::CBIndex::c_5;
    constexpr auto cb_left = tt::CBIndex::c_5;
    auto cb_right = cb_post_rhs;
#else  // ROW_B_COL_A
#define CB_PRE_BCAST cb_pre_lhs
#define CB_POST_BCAST cb_post_lhs
#define CB_PRE_OTHER cb_pre_rhs
#define CB_POST_OTHER cb_post_rhs
    constexpr auto cb_llk_post = tt::CBIndex::c_6;
    auto cb_left = cb_post_lhs;
    constexpr auto cb_right = tt::CBIndex::c_6;
#endif

    binary_op_init_common(cb_left, cb_right, cb_out);
    PREPROCESS(BCAST_OP, CB_PRE_BCAST, CB_POST_BCAST, cb_out, num_tiles_per_cycle);
    cb_wait_front(CB_POST_BCAST, num_tiles_per_cycle);

    for (uint32_t j = tile_start; j < freq; ++j) {
        PREPROCESS(OTHER_OP, CB_PRE_OTHER, CB_POST_OTHER, cb_out, num_tiles_per_cycle);
        cb_wait_front(CB_POST_OTHER, num_tiles_per_cycle);
        cb_reserve_back(cb_llk_post, num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(CB_POST_OTHER, cb_llk_post);

        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(CB_POST_OTHER, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_llk_post);
        cb_push_back(cb_llk_post, num_tiles_per_cycle);
        tile_regs_release();

        cb_pop_front(CB_POST_OTHER, num_tiles_per_cycle);
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_left, cb_right);

        cb_reserve_back(cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_llk_post, num_tiles_per_cycle);
        tile_regs_acquire();
        BINARY_OP(cb_left, cb_right, 0, 0, 0);
        PROCESS_POST_ACTIVATIONS(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_llk_post, num_tiles_per_cycle);
    }
    cb_pop_front(CB_POST_BCAST, num_tiles_per_cycle);
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(
            cb_pre_lhs, cb_post_lhs, cb_pre_rhs, cb_post_rhs, cb_out, tile_freq, tile_start, num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile(
            cb_pre_lhs,
            cb_post_lhs,
            cb_pre_rhs,
            cb_post_rhs,
            cb_out,
            remaining_iterations,
            tile_start,
            num_tiles_per_cycle);
    }
}
}  // namespace NAMESPACE
