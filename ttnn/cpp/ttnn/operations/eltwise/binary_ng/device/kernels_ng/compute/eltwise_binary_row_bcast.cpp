// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

#if SRC_BCAST
    constexpr auto cb_bcast = cb_post_lhs;
    constexpr auto cb_llk_post = tt::CBIndex::c_5;
    constexpr auto cb_left = tt::CBIndex::c_5;
    constexpr auto cb_right = cb_post_rhs;

#endif
#if SRC_BCAST_B
    constexpr auto cb_bcast = cb_post_rhs;
    constexpr auto cb_llk_post = tt::CBIndex::c_6;
    constexpr auto cb_left = cb_post_lhs;
    constexpr auto cb_right = tt::CBIndex::c_6;
#endif

    binary_op_init_common(cb_left, cb_right, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);

        cb_reserve_back(cb_llk_post, 1);
        unary_bcast_init<BroadcastType::ROW>(cb_bcast, cb_llk_post);

        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_bcast, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_llk_post);
        cb_push_back(cb_llk_post, 1);
        tile_regs_release();

        cb_pop_front(cb_bcast, 1);
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_left, cb_right);
        cb_reserve_back(cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_llk_post, 1);

        tile_regs_acquire();
        BINARY_OP(cb_left, cb_right, 0, 0, 0);
        PROCESS_POST_ACTIVATIONS(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_left, num_tiles_per_cycle);
        cb_pop_front(cb_right, num_tiles_per_cycle);
    }
}
}  // namespace NAMESPACE
