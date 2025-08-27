// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_binary.h"

#include "eltwise_utils_common.hpp"
#include "eltwise_utils.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS))
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif

    PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
    cb_wait_front(cb_post_rhs, num_tiles_per_cycle);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);

        cb_reserve_back(cb_out, num_tiles_per_cycle);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif
        tile_regs_acquire();
        BINARY_OP(cb_post_lhs, cb_post_rhs, 0, 0, 0);
        PROCESS_POST_ACTIVATIONS(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);
        cb_push_back(cb_out, num_tiles_per_cycle);
    }
}
}  // namespace NAMESPACE
