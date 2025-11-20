// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils.hpp"

#if WHERE_TTS || WHERE_TST
#include "compute_kernel_api/eltwise_unary/where.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#endif

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
#if WHERE_TTS || WHERE_TST
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);
    const auto scalar_val = reinterpret_cast<const float*>(&scalar_value);
#endif

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

#if WHERE_TTS || WHERE_TST
    unary_op_init_common(cb_post_lhs, cb_out);
    BINARY_SFPU_INIT
#else
    binary_op_init_common(cb_left, cb_right, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);

#if WHERE_TTS || WHERE_TST
        // WHERE operations: predicate is cb_post_lhs, tensor is cb_post_rhs
        // For ROW broadcast, broadcast the tensor if needed first
#if SRC_BCAST_B
        // Tensor (cb_post_rhs) needs to be broadcasted
        cb_reserve_back(cb_llk_post, num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_post_rhs, cb_llk_post);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_post_rhs, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_llk_post);
        cb_push_back(cb_llk_post, num_tiles_per_cycle);
        tile_regs_release();
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);
        cb_wait_front(cb_llk_post, num_tiles_per_cycle);
#endif
#if SRC_BCAST
        // Predicate (cb_post_lhs) needs to be broadcasted
        cb_reserve_back(cb_llk_post, num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_post_lhs, cb_llk_post);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_post_lhs, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_llk_post);
        cb_push_back(cb_llk_post, num_tiles_per_cycle);
        tile_regs_release();
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);
        cb_wait_front(cb_llk_post, num_tiles_per_cycle);
#endif

        cb_reserve_back(cb_out, num_tiles_per_cycle);
        tile_regs_acquire();
        // Copy predicate to reg 0
#if SRC_BCAST
        copy_tile_to_dst_init_short(cb_llk_post);
        copy_tile(cb_llk_post, 0, 0);  // Copy broadcasted predicate to reg 0
#else
        copy_tile_to_dst_init_short(cb_post_lhs);
        copy_tile(cb_post_lhs, 0, 0);  // Copy predicate to reg 0
#endif

#if WHERE_TTS
        // TTS: tensor (true) goes to reg 1, scalar (false) goes to reg 2
#if SRC_BCAST_B
        copy_tile_to_dst_init_short(cb_llk_post);
        copy_tile(cb_llk_post, 0, 1);  // Copy broadcasted tensor to reg 1
#else
        copy_tile_to_dst_init_short(cb_post_rhs);
        copy_tile(cb_post_rhs, 0, 1);  // Copy tensor to reg 1
#endif
        fill_tile_init();
#ifdef FILL_WITH_VALUE_FLOAT
        FILL_LLK(2, *scalar_val);  // Fill scalar (false) to reg 2
#endif
#ifdef FILL_WITH_VALUE_INT
        FILL_LLK(2, scalar_value);  // Fill scalar (false) to reg 2
#endif
        BINARY_SFPU_OP(0, 1, 2, 0);  // WHERE operation
        PROCESS_POST_ACTIVATIONS(0);
#endif

#if WHERE_TST
        // TST: scalar (true) goes to reg 1, tensor (false) goes to reg 2
        fill_tile_init();
#ifdef FILL_WITH_VALUE_FLOAT
        FILL_LLK(1, *scalar_val);  // Fill scalar (true) to reg 1
#endif
#ifdef FILL_WITH_VALUE_INT
        FILL_LLK(1, scalar_value);  // Fill scalar (true) to reg 1
#endif
#if SRC_BCAST_B
        copy_tile_to_dst_init_short(cb_llk_post);
        copy_tile(cb_llk_post, 0, 2);  // Copy broadcasted tensor to reg 2
#else
        copy_tile_to_dst_init_short(cb_post_rhs);
        copy_tile(cb_post_rhs, 0, 2);  // Copy tensor to reg 2
#endif
        BINARY_SFPU_OP(0, 1, 2, 0);  // WHERE operation
        PROCESS_POST_ACTIVATIONS(0);
#endif

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
#if SRC_BCAST
        cb_pop_front(cb_llk_post, num_tiles_per_cycle);
#else
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);
#endif
#if SRC_BCAST_B
        cb_pop_front(cb_llk_post, num_tiles_per_cycle);
#else
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);
#endif
#else
        cb_reserve_back(cb_llk_post, num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_bcast, cb_llk_post);

        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_bcast, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_llk_post);
        cb_push_back(cb_llk_post, num_tiles_per_cycle);
        tile_regs_release();

        cb_pop_front(cb_bcast, num_tiles_per_cycle);
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
        cb_pop_front(cb_left, num_tiles_per_cycle);
        cb_pop_front(cb_right, num_tiles_per_cycle);
#endif
    }
}
}  // namespace NAMESPACE
