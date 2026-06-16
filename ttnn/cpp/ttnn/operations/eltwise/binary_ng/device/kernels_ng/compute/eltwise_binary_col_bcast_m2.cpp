// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"

#include "experimental/kernel_args.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils.hpp"

ALWI void process_tile(
    uint32_t cb_bcast,
    uint32_t cb_llk_post,
    uint32_t cb_pre_lhs,
    uint32_t cb_post_lhs,
    uint32_t cb_pre_rhs,
    uint32_t cb_post_rhs,
    uint32_t cb_out,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

#if BCAST_INPUT
#define CB_PRE_BCAST cb_pre_rhs
#define CB_POST_BCAST cb_post_rhs
#define CB_PRE_OTHER cb_pre_lhs
#define CB_POST_OTHER cb_post_lhs
#else
#define CB_PRE_BCAST cb_pre_lhs
#define CB_POST_BCAST cb_post_lhs
#define CB_PRE_OTHER cb_pre_rhs
#define CB_POST_OTHER cb_post_rhs
#endif
    cb_wait_front(cb_bcast, num_tiles_per_cycle);
    pack_reconfig_data_format(cb_out, cb_llk_post);
    unary_bcast_init<BroadcastType::COL>(cb_bcast, cb_llk_post);
    cb_reserve_back(cb_llk_post, num_tiles_per_cycle);
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(cb_bcast, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_llk_post);
    cb_push_back(cb_llk_post, num_tiles_per_cycle);
    tile_regs_release();

    pack_reconfig_data_format(cb_llk_post, cb_out);
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_out)));
#endif

    PREPROCESS(BCAST_OP, CB_PRE_BCAST, CB_POST_BCAST, cb_out, num_tiles_per_cycle);
    cb_wait_front(CB_POST_BCAST, num_tiles_per_cycle);

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        PREPROCESS(OTHER_OP, CB_PRE_OTHER, CB_POST_OTHER, cb_out, num_tiles_per_cycle);
        cb_wait_front(CB_POST_OTHER, num_tiles_per_cycle);

        cb_reserve_back(cb_out, num_tiles_per_cycle);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST)
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif
        tile_regs_acquire();
        BINARY_OP(cb_post_lhs, cb_post_rhs, 0, 0, 0);
        PROCESS_POST_ACTIVATIONS(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(CB_POST_OTHER, num_tiles_per_cycle);
    }
    cb_pop_front(cb_bcast, num_tiles_per_cycle);
    cb_pop_front(CB_POST_BCAST, num_tiles_per_cycle);
}

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t tile_freq = get_arg(args::tile_freq);
    uint32_t tile_start = get_arg(args::tile_start);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_out = dfb::out;

#if SRC_BCAST
    constexpr auto cb_bcast = dfb::pre_lhs;        // c_0
    constexpr auto cb_llk_post = dfb::llk_post_a;  // c_5
    constexpr auto cb_pre_lhs = cb_llk_post;
    constexpr auto cb_pre_rhs = dfb::pre_rhs;  // c_1
#if HAS_ACTIVATIONS(LHS)
    constexpr auto cb_post_lhs = dfb::post_lhs;  // c_3
#else
    constexpr auto cb_post_lhs = cb_llk_post;
#endif
#if HAS_ACTIVATIONS(RHS)
    constexpr auto cb_post_rhs = dfb::post_rhs;  // c_4
#else
    constexpr auto cb_post_rhs = dfb::pre_rhs;  // c_1
#endif
#endif
#if SRC_BCAST_B
    constexpr auto cb_bcast = dfb::pre_rhs;        // c_1
    constexpr auto cb_llk_post = dfb::llk_post_b;  // c_6
    constexpr auto cb_pre_lhs = dfb::pre_lhs;      // c_0
    constexpr auto cb_pre_rhs = cb_llk_post;
#if HAS_ACTIVATIONS(LHS)
    constexpr auto cb_post_lhs = dfb::post_lhs;  // c_3
#else
    constexpr auto cb_post_lhs = dfb::pre_lhs;  // c_0
#endif
#if HAS_ACTIVATIONS(RHS)
    constexpr auto cb_post_rhs = dfb::post_rhs;  // c_4
#else
    constexpr auto cb_post_rhs = cb_llk_post;
#endif
#endif

    binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(
            cb_bcast,
            cb_llk_post,
            cb_pre_lhs,
            cb_post_lhs,
            cb_pre_rhs,
            cb_post_rhs,
            cb_out,
            tile_freq,
            tile_start,
            num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile(
            cb_bcast,
            cb_llk_post,
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
