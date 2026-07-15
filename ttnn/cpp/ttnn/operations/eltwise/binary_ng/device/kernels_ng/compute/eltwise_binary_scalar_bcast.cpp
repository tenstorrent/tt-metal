// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils.hpp"
// DataflowBuffer for the op's own buffers; CircularBuffer (via eltwise_utils.hpp) is
// still used for the shared preprocess_*_impl helper call sites (see PREPROCESS below).
#include "api/dataflow/dataflow_buffer.h"

ALWI void process_tile(
    tt::CBIndex cb_bcast,
    tt::CBIndex cb_llk_post,
    tt::CBIndex cb_pre_lhs,
    tt::CBIndex cb_post_lhs,
    tt::CBIndex cb_pre_rhs,
    tt::CBIndex cb_post_rhs,
    tt::CBIndex cb_out,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

#if BCAST_INPUT
#define CB_PRE_BCAST cb_pre_rhs
#define CB_POST_BCAST cb_post_rhs
#define CB_PRE_OTHER cb_pre_lhs
#define CB_POST_OTHER cb_post_lhs
#define EXP_CB_POST_BCAST exp_dfb_post_rhs
#define EXP_CB_POST_OTHER exp_dfb_post_lhs
#else
#define CB_PRE_BCAST cb_pre_lhs
#define CB_POST_BCAST cb_post_lhs
#define CB_PRE_OTHER cb_pre_rhs
#define CB_POST_OTHER cb_post_rhs
#define EXP_CB_POST_BCAST exp_dfb_post_lhs
#define EXP_CB_POST_OTHER exp_dfb_post_rhs
#endif
    DataflowBuffer exp_dfb_bcast(cb_bcast);
    DataflowBuffer exp_dfb_llk_post(cb_llk_post);
    DataflowBuffer exp_dfb_post_lhs(cb_post_lhs);
    DataflowBuffer exp_dfb_post_rhs(cb_post_rhs);
    DataflowBuffer exp_dfb_out(cb_out);

    exp_dfb_bcast.wait_front(num_tiles_per_cycle);
    pack_reconfig_data_format(cb_out, cb_llk_post);
    unary_bcast_init<BroadcastType::SCALAR>(cb_bcast, cb_llk_post);
    exp_dfb_llk_post.reserve_back(num_tiles_per_cycle);
    tile_regs_acquire();
    unary_bcast<BroadcastType::SCALAR>(cb_bcast, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_llk_post);
    exp_dfb_llk_post.push_back(num_tiles_per_cycle);
    tile_regs_release();

    pack_reconfig_data_format(cb_llk_post, cb_out);
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_out)));
#endif

    PREPROCESS(
        BCAST_OP,
        CircularBuffer(CB_PRE_BCAST),
        CircularBuffer(CB_POST_BCAST),
        CircularBuffer(cb_out),
        num_tiles_per_cycle);
    EXP_CB_POST_BCAST.wait_front(num_tiles_per_cycle);

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        PREPROCESS(
            OTHER_OP,
            CircularBuffer(CB_PRE_OTHER),
            CircularBuffer(CB_POST_OTHER),
            CircularBuffer(cb_out),
            num_tiles_per_cycle);
        EXP_CB_POST_OTHER.wait_front(num_tiles_per_cycle);

        exp_dfb_out.reserve_back(num_tiles_per_cycle);

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

        exp_dfb_out.push_back(num_tiles_per_cycle);
        EXP_CB_POST_OTHER.pop_front(num_tiles_per_cycle);
    }
    exp_dfb_bcast.pop_front(num_tiles_per_cycle);
    EXP_CB_POST_BCAST.pop_front(num_tiles_per_cycle);
}
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_out = tt::CBIndex::c_2;

#if SRC_BCAST
    constexpr auto cb_bcast = tt::CBIndex::c_0;
    constexpr auto cb_llk_post = tt::CBIndex::c_5;
    constexpr auto cb_pre_lhs = cb_llk_post;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_llk_post;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : tt::CBIndex::c_1;
#endif
#if SRC_BCAST_B
    constexpr auto cb_bcast = tt::CBIndex::c_1;
    constexpr auto cb_llk_post = tt::CBIndex::c_6;
    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = cb_llk_post;
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : tt::CBIndex::c_0;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_llk_post;
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
