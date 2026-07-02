// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/trigonometry.h"

#include "eltwise_utils_common.hpp"
#include "eltwise_utils.hpp"

ALWI void process_tile(
    tt::CBIndex cb_pre_lhs_id,
    tt::CBIndex cb_post_lhs_id,
    tt::CBIndex cb_pre_rhs_id,
    tt::CBIndex cb_post_rhs_id,
    tt::CBIndex cb_out_id,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

    CircularBuffer cb_post_lhs(cb_post_lhs_id);
    CircularBuffer cb_post_rhs(cb_post_rhs_id);
    CircularBuffer cb_out(cb_out_id);

#if BCAST_INPUT
#define CB_PRE_BCAST cb_pre_rhs_id
#define CB_PRE_OTHER cb_pre_lhs_id
    CircularBuffer& cb_post_bcast = cb_post_rhs;
    CircularBuffer& cb_post_other = cb_post_lhs;
#else
#define CB_PRE_BCAST cb_pre_lhs_id
#define CB_PRE_OTHER cb_pre_rhs_id
    CircularBuffer& cb_post_bcast = cb_post_lhs;
    CircularBuffer& cb_post_other = cb_post_rhs;
#endif

    PREPROCESS(BCAST_OP, CircularBuffer(CB_PRE_BCAST), cb_post_bcast, cb_out, num_tiles_per_cycle);
    cb_post_bcast.wait_front(num_tiles_per_cycle);

    for (uint32_t j = tile_start; j < freq; ++j) {
        PREPROCESS(OTHER_OP, CircularBuffer(CB_PRE_OTHER), cb_post_other, cb_out, num_tiles_per_cycle);
        cb_post_other.wait_front(num_tiles_per_cycle);

        cb_out.reserve_back(num_tiles_per_cycle);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST)
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id());
#endif
        tile_regs_acquire();
        BINARY_OP(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id(), 0, 0, 0);
        PROCESS_POST_ACTIVATIONS(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out.get_cb_id());
        tile_regs_release();

        cb_out.push_back(num_tiles_per_cycle);
        cb_post_other.pop_front(num_tiles_per_cycle);
    }
    cb_post_bcast.pop_front(num_tiles_per_cycle);
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_pre_lhs_id = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs_id = tt::CBIndex::c_1;
    constexpr auto cb_out_id = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs_id = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs_id;
    constexpr auto cb_post_rhs_id = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs_id;

    binary_op_init_common(cb_post_lhs_id, cb_post_rhs_id, cb_out_id);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs_id, cb_post_rhs_id);
#endif

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(
            cb_pre_lhs_id,
            cb_post_lhs_id,
            cb_pre_rhs_id,
            cb_post_rhs_id,
            cb_out_id,
            tile_freq,
            tile_start,
            num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile(
            cb_pre_lhs_id,
            cb_post_lhs_id,
            cb_pre_rhs_id,
            cb_post_rhs_id,
            cb_out_id,
            remaining_iterations,
            tile_start,
            num_tiles_per_cycle);
    }
}
