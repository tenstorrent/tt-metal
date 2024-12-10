// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

#include "eltwise_defines.hpp"
#include "eltwise_utils.hpp"

namespace NAMESPACE {

ALWI void process_tile(
    tt::CBIndex cb_pre_lhs,
    tt::CBIndex cb_post_lhs,
    tt::CBIndex cb_pre_rhs,
    tt::CBIndex cb_post_rhs,
    tt::CBIndex cb_out,
    uint32_t freq,
    uint32_t tile_start) {
    using namespace ckernel;
    constexpr uint32_t onetile = 1;

#if BCAST_INPUT
    auto cb_bcast = cb_post_rhs;
    auto cb_other = cb_post_lhs;
#else
    auto cb_bcast = cb_post_lhs;
    auto cb_other = cb_post_rhs;
#endif

#if PREPROCESS_A && (BCAST_INPUT == 0)
    PREPROCESS(PREPROCESS_A_INIT, PREPROCESS_A_APPLY, cb_pre_lhs, cb_post_lhs, cb_out, onetile);
#elif PREPROCESS_B && (BCAST_INPUT == 1)
    PREPROCESS(PREPROCESS_B_INIT, PREPROCESS_B_APPLY, cb_pre_rhs, cb_post_rhs, cb_out, onetile);
#endif

    cb_wait_front(cb_bcast, onetile);

    for (uint32_t j = tile_start; j < freq; ++j) {
#if PREPROCESS_A && (BCAST_INPUT == 1)
        PREPROCESS(PREPROCESS_A_INIT, PREPROCESS_A_APPLY, cb_pre_lhs, cb_post_lhs, cb_out, onetile);
#elif PREPROCESS_B && (BCAST_INPUT == 0)
        PREPROCESS(PREPROCESS_B_INIT, PREPROCESS_B_APPLY, cb_pre_rhs, cb_post_rhs, cb_out, onetile);
#endif
        cb_wait_front(cb_other, onetile);

        cb_reserve_back(cb_out, onetile);

#if PREPROCESS_A || PREPROCESS_B
        binary_op_specific_init<true, BINARY_OP_TYPE>();
#endif
        tile_regs_acquire();
        BINARY_OP(cb_post_lhs, cb_post_rhs, 0, 0, 0);
#if POSTPROCESS
        POSTPROCESS_INIT();
        POSTPROCESS_APPLY(0);
#endif
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, onetile);
        cb_pop_front(cb_other, onetile);
    }
    cb_pop_front(cb_bcast, onetile);
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = PREPROCESS_A ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = PREPROCESS_B ? tt::CBIndex::c_4 : cb_pre_rhs;

    binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out);

#if not(PREPROCESS_A || PREPROCESS_B)
    binary_op_specific_init<true, BINARY_OP_TYPE>();
#endif

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(cb_pre_lhs, cb_post_lhs, cb_pre_rhs, cb_post_rhs, cb_out, tile_freq, tile_start);
    }

    if (remaining_iterations > 0) {
        process_tile(cb_pre_lhs, cb_post_lhs, cb_pre_rhs, cb_post_rhs, cb_out, remaining_iterations, tile_start);
    }
}
}  // namespace NAMESPACE
