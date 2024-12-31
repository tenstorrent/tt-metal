// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

#include "eltwise_defines.hpp"
#include "eltwise_utils.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = PREPROCESS_A ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = PREPROCESS_B ? tt::CBIndex::c_4 : cb_pre_rhs;

    binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out);

#if not(PREPROCESS_A || PREPROCESS_B)
    binary_op_specific_init<true, BINARY_OP_TYPE>();
#endif

    constexpr uint32_t onetile = 1;

#if PREPROCESS_B
    PREPROCESS(PREPROCESS_B_INIT, PREPROCESS_B_APPLY, cb_pre_rhs, cb_post_rhs, cb_out, onetile);
#endif
    cb_wait_front(cb_post_rhs, onetile);

    for(uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
#if PREPROCESS_A
        PREPROCESS(PREPROCESS_A_INIT, PREPROCESS_A_APPLY, cb_pre_lhs, cb_post_lhs, cb_out, onetile);
#endif
        cb_wait_front(cb_post_lhs, onetile);

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

        cb_pop_front(cb_post_lhs, onetile);
        cb_push_back(cb_out, onetile);
    }
}
}  // namespace NAMESPACE
