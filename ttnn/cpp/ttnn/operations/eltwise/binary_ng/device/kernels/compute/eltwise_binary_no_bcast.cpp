// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_pre_lhs_id = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs_id = tt::CBIndex::c_1;

    CircularBuffer cb_post_lhs(HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs_id);
    CircularBuffer cb_post_rhs(HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs_id);
    CircularBuffer cb_out(tt::CBIndex::c_2);

    compute_kernel_hw_startup(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id(), cb_out.get_cb_id());
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id());
#endif

    // Inline helper to process n tiles
    auto process_tiles = [&](uint32_t n) {
        PREPROCESS(LHS, CircularBuffer(cb_pre_lhs_id), cb_post_lhs, cb_out, n);
        cb_post_lhs.wait_front(n);

        PREPROCESS(RHS, CircularBuffer(cb_pre_rhs_id), cb_post_rhs, cb_out, n);
        cb_post_rhs.wait_front(n);

        cb_out.reserve_back(n);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST)
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id());
#endif
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            BINARY_OP(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id(), i, i, i);
            PROCESS_POST_ACTIVATIONS(i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_out.get_cb_id());
        }
        tile_regs_release();

        cb_out.push_back(n);
        cb_post_lhs.pop_front(n);
        cb_post_rhs.pop_front(n);
    };

    // Process full chunks
    uint32_t num_full_chunks = num_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
        process_tiles(num_tiles_per_cycle);
    }

    // Process remainder
    uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_tiles(remainder);
    }
}
