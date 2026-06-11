// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);

    constexpr auto cb_pre_lhs = dfb::cb_a;
    constexpr auto cb_pre_rhs = dfb::cb_b;
    constexpr auto cb_out = dfb::cb_c;

    // The lhs/rhs activation-interim DFBs (c_3/c_4) are bound only when the corresponding activations
    // exist, so the dfb::<name> token must be removed by the preprocessor when absent: a discarded
    // `?:` branch is still name-looked-up at parse time. HAS_ACTIVATIONS is driven by the host-emitted
    // PROCESS_{LHS,RHS}_ACTIVATIONS defines — the same condition that gates the host-side DFB binding.
#if HAS_ACTIVATIONS(LHS)
    constexpr auto cb_post_lhs = dfb::cb_a_interim;
#else
    constexpr auto cb_post_lhs = cb_pre_lhs;
#endif
#if HAS_ACTIVATIONS(RHS)
    constexpr auto cb_post_rhs = dfb::cb_b_interim;
#else
    constexpr auto cb_post_rhs = cb_pre_rhs;
#endif

    binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif

    // Inline helper to process n tiles
    auto process_tiles = [&](uint32_t n) {
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, n);
        cb_wait_front(cb_post_lhs, n);

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, n);
        cb_wait_front(cb_post_rhs, n);

        cb_reserve_back(cb_out, n);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST)
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            BINARY_OP(cb_post_lhs, cb_post_rhs, i, i, i);
            PROCESS_POST_ACTIVATIONS(i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();

        cb_push_back(cb_out, n);
        cb_pop_front(cb_post_lhs, n);
        cb_pop_front(cb_post_rhs, n);
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
