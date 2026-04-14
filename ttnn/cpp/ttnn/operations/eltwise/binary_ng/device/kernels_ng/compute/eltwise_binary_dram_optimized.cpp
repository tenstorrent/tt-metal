// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils.hpp"

void kernel_main() {
    constexpr uint32_t num_batches = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles_per_batch = get_compile_time_arg_val(1);

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif

    const uint32_t large_chunk = num_batches * num_tiles_per_batch;
    uint32_t remaining = num_tiles;
    while (remaining > 0) {
        uint32_t n_tiles_proc;
        if (remaining >= large_chunk) {
            n_tiles_proc = large_chunk;
        } else if (remaining >= num_tiles_per_batch) {
            n_tiles_proc = num_tiles_per_batch;
        } else {
            n_tiles_proc = remaining;
        }

        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, n_tiles_proc);
        cb_wait_front(cb_post_lhs, n_tiles_proc);

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, n_tiles_proc);
        cb_wait_front(cb_post_rhs, n_tiles_proc);

        cb_reserve_back(cb_out, n_tiles_proc);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST)
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif
        tile_regs_acquire();
        for (uint32_t i = 0; i < n_tiles_proc; ++i) {
            BINARY_OP(cb_post_lhs, cb_post_rhs, i, i, i);
            PROCESS_POST_ACTIVATIONS(i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n_tiles_proc; ++i) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();

        cb_push_back(cb_out, n_tiles_proc);
        cb_pop_front(cb_post_lhs, n_tiles_proc);
        cb_pop_front(cb_post_rhs, n_tiles_proc);
        remaining -= n_tiles_proc;
    }
}
