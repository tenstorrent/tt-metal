// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/logsigmoid.h"
#include "compute_kernel_api/eltwise_unary/negative.h"

namespace NAMESPACE {
void MAIN {
    // Compile-time arguments
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    // Encode -1.0 for negation using union (C++17 compatible)
    union {
        float f;
        uint32_t u;
    } neg_one_converter;
    neg_one_converter.f = -1.0f;
    uint32_t neg_one_encoded = neg_one_converter.u;

    constexpr auto cb_input = tt::CBIndex::c_0;   // Input (x)
    constexpr auto cb_tmp1 = tt::CBIndex::c_3;    // Intermediate: exp(-x)
    constexpr auto cb_output = tt::CBIndex::c_2;  // Output

    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);

            // ===================================================================
            // Stage 1: Compute exp(-x) using fast+approx => store in cb_tmp1
            // ===================================================================
            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();

            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);  // Load x to DST[0]

            // Negate: -x
            mul_unary_tile(0, neg_one_encoded);

            // Apply exp with fast+approx mode: exp(-x)
            exp_tile_init<true, true>();  // Fast+approx exp
            exp_tile<true, true>(0);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_tmp1);
            tile_regs_release();

            cb_push_back(cb_tmp1, 1);

            // ===================================================================
            // Stage 2: LogSigmoid SFPU - combine x and exp(-x)
            // ===================================================================
            cb_wait_front(cb_tmp1, 1);  // exp(-x)

            tile_regs_acquire();

            // Load both values to DST
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);  // DST[0] = x

            // Negate x to get -x (required by SFPU)
            mul_unary_tile(0, neg_one_encoded);  // DST[0] = -x

            copy_tile_to_dst_init_short(cb_tmp1);
            copy_tile(cb_tmp1, 0, 1);  // DST[1] = exp(-x)

            // Apply logsigmoid SFPU: logsigmoid(x) = -softplus(-x)
            logsigmoid_tile_init();
            logsigmoid_tile(0, 1, 0);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_output);
            tile_regs_release();

            cb_pop_front(cb_input, 1);
            cb_pop_front(cb_tmp1, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
