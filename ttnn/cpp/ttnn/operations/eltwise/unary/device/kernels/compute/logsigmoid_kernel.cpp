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
#include "compute_kernel_api/eltwise_unary/logsigmoid.h"

namespace NAMESPACE {
void MAIN {
    // Runtime arguments
    uint32_t beta_encoded = get_arg_val<uint32_t>(0);
    uint32_t threshold_encoded = get_arg_val<uint32_t>(1);

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

    constexpr auto cb_input = tt::CBIndex::c_0;   // Input
    constexpr auto cb_tmp0 = tt::CBIndex::c_1;    // Intermediate: beta * x
    constexpr auto cb_tmp1 = tt::CBIndex::c_3;    // Intermediate: exp(-beta * x)
    constexpr auto cb_output = tt::CBIndex::c_2;  // Output

    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);

            // ===================================================================
            // Stage 1: Scale input by beta => store in cb_tmp0
            // ===================================================================
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();

            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);  // Load input to DST[0]

            // Scale by beta: beta * x
            binop_with_scalar_tile_init();
            mul_unary_tile(0, beta_encoded);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_tmp0);
            tile_regs_release();

            cb_push_back(cb_tmp0, 1);

            // ===================================================================
            // Stage 2: Compute exp(-beta * x) using fast+approx => store in cb_tmp1
            // ===================================================================
            cb_wait_front(cb_tmp0, 1);
            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();

            copy_tile_to_dst_init_short(cb_tmp0);
            copy_tile(cb_tmp0, 0, 0);  // Load beta * x to DST[0]

            // Negate: -(beta * x)
            mul_unary_tile(0, neg_one_encoded);

            // Apply exp with fast+approx mode: exp(-beta * x)
            exp_tile_init<true, true>();
            exp_tile<true, true>(0);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_tmp1);
            tile_regs_release();

            cb_push_back(cb_tmp1, 1);

            // ===================================================================
            // Stage 3: LogSigmoid SFPU - combine scaled input and exp
            // ===================================================================
            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);

            tile_regs_acquire();

            // Load both values to DST
            copy_tile_to_dst_init_short(cb_tmp0);
            copy_tile(cb_tmp0, 0, 0);  // DST[0] = beta * x

            copy_tile_to_dst_init_short(cb_tmp1);
            copy_tile(cb_tmp1, 0, 1);  // DST[1] = exp(-beta * x)

            // Apply logsigmoid SFPU with both inputs
            logsigmoid_tile_init();
            logsigmoid_tile(0, 1, 0, beta_encoded, threshold_encoded);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_output);
            tile_regs_release();

            cb_pop_front(cb_input, 1);
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
