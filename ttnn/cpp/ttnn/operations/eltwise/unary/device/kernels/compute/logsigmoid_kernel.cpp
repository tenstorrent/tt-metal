// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/logsigmoid.h"
#include "compute_kernel_api/eltwise_unary/negative.h"

namespace NAMESPACE {
void MAIN {
    // Compile-time arguments
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;   // Input (x)
    constexpr auto cb_output = tt::CBIndex::c_2;  // Output

    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);

            // ===================================================================
            // Compute logsigmoid(x) = -softplus(-x) directly in DST registers
            // ===================================================================
            tile_regs_acquire();

            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);  // Load x to DST[0]
            copy_tile(cb_input, 0, 1);  // Load x to DST[1]

            // Negate DST[1]: -x (using sign-bit flip)
            negative_tile_init();
            negative_tile(1);

            // Apply exp with fast+approx mode to DST[1]: exp(-x)
            exp_tile_init<true, true>();  // Fast+approx exp
            exp_tile<true, true>(1);

            // Apply logsigmoid SFPU: logsigmoid(x) = -softplus(-x)
            logsigmoid_tile_init();
            logsigmoid_tile(0, 1, 0);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_output);
            tile_regs_release();

            cb_pop_front(cb_input, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
