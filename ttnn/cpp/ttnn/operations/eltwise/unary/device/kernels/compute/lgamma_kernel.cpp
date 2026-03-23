// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/lgamma.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    constexpr float M_PI = 3.14159265358979323846f;

    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            tile_regs_acquire();

            // copy input to dst 0 and 1
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);  // x
            copy_tile(cb_input, 0, 1);  // x

            // tile 0 = res_stirling
            lgamma_stirling_tile_init();
            lgamma_stirling_tile(0);

            // fill tile 2 with M_PI
            fill_tile_init();
            fill_tile(2, M_PI);

            // tile 1 = frac (x)
            rounding_op_tile_init();
            frac_tile(1);

            // tile 1 = frac (x) * M_PI
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);

            // tile 1 =  sin(frac (x) * M_PI)
            sin_tile_init();
            sin_tile(1);

            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 2);  // x
            copy_tile(cb_input, 0, 3);  // x

            // tile 3 = floor(x)
            rounding_op_tile_init();
            floor_tile(3);

            // tile 2 = ( x == floor(x)) condition
            eq_binary_tile_init();
            eq_binary_tile(2, 3, 2);

            // fill tile 3 with 0.0f
            fill_tile_init();
            fill_tile(3, 0.0f);

            // tile 1 = 0 if x == floor(x), otherwise sin(frac (x) * M_PI)
            where_tile_init();
            where_tile<DataFormat::Float16_b>(2, 3, 1, 1);

            // abs(integer adjusted sin(frac (x) * M_PI))
            abs_tile_init();
            abs_tile(1);

            // log|sin(pi*x)|. Zero sin(pi*x) at integers handled
            log_tile_init();
            log_tile(1);

            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 2);  // x

            lgamma_adjusted_tile_init();
            lgamma_adjusted_tile(0, 1, 2, 0);

            tile_regs_commit();

            tile_regs_wait();

            pack_tile(0, cb_output);

            tile_regs_release();

            cb_pop_front(cb_input, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
