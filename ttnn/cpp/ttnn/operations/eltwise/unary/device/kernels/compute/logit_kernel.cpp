// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/clamp.h"
#include "compute_kernel_api/eltwise_unary/rsub.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/where.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(1);

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1;

    init_sfpu(cb_input, cb_output);
    // When epsilon is provided, clamp input x to [eps, 1-eps]
    // Compute logit(x) = log(x / (1-x))
    // For eps > 0.5, use conditional where operation to handle sign correctly
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            cb_reserve_back(cb_tmp0, 1);

            tile_regs_acquire();

            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);
#ifdef CLAMP
            clamp_tile_init();
            clamp_tile(0, packed_scalar1, packed_scalar2);
#endif
            tile_regs_commit();

            tile_regs_wait();

            pack_tile(0, cb_tmp0);

            tile_regs_release();

            cb_push_back(cb_tmp0, 1);
            cb_wait_front(cb_tmp0, 1);

            tile_regs_acquire();

            copy_tile_init(cb_tmp0);
            copy_tile(cb_tmp0, 0, 0);
            copy_tile(cb_tmp0, 0, 1);

            rsub_tile_init();
            rsub_tile(0, 0x3F800000u);  // 1.0 - x

            div_binary_tile_init();
            div_binary_tile(1, 0, 0);

            log_tile_init();
            log_tile(0);
#ifdef WHERE  // Conditional negation: when eps > 0.5 and input < eps, negate the logit result (multiply by -1.0) to
              // ensure positive output. WHERE selects negated result (true) or original result (false).
            copy_dest_values(2, 0);

            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 1);

            unary_lt_tile_init();
            unary_lt_tile(1, packed_scalar1);

            binop_with_scalar_tile_init();
            mul_unary_tile(0, 0xBF800000);  // multiply by -1.0

            where_tile_init();
            where_tile(1, 0, 2, 0);
#endif

            tile_regs_commit();

            tile_regs_wait();

            pack_tile(0, cb_output);

            tile_regs_release();

            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_input, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
