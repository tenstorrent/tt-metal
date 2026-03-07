// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t approx_arg = get_arg_val<uint32_t>(0);
    const bool use_approx = (approx_arg != 0u);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);

            if (use_approx) {
                exp_tile_init<true>();
                exp_tile<true>(0);
                log1p_tile_init<true>();
                log1p_tile<true>(0);
            } else {
                exp_tile_init<false, true>();
                exp_tile<false, true>(0);
                log1p_tile_init<false>();
                log1p_tile<false>(0);
            }
            tanh_tile_init<false>();
            tanh_tile<false>(0);

#ifdef INP_FLOAT32
            copy_tile(cb_input, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
#endif
#ifdef INP_FLOAT
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_input, 0, 0);
#endif

            tile_regs_commit();

            tile_regs_wait();

            pack_tile(0, cb_output);

            tile_regs_release();

            cb_pop_front(cb_input, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
