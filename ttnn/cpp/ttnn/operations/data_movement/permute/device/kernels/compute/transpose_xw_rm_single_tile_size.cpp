// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t x_block_size = get_named_compile_time_arg_val("x_block_size");
    constexpr uint32_t w_block_size = get_named_compile_time_arg_val("w_block_size");

    // constexpr uint32_t x_block_size = get_compile_time_arg_val(0);
    // constexpr uint32_t w_block_size = get_compile_time_arg_val(1);

    uint32_t num_blocks = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_tilize = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t n = 0; n < num_blocks; n++) {
        // Tilize input via unpack and then pack (asymmetric: x_block_size rows → 1 tile)
        compute_kernel_lib::tilize(
            cb_in,        // Input CB (row-major)
            1,            // block_w (1 tile output)
            cb_tilize,    // Output CB (tiled)
            1,            // num_blocks (1 iteration per outer loop)
            1,            // subblock_h (default)
            0,            // old_icb (not used)
            x_block_size  // input_count (asymmetric: rows != tiles)
        );

        // transpose input
        cb_wait_front(cb_tilize, 1);
        transpose_wh_init_short(cb_tilize);
        pack_untilize_dest_init<1>(cb_out);

        tile_regs_acquire();
        transpose_wh_tile(cb_tilize, 0, 0);  // transpose call
        tile_regs_commit();

        // pack and untilize
        cb_reserve_back(cb_out, w_block_size);

        tile_regs_wait();
        pack_untilize_dest<1>(cb_out);  // pack call
        tile_regs_release();

        cb_push_back(cb_out, w_block_size);

        pack_untilize_uninit(cb_out);

        cb_pop_front(cb_tilize, 1);
    }
}
}  // namespace NAMESPACE
