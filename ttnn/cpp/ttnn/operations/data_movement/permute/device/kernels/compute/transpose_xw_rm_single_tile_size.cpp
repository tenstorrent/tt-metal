// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "api/compute/pack_untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t x_block_size = get_named_compile_time_arg_val("x_block_size");
    constexpr uint32_t w_block_size = get_named_compile_time_arg_val("w_block_size");

    // constexpr uint32_t x_block_size = get_compile_time_arg_val(0);
    // constexpr uint32_t w_block_size = get_compile_time_arg_val(1);

    uint32_t num_blocks = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_tilize = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    experimental::CircularBuffer cb_tilize_exp(cb_tilize);
    experimental::CircularBuffer cb_out_exp(cb_out);

    unary_op_init_common(cb_in, cb_out);
    transpose_wh_init(cb_tilize, cb_out);

    for (uint32_t n = 0; n < num_blocks; n++) {
        // Tilize input via unpack and then pack (asymmetric: x_block_size rows → 1 tile)
        compute_kernel_lib::tilize<
            1,
            cb_in,
            cb_tilize,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1, x_block_size);

        // transpose input
        cb_tilize_exp.wait_front(1);
        transpose_wh_init_short(cb_tilize);
        pack_untilize_dest_init<1>(cb_out);

        tile_regs_acquire();
        transpose_wh_tile(cb_tilize, 0, 0);  // transpose call
        tile_regs_commit();

        // pack and untilize
        cb_out_exp.reserve_back(w_block_size);

        tile_regs_wait();
        pack_untilize_dest<1>(cb_out);  // pack call
        tile_regs_release();

        cb_out_exp.push_back(w_block_size);

        pack_untilize_uninit(cb_out);

        cb_tilize_exp.pop_front(1);
    }
}
