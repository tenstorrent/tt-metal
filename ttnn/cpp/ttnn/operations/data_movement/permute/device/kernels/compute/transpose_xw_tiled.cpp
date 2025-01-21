// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "tt_metal/hw/inc/circular_buffer.h"

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t x_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t w_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);
    constexpr uint32_t misalignment = get_compile_time_arg_val(4);

    constexpr uint32_t misalignment_div_16 = misalignment >> cb_addr_shift;

    uint32_t offset_div_16 = 0;
    uint32_t start_block = get_arg_val<uint32_t>(0);
    uint32_t end_block = get_arg_val<uint32_t>(1);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_tilize = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t block = start_block; block < end_block; block++) {
        // Decompose block into w_block, x_block, and xw_block indices
#ifdef TRISC_UNPACK
        uint32_t rem = block;
        uint32_t w_block = rem % w_blocks;  // Which W block are we in?
        rem /= w_blocks;

        uint32_t x_block = rem % x_blocks;  // Which X block?
        rem /= x_blocks;

        uint32_t h = rem % H;
#endif

        // tilize input via unpack and then pack
        tilize_init_short(cb_in, 1, cb_tilize);

        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_tilize, 1);

        // For BH, DRAM read alignment is 64B, but each subtile/face line is 32B, so every odd numbered row in BFLOAT16
        // is misaligned
#ifdef TRISC_UNPACK
        if constexpr (misalignment > 0) {
            // if h is an odd number, offset_div_16 by misalignment
            if ((h & 1) == 1) {
                std::uint32_t operand_id = get_operand_id(cb_in);
                get_local_cb_interface(operand_id).fifo_rd_ptr += misalignment_div_16;
            }
        }
#endif
        // custom_tilize_block(cb_in, offset_div_16, 1, cb_tilize);  // tilize and pack into cb_tilize
        tilize_block(cb_in, 1, cb_tilize);

#ifdef TRISC_UNPACK
        if constexpr (misalignment > 0) {
            // if h is an odd number, offset_div_16 by misalignment
            if ((h & 1) == 1) {
                std::uint32_t operand_id = get_operand_id(cb_in);
                get_local_cb_interface(operand_id).fifo_rd_ptr -= misalignment_div_16;
            }
        }
#endif

        cb_push_back(cb_tilize, 1);
        cb_pop_front(cb_in, 1);

        tilize_uninit(cb_in, cb_tilize);

        // transpose input
        cb_wait_front(cb_tilize, 1);

        transpose_wh_init_short(cb_tilize);
        pack_untilize_dst_init_short<1>(cb_out);

        tile_regs_acquire();
        transpose_wh_tile(cb_tilize, 0, 0);  // transpose call
        tile_regs_commit();

        // pack and untilize
        cb_reserve_back(cb_out, 1);

        tile_regs_wait();
        pack_untilize_dst<1>(cb_out);  // pack call
        tile_regs_release();

        cb_push_back(cb_out, 1);

        pack_untilize_uninit(cb_out);

        cb_pop_front(cb_tilize, 1);
    }
}
}  // namespace NAMESPACE
