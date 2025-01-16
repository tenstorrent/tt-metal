// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
// #include "debug/dprint_tensix.h"

inline void print_bf16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t x_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t w_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);

    uint32_t start_block = get_arg_val<uint32_t>(0);
    uint32_t end_block = get_arg_val<uint32_t>(1);

    UNPACK(DPRINT << "N: " << N << ENDL());
    UNPACK(DPRINT << "start_block: " << start_block << ENDL());
    UNPACK(DPRINT << "end_block: " << end_block << ENDL());
    UNPACK(DPRINT << "x_blocks: " << x_blocks << ENDL());
    UNPACK(DPRINT << "w_blocks: " << w_blocks << ENDL());
    UNPACK(DPRINT << "H: " << H << ENDL());

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_tilize = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t block = start_block; block < end_block; block++) {
        // Decompose block into w_block, x_block, and xw_block indices
        uint32_t rem = block;
        uint32_t w_block = rem % w_blocks;  // Which W block are we in?
        rem /= w_blocks;

        uint32_t x_block = rem % x_blocks;  // Which X block?
        rem /= x_blocks;

        uint32_t h = rem % H;

        // tilize input via unpack and then pack
        tilize_init_short(cb_in, 1, cb_tilize);

        cb_wait_front(cb_in, 1);
        UNPACK(
            uint32_t base_address = get_local_cb_interface(tt::CBIndex::c_0).fifo_rd_ptr
                                    << 4);  // Remove header size added by descriptor
        UNPACK(DPRINT << "base_address: " << base_address << ENDL());
        UNPACK(print_bf16_pages(base_address, 32, 1));
        cb_reserve_back(cb_tilize, 1);

        tilize_block(cb_in, 1, cb_tilize);  // tilize and pack into cb_tilize

        cb_push_back(cb_tilize, 1);
        cb_pop_front(cb_in, 1);

        tilize_uninit(cb_in, cb_tilize);

        // transpose input
        cb_wait_front(cb_tilize, 1);

        UNPACK(DPRINT << "after tilizing" << ENDL());
        for (uint8_t r = 0; r < 1; ++r) {
            SliceRange sr = SliceRange{.h0 = r, .h1 = uint8_t(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            UNPACK(DPRINT << TileSlice(cb_tilize, 0, sr, true, true) << ENDL());
        }

        transpose_wh_init_short(cb_tilize);
        pack_untilize_dst_init_short<1>(cb_out);

        tile_regs_acquire();
        transpose_wh_tile(cb_tilize, 0, 0);  // transpose call
        // dprint_tensix_dest_reg(0);
        tile_regs_commit();

        // pack and untilize
        cb_reserve_back(cb_out, 1);

        tile_regs_wait();
        pack_untilize_dst<1>(cb_out);  // pack call
        tile_regs_release();

        PACK(DPRINT << "after transposing" << ENDL());
        PACK(
            uint32_t base_address = get_local_cb_interface(cb_out).fifo_wr_ptr
                                    << 4);  // Remove header size added by descriptor
        PACK(DPRINT << "base_address: " << base_address << ENDL());
        PACK(print_bf16_pages(base_address, 32, 32));
        PACK(DPRINT << ENDL());

        cb_push_back(cb_out, 1);

        pack_untilize_uninit(cb_out);

        cb_pop_front(cb_tilize, 1);
    }
}
}  // namespace NAMESPACE
