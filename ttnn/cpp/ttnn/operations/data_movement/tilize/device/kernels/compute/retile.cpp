// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "internal/circular_buffer_interface.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Fused retile: untilize input tiles → intermediate RM → tilize to output tile shape.
//
// Intermediate page size is max(in_tile_size, out_tile_size), passed as a CT arg.
// Never call get_tile_size() on the mid CBs.
//
// c_1 (mid_untilize_cb): input-tile face geometry for pack_untilize.
// c_2 (mid_tilize_cb):   output-tile face geometry for tilize.
// RM bytes are copied c_1 → c_2 (same page size) via a PACK-side L1 memcpy.
//
// Requires in_tile_height >= out_tile_height and in_tile_height % out_tile_height == 0.

namespace {

ALWI void copy_mid_pages(uint32_t src_cb, uint32_t dst_cb, uint32_t num_pages, uint32_t page_size) {
    DataflowBuffer src(src_cb);
    DataflowBuffer dst(dst_cb);

    // wait_front is UNPACK-only (valid rd_ptr); reserve_back is PACK-only (valid wr_ptr).
    src.wait_front(num_pages);
    dst.reserve_back(num_pages);

    // Exchange the UNPACK rd address to PACK, which performs the L1 copy.
    UNPACK({ mailbox_write(ckernel::ThreadId::PackThreadId, src.get_read_ptr() << cb_addr_shift); })
    PACK({
        const uint32_t src_addr = mailbox_read(ckernel::ThreadId::UnpackThreadId);
        const uint32_t dst_addr = dst.get_write_ptr() << cb_addr_shift;
        volatile tt_l1_ptr uint32_t* src_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
        volatile tt_l1_ptr uint32_t* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
        const uint32_t num_words = (num_pages * page_size) / sizeof(uint32_t);
        for (uint32_t i = 0; i < num_words; ++i) {
            dst_ptr[i] = src_ptr[i];
        }
        mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
        mailbox_write(ckernel::ThreadId::MathThreadId, 1);
    })
    UNPACK({ mailbox_read(ckernel::ThreadId::PackThreadId); })
    MATH({ mailbox_read(ckernel::ThreadId::PackThreadId); })

    dst.push_back(num_pages);
    src.pop_front(num_pages);
}

}  // namespace

void kernel_main() {
    const uint32_t num_input_blocks = get_arg_val<uint32_t>(0);
    if (num_input_blocks == 0) {
        return;
    }

    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(0);
    constexpr uint32_t src_cb = get_compile_time_arg_val(1);
    constexpr uint32_t mid_untilize_cb = get_compile_time_arg_val(2);
    constexpr uint32_t mid_tilize_cb = get_compile_time_arg_val(3);
    constexpr uint32_t out_cb = get_compile_time_arg_val(4);
    constexpr uint32_t in_tile_height = get_compile_time_arg_val(5);
    constexpr uint32_t out_tile_height = get_compile_time_arg_val(6);
    constexpr uint32_t mid_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(8);

    static_assert(in_tile_height >= out_tile_height, "retile kernel requires in_tile_height >= out_tile_height");
    static_assert(
        out_tile_height > 0 && (in_tile_height % out_tile_height) == 0,
        "retile kernel requires in_tile_height to be divisible by out_tile_height");

    constexpr uint32_t height_ratio = in_tile_height / out_tile_height;
    constexpr uint32_t words_per_out_tile_row = (tiles_per_block * out_tile_size) >> 4;

    compute_kernel_hw_startup(src_cb, mid_untilize_cb);

    DataflowBuffer mid_tilize(mid_tilize_cb);
    DataflowBuffer out_dfb(out_cb);

    for (uint32_t b = 0; b < num_input_blocks; ++b) {
        compute_kernel_lib::untilize<
            tiles_per_block,
            src_cb,
            mid_untilize_cb,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);

        copy_mid_pages(mid_untilize_cb, mid_tilize_cb, tiles_per_block, mid_page_size);

        // One untilize block holds height_ratio output tile-rows of RM data in N max-sized pages.
        mid_tilize.wait_front(tiles_per_block);
        uint32_t saved_rd_ptr = 0;
        UNPACK({ saved_rd_ptr = get_local_cb_interface(mid_tilize_cb).fifo_rd_ptr; })

        tilize_init(mid_tilize_cb, tiles_per_block, out_cb);
        for (uint32_t r = 0; r < height_ratio; ++r) {
            UNPACK({
                if (r > 0) {
                    get_local_cb_interface(mid_tilize_cb).fifo_rd_ptr = saved_rd_ptr + r * words_per_out_tile_row;
                }
            })
            out_dfb.reserve_back(tiles_per_block);
            tilize_block(mid_tilize_cb, tiles_per_block, out_cb);
            out_dfb.push_back(tiles_per_block);
        }
        UNPACK({ get_local_cb_interface(mid_tilize_cb).fifo_rd_ptr = saved_rd_ptr; })
        mid_tilize.pop_front(tiles_per_block);
        tilize_uninit(mid_tilize_cb, out_cb);

        reconfig_data_format_srca(mid_tilize_cb, src_cb);
        pack_reconfig_data_format(out_cb, mid_untilize_cb);
    }
}
