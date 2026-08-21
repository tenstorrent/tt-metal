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

// Retile: untilize input tiles into an intermediate row-major buffer, then tilize into the output
// tile shape. The intermediate is a single L1 allocation shared by untilize (producer) and tilize
// (consumer) to avoid a copy, exposed as two aliased CB views because the producer and consumer
// need different fixed tile/face geometry: mid_cb has the input tile shape, mid_view_cb the output
// tile shape (its bytes stay in the input data format; conversion happens on the final pack).
//
// Untilize the whole assignment, then tilize it, rather than interleaving the two. The output CB
// may be buffer-backed (sharded zero-copy); re-entering tilize_init there would re-base the packer
// write pointer and overwrite already-packed rows.

namespace {

// PACK owns the valid write pointer, so the zero fill runs inside a PACK block.
ALWI void fill_zeros_pages(DataflowBuffer& dfb, uint32_t num_pages, uint32_t page_size) {
    dfb.reserve_back(num_pages);
    PACK({
        const uint32_t dst_addr = dfb.get_write_ptr() << cb_addr_shift;
        volatile tt_l1_ptr uint32_t* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
        const uint32_t num_words = (num_pages * page_size) / sizeof(uint32_t);
        for (uint32_t i = 0; i < num_words; ++i) {
            dst_ptr[i] = 0;
        }
    })
    dfb.push_back(num_pages);
}

}  // namespace

void kernel_main() {
    const uint32_t num_input_blocks = get_arg_val<uint32_t>(0);
    const uint32_t num_real_input_rows = get_arg_val<uint32_t>(1);
    // Shrink-case output cap: emit real rows only. Padded rows would OOB the output DRAM buffer.
    const uint32_t num_real_output_rows = get_arg_val<uint32_t>(2);
    if (num_input_blocks == 0 || num_real_output_rows == 0) {
        return;
    }

    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(0);
    constexpr uint32_t src_cb = get_compile_time_arg_val(1);
    constexpr uint32_t mid_cb = get_compile_time_arg_val(2);
    constexpr uint32_t mid_view_cb = get_compile_time_arg_val(3);
    constexpr uint32_t out_cb = get_compile_time_arg_val(4);
    constexpr uint32_t in_tile_height = get_compile_time_arg_val(5);
    constexpr uint32_t out_tile_height = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(7);
    constexpr uint32_t mid_page_size = get_compile_time_arg_val(8);

    static_assert(in_tile_height > 0 && out_tile_height > 0, "retile kernel requires positive tile heights");
    static_assert(
        (in_tile_height >= out_tile_height && (in_tile_height % out_tile_height) == 0) ||
            (out_tile_height > in_tile_height && (out_tile_height % in_tile_height) == 0),
        "retile kernel requires one tile height to divide the other exactly");

    constexpr uint32_t words_per_out_tile_row = (tiles_per_block * out_tile_size) >> 4;
    const uint32_t total_mid_pages = num_input_blocks * tiles_per_block;

    compute_kernel_hw_startup(src_cb, mid_cb);

    DataflowBuffer mid(mid_cb);
    DataflowBuffer out_dfb(out_cb);

    if (num_real_input_rows > 0) {
        // One tile-row at a time: pack_untilize of a full shard can exceed dest capacity
        // for wide sharded blocks, and InitAndUninit per row matches the previously working path.
        for (uint32_t row = 0; row < num_real_input_rows; ++row) {
            compute_kernel_lib::untilize<
                tiles_per_block,
                src_cb,
                mid_cb,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
        }
    }
    const uint32_t pad_rows = num_input_blocks > num_real_input_rows ? (num_input_blocks - num_real_input_rows) : 0;
    for (uint32_t k = 0; k < pad_rows; ++k) {
        fill_zeros_pages(mid, tiles_per_block, mid_page_size);
    }

    mid.wait_front(total_mid_pages);
    uint32_t block_rd_ptr = 0;
    UNPACK({ block_rd_ptr = get_local_cb_interface(mid_cb).fifo_rd_ptr; })

    // Reconfigure the unpacker/packer from the untilize config (src_cb/mid_cb) to the tilize
    // config (mid_view_cb/out_cb). tilize_init's state_configure is sentinel-only, so the
    // hardware reconfig must be explicit — for bf16 it's a no-op, for bfloat8 it's required.
    reconfig_data_format_srca(src_cb, mid_view_cb);
    pack_reconfig_data_format(mid_cb, out_cb);
    tilize_init(mid_view_cb, tiles_per_block, out_cb);
    for (uint32_t r = 0; r < num_real_output_rows; ++r) {
        UNPACK({ get_local_cb_interface(mid_view_cb).fifo_rd_ptr = block_rd_ptr + r * words_per_out_tile_row; })
        out_dfb.reserve_back(tiles_per_block);
        tilize_block(mid_view_cb, tiles_per_block, out_cb);
        out_dfb.push_back(tiles_per_block);
    }
    tilize_uninit(mid_view_cb, out_cb);

    mid.pop_front(total_mid_pages);
}
