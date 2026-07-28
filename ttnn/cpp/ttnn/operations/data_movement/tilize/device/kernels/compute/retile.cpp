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
// A tiled tensor is a batch of 2-D matrix slices (all leading dims flattened), each independently
// padded on its -2 dim to a whole number of tiles. Because the input and output tile heights
// differ, each slice's height rounds up differently (round_up(S, in_tile_h) vs round_up(S,
// out_tile_h)), so alignment padding must be applied *per slice*, not once at the end of the
// flattened tensor. This kernel therefore maps every unit of work back to its slice via the slice
// geometry (slice_in_rows / slice_out_rows) so the boundary tile of each slice is padded (grow) or
// truncated (shrink) correctly, independent of how the host split the work across cores.

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
    // Work is split into "units": one unit is a tile-row of the taller tile — an output tile-row
    // when growing (out taller) or an input tile-row when shrinking (in taller). `global_unit_start`
    // is the index of this core's first unit within the flattened tensor, used with the slice
    // geometry to recover each unit's position inside its slice.
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    const uint32_t global_unit_start = get_arg_val<uint32_t>(1);
    const uint32_t slice_in_rows = get_arg_val<uint32_t>(2);   // input tile-rows per slice
    const uint32_t slice_out_rows = get_arg_val<uint32_t>(3);  // output tile-rows per slice
    if (num_units == 0) {
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

    // Shrink: one input tile-row untilizes to `ratio` output tile-rows. Grow: `ratio` input
    // tile-rows form one output tile-row. One tile height must divide the other exactly.
    constexpr bool shrink = in_tile_height >= out_tile_height;
    constexpr uint32_t ratio = shrink ? (in_tile_height / out_tile_height) : (out_tile_height / in_tile_height);

    // The intermediate block holds one taller-tile-row of row-major data: `ratio` input rows when
    // growing, one input row when shrinking.
    constexpr uint32_t in_rows_per_iter = shrink ? 1u : ratio;
    constexpr uint32_t block_pages = in_rows_per_iter * tiles_per_block;
    constexpr uint32_t words_per_out_tile_row = (tiles_per_block * out_tile_size) >> 4;

    compute_kernel_hw_startup(src_cb, mid_cb);

    DataflowBuffer mid(mid_cb);
    DataflowBuffer out_dfb(out_cb);

    for (uint32_t i = 0; i < num_units; ++i) {
        const uint32_t unit = global_unit_start + i;

        // Resolve this unit's position within its slice, then derive how many input tile-rows are
        // real (untilized from DRAM) vs. alignment padding (zero-filled), and how many output
        // tile-rows to emit.
        uint32_t real_rows;  // input tile-rows to untilize this iteration
        uint32_t pad_rows;   // zero-filled (grow-boundary) input tile-rows this iteration
        uint32_t out_count;  // output tile-rows to tilize this iteration
        if constexpr (shrink) {
            // One input tile-row → up to `ratio` output tile-rows; the last input row of a slice
            // may map to fewer real output rows (the extra output rows would be below the slice's
            // real height and are dropped rather than written).
            const uint32_t local_in_row = unit % slice_in_rows;
            const uint32_t out_base = local_in_row * ratio;
            const uint32_t remaining = slice_out_rows - out_base;
            out_count = remaining < ratio ? remaining : ratio;
            real_rows = 1;
            pad_rows = 0;
        } else {
            // `ratio` input tile-rows → one output tile-row; the last output row of a slice may be
            // formed from fewer real input rows, with the remainder zero-padded within the tile.
            const uint32_t local_out_row = unit % slice_out_rows;
            const uint32_t in_base = local_out_row * ratio;
            const uint32_t remaining = slice_in_rows - in_base;
            real_rows = remaining < ratio ? remaining : ratio;
            pad_rows = ratio - real_rows;
            out_count = 1;
        }

        if (real_rows > 0) {
            compute_kernel_lib::untilize<
                tiles_per_block,
                src_cb,
                mid_cb,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(real_rows);
        }
        for (uint32_t k = 0; k < pad_rows; ++k) {
            fill_zeros_pages(mid, tiles_per_block, mid_page_size);
        }

        mid.wait_front(block_pages);
        uint32_t block_rd_ptr = 0;
        UNPACK({ block_rd_ptr = get_local_cb_interface(mid_cb).fifo_rd_ptr; })

        // mid_view_cb aliases the mid_cb L1 region but has no producer of its own, and its output
        // tile-rows sit at non-page-aligned byte offsets within the block that pops can't express.
        // So set its fifo_rd_ptr directly to the block base plus each output tile-row's offset.
        // Reconfigure the unpacker/packer from the untilize config (src_cb/mid_cb) to the tilize
        // config (mid_view_cb/out_cb). tilize_init's state_configure is sentinel-only, so the
        // hardware reconfig must be explicit — for bf16 it's a no-op, for bfloat8 it's required.
        reconfig_data_format_srca(src_cb, mid_view_cb);
        pack_reconfig_data_format(mid_cb, out_cb);
        tilize_init(mid_view_cb, tiles_per_block, out_cb);
        for (uint32_t r = 0; r < out_count; ++r) {
            UNPACK({ get_local_cb_interface(mid_view_cb).fifo_rd_ptr = block_rd_ptr + r * words_per_out_tile_row; })
            out_dfb.reserve_back(tiles_per_block);
            tilize_block(mid_view_cb, tiles_per_block, out_cb);
            out_dfb.push_back(tiles_per_block);
        }
        tilize_uninit(mid_view_cb, out_cb);

        mid.pop_front(block_pages);

        reconfig_data_format_srca(mid_view_cb, src_cb);
        pack_reconfig_data_format(out_cb, mid_cb);
    }
}
