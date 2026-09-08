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
// Two paths, selected at compile time via num_width_chunks:
//   * num_width_chunks == 1 — legacy streaming path. Used by the interleaved retile factory (small
//     double-buffered src_cb/out_cb with streaming reader/writer) and by any sharded shard whose
//     width already fits under the chunk cap. Sequential per-row untilize/tilize, no pointer
//     surgery; the CB helpers advance src_cb / out_cb naturally.
//   * num_width_chunks > 1 — chunked path. Used only by the sharded retile factory when the shard
//     width exceeds the byte cap. src_cb and out_cb are aliased zero-copy to the full-shard L1
//     buffers; the kernel processes `num_width_chunks` sub-blocks of `chunk_tiles` tiles each and
//     manually seeks fifo_rd_ptr / fifo_wr_ptr per (chunk, tile-row) so the ordered CB helpers
//     land at the correct L1 addresses. Total pop_front(src) / push_back(out) sums match the
//     legacy path exactly.

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

    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t src_cb = get_compile_time_arg_val(1);
    constexpr uint32_t mid_cb = get_compile_time_arg_val(2);
    constexpr uint32_t mid_view_cb = get_compile_time_arg_val(3);
    constexpr uint32_t out_cb = get_compile_time_arg_val(4);
    constexpr uint32_t in_tile_height = get_compile_time_arg_val(5);
    constexpr uint32_t out_tile_height = get_compile_time_arg_val(6);
    constexpr uint32_t mid_out_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t mid_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(9);
    constexpr uint32_t input_tile_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t output_tile_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t num_width_chunks = get_compile_time_arg_val(12);

    static_assert(in_tile_height > 0 && out_tile_height > 0, "retile kernel requires positive tile heights");
    static_assert(
        (in_tile_height >= out_tile_height && (in_tile_height % out_tile_height) == 0) ||
            (out_tile_height > in_tile_height && (out_tile_height % in_tile_height) == 0),
        "retile kernel requires one tile height to divide the other exactly");
    static_assert(chunk_tiles > 0, "retile kernel requires positive chunk_tiles");
    static_assert(num_width_chunks > 0, "retile kernel requires positive num_width_chunks");
    static_assert(
        chunk_tiles * num_width_chunks == tiles_per_block,
        "retile kernel requires chunk_tiles * num_width_chunks == tiles_per_block");

    constexpr uint32_t words_per_out_tile_row = (chunk_tiles * mid_out_page_size) >> cb_addr_shift;

    compute_kernel_hw_startup(src_cb, mid_cb);

    DataflowBuffer mid(mid_cb);
    DataflowBuffer out_dfb(out_cb);

    if constexpr (num_width_chunks == 1) {
        // ================================================================
        // Legacy streaming path: no width chunking, no pointer surgery.
        // Matches the pre-chunking behaviour byte-for-byte; used by the
        // interleaved retile factory and by sharded shards whose full width
        // already fits under the chunk cap.
        // ================================================================
        if (num_real_input_rows > 0) {
            // One tile-row at a time: pack_untilize of a full shard can exceed dest capacity
            // for wide sharded blocks, and InitAndUninit per row matches the previously working
            // path.
            for (uint32_t row = 0; row < num_real_input_rows; ++row) {
                compute_kernel_lib::untilize<
                    chunk_tiles,
                    src_cb,
                    mid_cb,
                    compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                    compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                    compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
            }
        }
        const uint32_t pad_rows = num_input_blocks > num_real_input_rows ? (num_input_blocks - num_real_input_rows) : 0;
        for (uint32_t k = 0; k < pad_rows; ++k) {
            fill_zeros_pages(mid, chunk_tiles, mid_page_size);
        }

        const uint32_t mid_pages = num_input_blocks * chunk_tiles;
        mid.wait_front(mid_pages);
        uint32_t block_rd_ptr = 0;
        UNPACK({ block_rd_ptr = get_local_cb_interface(mid_cb).fifo_rd_ptr; })

        // Reconfigure the unpacker/packer from the untilize config (src_cb/mid_cb) to the tilize
        // config (mid_view_cb/out_cb). tilize_init's state_configure is sentinel-only, so the
        // hardware reconfig must be explicit — for bf16 it's a no-op, for bfloat8 it's required.
        reconfig_data_format_srca(src_cb, mid_view_cb);
        pack_reconfig_data_format(mid_cb, out_cb);
        tilize_init(mid_view_cb, chunk_tiles, out_cb);
        for (uint32_t r = 0; r < num_real_output_rows; ++r) {
            UNPACK({ get_local_cb_interface(mid_view_cb).fifo_rd_ptr = block_rd_ptr + r * words_per_out_tile_row; })
            out_dfb.reserve_back(chunk_tiles);
            tilize_block(mid_view_cb, chunk_tiles, out_cb);
            out_dfb.push_back(chunk_tiles);
        }
        tilize_uninit(mid_view_cb, out_cb);

        mid.pop_front(mid_pages);
        return;
    }

    // ====================================================================
    // Chunked path: sharded output only. Manual per-(chunk, row) pointer
    // surgery on the aliased src_cb / out_cb so the ordered helpers land at
    // the correct L1 addresses inside the full-shard buffer.
    // ====================================================================

    // Address-unit strides. CB fifo_rd_ptr / fifo_wr_ptr are stored in cb_addr_shift units
    // (16 B on WH/BH TRISC), matching what the mid_view seeking above relies on.
    constexpr uint32_t input_tile_stride = input_tile_bytes >> cb_addr_shift;
    constexpr uint32_t output_tile_stride = output_tile_bytes >> cb_addr_shift;
    constexpr uint32_t chunk_input_stride = chunk_tiles * input_tile_stride;
    constexpr uint32_t chunk_output_stride = chunk_tiles * output_tile_stride;
    constexpr uint32_t row_input_stride = tiles_per_block * input_tile_stride;
    constexpr uint32_t row_output_stride = tiles_per_block * output_tile_stride;

    DataflowBuffer src(src_cb);

    // Wait once for the entire real input span and capture the aliased base pointers before any
    // per-chunk seeks. The reader (reader_unary_sharded) pushes the whole shard up front, and the
    // sharded writer only does a readiness wait_front, so both src and out can safely be driven
    // via manual seeks + a single sync at the end.
    const uint32_t total_src_tiles = num_real_input_rows * tiles_per_block;
    src.wait_front(total_src_tiles);

    uint32_t src_base_rd_ptr = 0;
    UNPACK({ src_base_rd_ptr = get_local_cb_interface(src_cb).fifo_rd_ptr; })
    uint32_t out_base_wr_ptr = 0;
    PACK({ out_base_wr_ptr = get_local_cb_interface(out_cb).fifo_wr_ptr; })

    const uint32_t pad_rows = num_input_blocks > num_real_input_rows ? (num_input_blocks - num_real_input_rows) : 0;
    const uint32_t chunk_mid_pages = num_input_blocks * chunk_tiles;

    for (uint32_t c = 0; c < num_width_chunks; ++c) {
        const uint32_t chunk_src_offset = c * chunk_input_stride;
        const uint32_t chunk_out_offset = c * chunk_output_stride;

        // 1. Untilize this width-chunk for every real input tile-row into mid.
        for (uint32_t row = 0; row < num_real_input_rows; ++row) {
            UNPACK({
                get_local_cb_interface(src_cb).fifo_rd_ptr =
                    src_base_rd_ptr + row * row_input_stride + chunk_src_offset;
            })
            compute_kernel_lib::untilize<
                chunk_tiles,
                src_cb,
                mid_cb,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::NoWait,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
        }

        // Zero-fill any padding rows (grow-case: input has fewer real tile-rows than mid holds).
        for (uint32_t k = 0; k < pad_rows; ++k) {
            fill_zeros_pages(mid, chunk_tiles, mid_page_size);
        }

        // 2. Tilize the chunk's mid contents into the aliased out_cb.
        mid.wait_front(chunk_mid_pages);
        uint32_t block_rd_ptr = 0;
        UNPACK({ block_rd_ptr = get_local_cb_interface(mid_cb).fifo_rd_ptr; })

        reconfig_data_format_srca(src_cb, mid_view_cb);
        pack_reconfig_data_format(mid_cb, out_cb);
        tilize_init(mid_view_cb, chunk_tiles, out_cb);

        for (uint32_t r = 0; r < num_real_output_rows; ++r) {
            UNPACK({ get_local_cb_interface(mid_view_cb).fifo_rd_ptr = block_rd_ptr + r * words_per_out_tile_row; })
            PACK({
                get_local_cb_interface(out_cb).fifo_wr_ptr = out_base_wr_ptr + r * row_output_stride + chunk_out_offset;
            })
            out_dfb.reserve_back(chunk_tiles);
            tilize_block(mid_view_cb, chunk_tiles, out_cb);
            out_dfb.push_back(chunk_tiles);
        }
        tilize_uninit(mid_view_cb, out_cb);

        mid.pop_front(chunk_mid_pages);

        // Reconfigure back to the untilize direction for the next chunk's first untilize call.
        // The untilize helper uses NoReconfigure, so the srcA reconfig has to happen here.
        if (c + 1 < num_width_chunks) {
            reconfig_data_format_srca(mid_view_cb, src_cb);
            pack_reconfig_data_format(out_cb, mid_cb);
        }
    }

    // Retire the full real input span in one shot; total push_back(out) across chunks/rows above
    // equals num_real_output_rows * tiles_per_block, matching the legacy path's writer contract.
    src.pop_front(total_src_tiles);
}
