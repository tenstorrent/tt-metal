// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_reshard issuer (PULL). One source, one instance per data-movement RISC-V of each out-block holder.
// The instance owns rows [row_begin, row_end) x cols [col_begin, col_end) of the out block (tiles of the padded
// 2-D view), reads every overlapping piece of in0's blocks straight into out's shard, then waits once.
//
// Raw noc_async_read / noc_async_read_barrier: no kernel_lib helper reads from a remote core into a strided
// destination with multi-packet runs (local_copy_helpers are self-aimed and single-packet, mcast is one-to-many).
// The source address comes from in0's TensorAccessor, so any legal placement of in0 is served.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

// Shard slots keep the full shard row stride even in a cut-short tail shard, so rows of a piece merge into one
// run only when the piece spans the whole shard row on both sides.
template <uint32_t in_block_width_tiles, uint32_t block_width_tiles, uint32_t aligned_page_bytes>
FORCE_INLINE void move_piece(uint64_t src, uint32_t dst, uint32_t piece_rows, uint32_t piece_width) {
    constexpr uint32_t in_row_stride_bytes = in_block_width_tiles * aligned_page_bytes;
    constexpr uint32_t out_row_stride_bytes = block_width_tiles * aligned_page_bytes;
    if (in_block_width_tiles == block_width_tiles && piece_width == block_width_tiles) {
        noc_async_read(src, dst, piece_rows * out_row_stride_bytes);
        return;
    }
    const uint32_t run_bytes = piece_width * aligned_page_bytes;
    for (uint32_t r = 0; r < piece_rows; ++r) {
        noc_async_read(src, dst, run_bytes);
        src += in_row_stride_bytes;
        dst += out_row_stride_bytes;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t tensor_width_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t in_block_height_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in_block_width_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t aligned_page_bytes = get_compile_time_arg_val(4);
    constexpr auto in0_args = TensorAccessorArgs<5>();

    const uint32_t in0_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);

    const uint32_t block_row_begin = get_arg_val<uint32_t>(0);
    const uint32_t row_begin = get_arg_val<uint32_t>(1);
    const uint32_t row_end = get_arg_val<uint32_t>(2);
    const uint32_t col_begin = get_arg_val<uint32_t>(3);
    const uint32_t col_end = get_arg_val<uint32_t>(4);

    constexpr uint32_t out_row_stride_bytes = block_width_tiles * aligned_page_bytes;

    {
        DeviceZoneScopedN("g6_reshard_issue");
        if (row_begin < row_end) {
            const auto in0 = TensorAccessor(in0_args, in0_addr, aligned_page_bytes);

            // locate_pieces: first in0 block boundary past the sub-block's origin on each axis; the walk then
            // advances by whole in0 blocks.
            const uint32_t first_col_boundary = (col_begin / in_block_width_tiles + 1) * in_block_width_tiles;
            uint32_t row_boundary = (row_begin / in_block_height_tiles + 1) * in_block_height_tiles;
            uint32_t piece_row0 = row_begin;
            uint32_t dst_row = out_addr + (row_begin - block_row_begin) * out_row_stride_bytes;
            uint32_t src_row_page = row_begin * tensor_width_tiles;

            // move_pieces
            while (piece_row0 < row_end) {
                const uint32_t piece_row1 = row_boundary < row_end ? row_boundary : row_end;
                const uint32_t piece_rows = piece_row1 - piece_row0;

                uint32_t col_boundary = first_col_boundary;
                uint32_t piece_col0 = col_begin;
                uint32_t dst = dst_row;
                while (piece_col0 < col_end) {
                    const uint32_t piece_col1 = col_boundary < col_end ? col_boundary : col_end;
                    const uint32_t piece_width = piece_col1 - piece_col0;
                    move_piece<in_block_width_tiles, block_width_tiles, aligned_page_bytes>(
                        in0.get_noc_addr(src_row_page + piece_col0), dst, piece_rows, piece_width);
                    dst += piece_width * aligned_page_bytes;
                    piece_col0 = piece_col1;
                    col_boundary += in_block_width_tiles;
                }

                dst_row += piece_rows * out_row_stride_bytes;
                src_row_page += piece_rows * tensor_width_tiles;
                piece_row0 = piece_row1;
                row_boundary += in_block_height_tiles;
            }
        }
    }

    {
        // complete_block: nothing may stay in flight past return (BB-03).
        DeviceZoneScopedN("g6_reshard_wait");
        noc_async_read_barrier();
    }
}
