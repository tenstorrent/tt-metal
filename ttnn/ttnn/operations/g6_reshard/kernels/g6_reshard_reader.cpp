// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_reshard PULL issuer. One source, one instance per issuer of an out-holding core
// (issuer 0 on RISCV_1/NoC0, issuer 1 on RISCV_0/NoC1). Each instance reads the pieces of
// in0's blocks that overlap its row range of this core's out block straight into the out
// shard, then waits once.
//
// No kernel_lib helper covers this move: the local-copy helpers are self-aimed and
// single-packet, mcast is one-to-many, and noc_async_read_shard / num_contiguous_pages know
// nothing of the destination stride. Raw noc_async_read per maximal run is the operation.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in_block_height_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in_block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_width_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t aligned_page_bytes = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();

    // A tail shard keeps the full shard row stride on both sides, so rows of a piece merge
    // into one run only when the piece spans both full shard widths.
    constexpr bool rows_contiguous_on_both_sides = in_block_width_tiles == block_width_tiles;
    constexpr uint32_t src_row_stride_bytes = in_block_width_tiles * aligned_page_bytes;
    constexpr uint32_t dst_row_stride_bytes = block_width_tiles * aligned_page_bytes;

    const uint32_t src_base_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_shard_addr = get_common_arg_val<uint32_t>(1);

    // Tile coordinates of the padded 2-D view, already clipped to the tensor on the host.
    const uint32_t issuer_row_begin = get_arg_val<uint32_t>(0);
    const uint32_t issuer_row_end = get_arg_val<uint32_t>(1);
    const uint32_t block_row_begin = get_arg_val<uint32_t>(2);
    const uint32_t block_col_begin = get_arg_val<uint32_t>(3);
    const uint32_t block_col_end = get_arg_val<uint32_t>(4);

    {
        DeviceZoneScopedN("RESHARD-ISSUE");
        if (issuer_row_begin < issuer_row_end) {
            const auto src = TensorAccessor(src_args, src_base_addr, aligned_page_bytes);

            const uint32_t first_in_block_row_start =
                (issuer_row_begin / in_block_height_tiles) * in_block_height_tiles;
            const uint32_t first_in_block_col_start = (block_col_begin / in_block_width_tiles) * in_block_width_tiles;

            uint32_t piece_row0 = issuer_row_begin;
            uint32_t next_in_block_row_start = first_in_block_row_start + in_block_height_tiles;
            uint32_t dst_piece_row_addr = out_shard_addr + (issuer_row_begin - block_row_begin) * dst_row_stride_bytes;
            uint32_t src_row_page_id = issuer_row_begin * tensor_width_tiles;

            while (piece_row0 < issuer_row_end) {
                const uint32_t piece_row1 =
                    next_in_block_row_start < issuer_row_end ? next_in_block_row_start : issuer_row_end;
                const uint32_t piece_rows = piece_row1 - piece_row0;

                uint32_t piece_col0 = block_col_begin;
                uint32_t next_in_block_col_start = first_in_block_col_start + in_block_width_tiles;
                uint32_t dst_piece_addr = dst_piece_row_addr;

                while (piece_col0 < block_col_end) {
                    const uint32_t piece_col1 =
                        next_in_block_col_start < block_col_end ? next_in_block_col_start : block_col_end;
                    const uint32_t piece_width = piece_col1 - piece_col0;
                    const uint32_t piece_width_bytes = piece_width * aligned_page_bytes;

                    uint64_t src_noc_addr = src.get_noc_addr(src_row_page_id + piece_col0);
                    if (rows_contiguous_on_both_sides && piece_width == in_block_width_tiles) {
                        noc_async_read(src_noc_addr, dst_piece_addr, piece_rows * piece_width_bytes);
                    } else {
                        uint32_t dst_addr = dst_piece_addr;
                        for (uint32_t r = 0; r < piece_rows; ++r) {
                            noc_async_read(src_noc_addr, dst_addr, piece_width_bytes);
                            src_noc_addr += src_row_stride_bytes;
                            dst_addr += dst_row_stride_bytes;
                        }
                    }

                    dst_piece_addr += piece_width_bytes;
                    piece_col0 = piece_col1;
                    next_in_block_col_start += in_block_width_tiles;
                }

                dst_piece_row_addr += piece_rows * dst_row_stride_bytes;
                src_row_page_id += piece_rows * tensor_width_tiles;
                piece_row0 = piece_row1;
                next_in_block_row_start += in_block_height_tiles;
            }
        }
    }
    {
        DeviceZoneScopedN("RESHARD-WAIT");
        noc_async_read_barrier();
    }
}
