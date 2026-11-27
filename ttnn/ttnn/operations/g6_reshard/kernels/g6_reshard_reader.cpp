// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_reshard PULL issuer. One source, instantiated once per issuer of an out-holding core
// (issuer 0: RISCV_1 / NoC0, issuer 1: RISCV_0 / NoC1). Each issuer owns a contiguous range of
// tile rows of its core's out block and the block's full column range. It walks the in0 blocks
// overlapping that sub-block on device, reads every piece straight into the out shard as maximal
// runs (one noc_async_read per run), then waits once.
//
// No CB and no compute kernel: reads land in the out shard by address. Raw noc_async_read is used
// because no kernel_lib helper covers a remote, multi-packet, strided piece read (op_design.md,
// "Helpers considered and rejected").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t in_block_height_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in_block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_width_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t aligned_page_bytes = get_compile_time_arg_val(4);
    constexpr auto in_args = TensorAccessorArgs<5>();

    // Slot strides are the full shard widths: a cut-short tail shard keeps them.
    constexpr uint32_t in_row_stride_bytes = in_block_width_tiles * aligned_page_bytes;
    constexpr uint32_t out_row_stride_bytes = block_width_tiles * aligned_page_bytes;

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);

    // Issuer sub-block, in tiles of the padded 2-D view; row and column ends are already clipped
    // to the tensor edge.
    const uint32_t issuer_row_begin = get_arg_val<uint32_t>(0);
    const uint32_t issuer_row_end = get_arg_val<uint32_t>(1);
    const uint32_t block_row_begin = get_arg_val<uint32_t>(2);
    const uint32_t block_col_begin = get_arg_val<uint32_t>(3);
    const uint32_t block_col_end = get_arg_val<uint32_t>(4);

    {
        DeviceZoneScopedN("g6_reshard_issue");
        const auto in_accessor = TensorAccessor(in_args, in_addr, aligned_page_bytes);

        if (issuer_row_begin < issuer_row_end) {
            const uint32_t in_row_first = issuer_row_begin / in_block_height_tiles;
            const uint32_t in_row_last = (issuer_row_end - 1) / in_block_height_tiles;
            const uint32_t in_col_first = block_col_begin / in_block_width_tiles;
            const uint32_t in_col_last = (block_col_end - 1) / in_block_width_tiles;

            uint32_t in_row_origin = in_row_first * in_block_height_tiles;
            for (uint32_t in_row = in_row_first; in_row <= in_row_last; ++in_row) {
                const uint32_t in_row_next = in_row_origin + in_block_height_tiles;
                const uint32_t piece_row0 = in_row_origin > issuer_row_begin ? in_row_origin : issuer_row_begin;
                const uint32_t piece_row1 = in_row_next < issuer_row_end ? in_row_next : issuer_row_end;
                const uint32_t piece_rows = piece_row1 - piece_row0;
                const uint32_t dst_row_base = out_addr + (piece_row0 - block_row_begin) * out_row_stride_bytes;

                uint32_t in_col_origin = in_col_first * in_block_width_tiles;
                for (uint32_t in_col = in_col_first; in_col <= in_col_last; ++in_col) {
                    const uint32_t in_col_next = in_col_origin + in_block_width_tiles;
                    const uint32_t piece_col0 = in_col_origin > block_col_begin ? in_col_origin : block_col_begin;
                    const uint32_t piece_col1 = in_col_next < block_col_end ? in_col_next : block_col_end;
                    const uint32_t piece_width = piece_col1 - piece_col0;
                    const uint32_t run_bytes = piece_width * aligned_page_bytes;

                    uint64_t src = in_accessor.get_noc_addr(piece_row0 * tensor_width_tiles + piece_col0);
                    uint32_t dst = dst_row_base + (piece_col0 - block_col_begin) * aligned_page_bytes;

                    // Rows are contiguous on both sides only when the piece spans both full shard
                    // widths; valid (tail) widths never qualify.
                    if (piece_width == in_block_width_tiles && piece_width == block_width_tiles) {
                        noc_async_read(src, dst, piece_rows * run_bytes);
                    } else {
                        for (uint32_t r = 0; r < piece_rows; ++r) {
                            noc_async_read(src, dst, run_bytes);
                            src += in_row_stride_bytes;
                            dst += out_row_stride_bytes;
                        }
                    }
                    in_col_origin = in_col_next;
                }
                in_row_origin = in_row_next;
            }
        }
    }

    {
        DeviceZoneScopedN("g6_reshard_wait");
        noc_async_read_barrier();
    }
}
