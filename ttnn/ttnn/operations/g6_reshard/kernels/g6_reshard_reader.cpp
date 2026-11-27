// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PULL reshard issuer. The host launches this one source once per issuer of every out-holding
// core (issuer 0: RISCV_1 / NoC0, issuer 1: RISCV_0 / NoC1). An issuer owns the tile rows
// [row_begin, row_end) x [block_col_begin, block_col_end) of its core's out block and reads each
// piece of in0 that overlaps them straight into the out shard: no CB, no compute, no conversion.
//
// Raw noc_async_read / noc_async_read_barrier instead of a kernel_lib helper:
// local_copy_helpers_dataflow.hpp is self-aimed and single-packet only, while pieces come from
// remote cores and a merged run can exceed one NoC packet (16 tiles = 17408 B for bfloat8_b).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in_block_height_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in_block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_width_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t aligned_page_bytes = get_compile_time_arg_val(4);
    constexpr auto in_accessor_args = TensorAccessorArgs<5>();

    // Shards keep their full row stride even when cut short at the tensor edge.
    constexpr uint32_t in_row_stride_bytes = in_block_width_tiles * aligned_page_bytes;
    constexpr uint32_t out_row_stride_bytes = block_width_tiles * aligned_page_bytes;
    // Rows of a piece are contiguous on both sides only when both shard widths equal the piece width.
    constexpr bool shard_widths_match = in_block_width_tiles == block_width_tiles;

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);

    const uint32_t block_row_begin = get_arg_val<uint32_t>(0);
    const uint32_t block_col_begin = get_arg_val<uint32_t>(1);
    const uint32_t block_col_end = get_arg_val<uint32_t>(2);
    const uint32_t row_begin = get_arg_val<uint32_t>(3);
    const uint32_t row_end = get_arg_val<uint32_t>(4);

    {
        DeviceZoneScopedN("g6_reshard_issue");
        const auto in_accessor = TensorAccessor(in_accessor_args, in_addr, aligned_page_bytes);

        // An issuer with no rows (a one-row block split two ways) issues nothing but still waits below.
        if (row_begin < row_end) {
            // locate_pieces: first in0 block row and column that overlap this issuer's sub-block.
            const uint32_t first_in_row = (row_begin / in_block_height_tiles) * in_block_height_tiles;
            const uint32_t first_in_col = (block_col_begin / in_block_width_tiles) * in_block_width_tiles;

            for (uint32_t in_row0 = first_in_row; in_row0 < row_end; in_row0 += in_block_height_tiles) {
                const uint32_t in_row_end = in_row0 + in_block_height_tiles;
                const uint32_t piece_row_begin = in_row0 > row_begin ? in_row0 : row_begin;
                const uint32_t piece_row_end = in_row_end < row_end ? in_row_end : row_end;
                const uint32_t piece_rows = piece_row_end - piece_row_begin;
                const uint32_t out_row_slot = (piece_row_begin - block_row_begin) * block_width_tiles;

                for (uint32_t in_col0 = first_in_col; in_col0 < block_col_end; in_col0 += in_block_width_tiles) {
                    const uint32_t in_col_end = in_col0 + in_block_width_tiles;
                    const uint32_t piece_col_begin = in_col0 > block_col_begin ? in_col0 : block_col_begin;
                    const uint32_t piece_col_end = in_col_end < block_col_end ? in_col_end : block_col_end;
                    const uint32_t piece_width = piece_col_end - piece_col_begin;
                    const uint32_t row_bytes = piece_width * aligned_page_bytes;

                    // move_pieces: one source resolution per piece, one NoC read per maximal run.
                    uint64_t src_noc_addr =
                        in_accessor.get_noc_addr(piece_row_begin * tensor_width_tiles + piece_col_begin);
                    uint32_t dst_l1_addr =
                        out_addr + (out_row_slot + (piece_col_begin - block_col_begin)) * aligned_page_bytes;

                    if (shard_widths_match && piece_width == in_block_width_tiles) {
                        noc_async_read(src_noc_addr, dst_l1_addr, piece_rows * row_bytes);
                    } else {
                        for (uint32_t r = 0; r < piece_rows; ++r) {
                            noc_async_read(src_noc_addr, dst_l1_addr, row_bytes);
                            src_noc_addr += in_row_stride_bytes;
                            dst_l1_addr += out_row_stride_bytes;
                        }
                    }
                }
            }
        }
    }

    {
        DeviceZoneScopedN("g6_reshard_wait");
        // complete_block: nothing may be left in flight when the next program starts.
        noc_async_read_barrier();
    }
}
