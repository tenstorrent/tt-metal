// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_reshard issuer (PULL). One source, one instance per issuer of an out block
// (issuer 0 on RISCV_1, issuer 1 on RISCV_0). The instance pulls every
// piece of in0 that overlaps its tile rows of this core's out block straight
// into the out shard: no CB, no compute, one read barrier per NoC it reads on.
//
// No kernel_lib helper is used: the local-copy helpers are self-aimed and
// single-packet only, while pieces come from remote cores and full-width runs
// exceed one packet. noc_async_read splits a run into packets itself.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in_block_height_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in_block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_width_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t aligned_page_bytes = get_compile_time_arg_val(4);
    // The issuer's sub-block is cut into at most this many rectangles, each read on its own NoC.
    constexpr uint32_t max_segments = get_compile_time_arg_val(5);
    constexpr auto in_args = TensorAccessorArgs<6>();

    // Slot stride is always the shard width, also inside a cut-short tail shard.
    constexpr uint32_t in_row_stride_bytes = in_block_width_tiles * aligned_page_bytes;
    constexpr uint32_t out_row_stride_bytes = block_width_tiles * aligned_page_bytes;
    // Rows are contiguous on both sides only when a piece spans both shard widths.
    constexpr bool shard_widths_match = in_block_width_tiles == block_width_tiles;

    // Per segment: tile rows [begin, end), tile columns [begin, end), NoC.
    constexpr uint32_t args_per_segment = 5;
    constexpr uint32_t first_segment_arg = 1;

    // NoCs are chosen on the host from a route model of the placement pair (NoC0 carries data
    // east/south, NoC1 west/north). Any NoC other than the configured one makes the host build
    // the kernel in dynamic NoC mode.
    uint8_t first_noc;
    uint8_t second_noc;
    bool second_noc_used = false;

    {
        // Argument loads and accessor setup are per-block fixed cost: kept inside the zone (BB-21).
        DeviceZoneScopedN("RESHARD-ISSUE");
        const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
        const uint32_t out_addr = get_common_arg_val<uint32_t>(1);

        // Origin of this core's out block; segment 0 starts at its first column.
        const uint32_t block_row_begin = get_arg_val<uint32_t>(0);
        const uint32_t block_col_begin = get_arg_val<uint32_t>(first_segment_arg + 2);
        first_noc = static_cast<uint8_t>(get_arg_val<uint32_t>(first_segment_arg + 4));
        second_noc = first_noc;

        const auto in_accessor = TensorAccessor(in_args, in_addr, aligned_page_bytes);

        // One copy of the walk, run once per segment: a second pass executes warm code.
        for (uint32_t segment = 0; segment < max_segments; ++segment) {
            const uint32_t arg = first_segment_arg + segment * args_per_segment;
            // This segment's rectangle of the padded 2-D view, already clipped to the tensor.
            const uint32_t row_begin = get_arg_val<uint32_t>(arg);
            const uint32_t row_end = get_arg_val<uint32_t>(arg + 1);
            const uint32_t col_begin = get_arg_val<uint32_t>(arg + 2);
            const uint32_t col_end = get_arg_val<uint32_t>(arg + 3);
            const uint8_t noc = static_cast<uint8_t>(get_arg_val<uint32_t>(arg + 4));
            if (row_begin >= row_end) {
                continue;
            }
            if (segment > 0) {
                second_noc = noc;
                second_noc_used = true;
            }

            const uint32_t first_in_block_col = col_begin / in_block_width_tiles;
            const uint32_t first_col_edge = (first_in_block_col + 1) * in_block_width_tiles;

            uint32_t piece_row0 = row_begin;
            uint32_t row_edge = (row_begin / in_block_height_tiles + 1) * in_block_height_tiles;
            while (piece_row0 < row_end) {
                const uint32_t piece_row_end = row_edge < row_end ? row_edge : row_end;
                const uint32_t piece_rows = piece_row_end - piece_row0;
                const uint32_t out_row_addr = out_addr + (piece_row0 - block_row_begin) * out_row_stride_bytes;
                const uint32_t page_row0 = piece_row0 * tensor_width_tiles;

                uint32_t piece_col0 = col_begin;
                uint32_t col_edge = first_col_edge;
                while (piece_col0 < col_end) {
                    const uint32_t piece_col_end = col_edge < col_end ? col_edge : col_end;
                    const uint32_t piece_width = piece_col_end - piece_col0;

                    uint64_t src = in_accessor.get_noc_addr(page_row0 + piece_col0, 0, noc);
                    uint32_t dst = out_row_addr + (piece_col0 - block_col_begin) * aligned_page_bytes;

                    if (shard_widths_match && piece_width == in_block_width_tiles) {
                        noc_async_read(src, dst, piece_rows * piece_width * aligned_page_bytes, noc);
                    } else {
                        const uint32_t run_bytes = piece_width * aligned_page_bytes;
                        for (uint32_t i = 0; i < piece_rows; ++i) {
                            noc_async_read(src, dst, run_bytes, noc);
                            src += in_row_stride_bytes;
                            dst += out_row_stride_bytes;
                        }
                    }

                    piece_col0 = piece_col_end;
                    col_edge += in_block_width_tiles;
                }

                piece_row0 = piece_row_end;
                row_edge += in_block_height_tiles;
            }
        }
    }

    {
        DeviceZoneScopedN("RESHARD-WAIT");
        noc_async_read_barrier(first_noc);
        if (second_noc_used && second_noc != first_noc) {
            noc_async_read_barrier(second_noc);
        }
    }
}
