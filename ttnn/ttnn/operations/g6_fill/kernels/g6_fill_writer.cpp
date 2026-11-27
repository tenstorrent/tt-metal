// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_fill writer (BRISC, NoC1): regime R1, tile row-0 segment scatter.
//
// The only logical positions of a rank-1 TILE tensor are row 0 of each tile, which the
// face layout splits into `segments_per_tile` runs of `segment_bytes`, one per top face,
// `face_bytes` apart. Every run receives the same constant, so the core builds one
// segment of `e` in cb_fill_row and sources every write from it.
//
// Raw dataflow API, no kernel_lib helper: no helper writes one constant L1 run to a set of
// (page, in-page offset) destinations of an interleaved tensor (op_design.md, API Mapping).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_fill_row = get_compile_time_arg_val(0);
    constexpr uint32_t segment_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t segments_per_tile = get_compile_time_arg_val(3);
    constexpr uint32_t block_w_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t page_size = get_compile_time_arg_val(5);
    constexpr auto dst_args = TensorAccessorArgs<6>();

    static_assert(segment_bytes % sizeof(uint32_t) == 0, "a segment is whole 32-bit elements");
    static_assert(segment_bytes <= NOC_MAX_BURST_SIZE, "one-packet write must fit one NoC burst");
    static_assert(face_bytes % 16 == 0, "BH DRAM writes need L1 and DRAM offsets equal mod 16");
    static_assert(block_w_tiles > 0, "block extent must be positive");
    constexpr uint32_t segment_words = segment_bytes / sizeof(uint32_t);

    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t fill_bits = get_common_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(0);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(1);

    // Single-thread producer and consumer: no reserve/push/wait/pop. The row is complete
    // before the first write reads it over the NoC.
    const uint32_t fill_row_addr = get_write_ptr(cb_fill_row);
    {
        DeviceZoneScopedN("G6FILL-BUILD-ROW");
        volatile tt_l1_ptr uint32_t* fill_row = dataflow_kernel_lib::addr_to_l1_ptr(fill_row_addr);
        for (uint32_t i = 0; i < segment_words; ++i) {
            fill_row[i] = fill_bits;
        }
    }

    {
        DeviceZoneScopedN("G6FILL-ISSUE");
        const auto dst = TensorAccessor(dst_args, dst_addr, page_size);
        const uint32_t end_tile = start_tile + core_w_tiles;
        for (uint32_t block_start = start_tile; block_start < end_tile; block_start += block_w_tiles) {
            const uint32_t block_end = block_start + block_w_tiles < end_tile ? block_start + block_w_tiles : end_tile;
            for (uint32_t tile = block_start; tile < block_end; ++tile) {
                const uint64_t tile_noc_addr = dst.get_noc_addr(tile);
                for (uint32_t s = 0; s < segments_per_tile; ++s) {
                    noc_async_write_one_packet(fill_row_addr, tile_noc_addr + s * face_bytes, segment_bytes);
                }
            }
        }
    }

    {
        // Every write must be acked before return: the next program starts right after (BB-03).
        DeviceZoneScopedN("G6FILL-DRAIN");
        noc_async_write_barrier();
    }
}
