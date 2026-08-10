// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t dots_page_offset = get_compile_time_arg_val(0);
    constexpr uint32_t num_partials = get_compile_time_arg_val(1);
    constexpr uint32_t partial_page_stride = get_compile_time_arg_val(2);
    constexpr auto stats_args = TensorAccessorArgs<3>();

    // runtime args
    const auto stats_addr = get_arg_val<uint32_t>(0);
    const auto num_output_tiles = get_arg_val<uint32_t>(1);
    const auto start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_stats = 0;
    constexpr uint32_t stats_tile_bytes = get_tile_size(cb_id_stats);
    constexpr uint32_t kOperands = 2;
    constexpr uint32_t tiles_per_candidate = kOperands * num_partials;

    Noc noc;
    CircularBuffer cb_stats(cb_id_stats);

    auto stats_accessor = TensorAccessor(stats_args, stats_addr);

    // Output page `i` is candidate c of the score tensor, whose sum of squares is
    // input page `i` and whose dot is `dots_page_offset` pages further on. The two
    // go into one CB as a pair so compute reads both through a single unpack
    // configuration.
    //
    // A gathering collective stacks each rank's pair on dim 1, so rank r repeats
    // that layout `partial_page_stride` pages on. The pairs are pushed rank-major
    // and compute sums across them.
    for (uint32_t i = start_id; i < start_id + num_output_tiles; ++i) {
        cb_stats.reserve_back(tiles_per_candidate);
        uint32_t page = i;
        uint32_t offset_bytes = 0;
        for (uint32_t p = 0; p < num_partials; ++p) {
            noc.async_read(
                stats_accessor, cb_stats, stats_tile_bytes, {.page_id = page}, {.offset_bytes = offset_bytes});
            noc.async_read(
                stats_accessor,
                cb_stats,
                stats_tile_bytes,
                {.page_id = page + dots_page_offset},
                {.offset_bytes = offset_bytes + stats_tile_bytes});
            page += partial_page_stride;
            offset_bytes += kOperands * stats_tile_bytes;
        }
        noc.async_read_barrier();
        cb_stats.push_back(tiles_per_candidate);
    }
}
