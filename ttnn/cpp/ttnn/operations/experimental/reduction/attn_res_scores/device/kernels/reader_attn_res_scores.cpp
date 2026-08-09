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
    constexpr auto stats_args = TensorAccessorArgs<1>();

    // runtime args
    const auto stats_addr = get_arg_val<uint32_t>(0);
    const auto num_output_tiles = get_arg_val<uint32_t>(1);
    const auto start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_stats = 0;
    constexpr uint32_t stats_tile_bytes = get_tile_size(cb_id_stats);
    constexpr uint32_t kOperands = 2;

    Noc noc;
    CircularBuffer cb_stats(cb_id_stats);

    auto stats_accessor = TensorAccessor(stats_args, stats_addr);

    // Output page `i` is candidate c of the score tensor, whose sum of squares is
    // input page `i` and whose dot is `dots_page_offset` pages further on. The two
    // go into one CB as a pair so compute reads both through a single unpack
    // configuration.
    for (uint32_t i = start_id; i < start_id + num_output_tiles; ++i) {
        cb_stats.reserve_back(kOperands);
        noc.async_read(stats_accessor, cb_stats, stats_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(
            stats_accessor,
            cb_stats,
            stats_tile_bytes,
            {.page_id = i + dots_page_offset},
            {.offset_bytes = stats_tile_bytes});
        noc.async_read_barrier();
        cb_stats.push_back(kOperands);
    }
}
