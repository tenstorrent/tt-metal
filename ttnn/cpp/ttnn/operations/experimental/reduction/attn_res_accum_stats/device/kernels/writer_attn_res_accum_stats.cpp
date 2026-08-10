// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t dots_page_offset = get_compile_time_arg_val(1);
    constexpr auto total_args = TensorAccessorArgs<2>();
    constexpr auto stats_args = TensorAccessorArgs<total_args.next_compile_time_args_offset()>();

    // runtime args
    const auto total_addr = get_arg_val<uint32_t>(0);
    const auto stats_addr = get_arg_val<uint32_t>(1);
    const auto num_rows = get_arg_val<uint32_t>(2);
    const auto start_row = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_stats = 16;
    constexpr uint32_t cb_id_total = 17;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t stats_tile_bytes = get_tile_size(cb_id_stats);
    constexpr uint32_t total_tile_bytes = get_tile_size(cb_id_total);

    Noc noc;
    DataflowBuffer stats_buf(cb_id_stats);
    DataflowBuffer total_buf(cb_id_total);

    auto total_accessor = TensorAccessor(total_args, total_addr);
    auto stats_accessor = TensorAccessor(stats_args, stats_addr);

    // Compute packs the sum first, then the sum of squares, then the dot; draining in
    // that order is what keeps the row's own backpressure from deadlocking against it.
    // The two statistics land a whole candidate axis apart so that the pair arrives at
    // the collective stacked.
    for (uint32_t r = start_row; r < start_row + num_rows; ++r) {
        total_buf.wait_front(Wt);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            noc.async_write(
                total_buf,
                total_accessor,
                total_tile_bytes,
                {.offset_bytes = wt * total_tile_bytes},
                {.page_id = r * Wt + wt});
        }
        noc.async_write_barrier();
        total_buf.pop_front(Wt);

        stats_buf.wait_front(onetile);
        noc.async_write(stats_buf, stats_accessor, stats_tile_bytes, {.offset_bytes = 0}, {.page_id = r});
        noc.async_write_barrier();
        stats_buf.pop_front(onetile);

        stats_buf.wait_front(onetile);
        noc.async_write(
            stats_buf, stats_accessor, stats_tile_bytes, {.offset_bytes = 0}, {.page_id = r + dots_page_offset});
        noc.async_write_barrier();
        stats_buf.pop_front(onetile);
    }
}
