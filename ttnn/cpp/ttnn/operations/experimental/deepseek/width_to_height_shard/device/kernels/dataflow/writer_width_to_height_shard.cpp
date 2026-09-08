// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_untilized = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_core = get_compile_time_arg_val(2);
    constexpr uint32_t output_rows = get_compile_time_arg_val(3);
    constexpr uint32_t shard_width_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t full_width_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t gather_semaphore_id = get_compile_time_arg_val(6);

    const uint32_t sender_id = get_arg_val<uint32_t>(0);
    const uint32_t hub_output_addr = get_arg_val<uint32_t>(1);
    const uint32_t hub_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t hub_noc_y = get_arg_val<uint32_t>(3);

    CircularBuffer input_cb(cb_in);
    CircularBuffer untilized_cb(cb_untilized);

    // The input CB is backed by this core's WIDTH_SHARDED input allocation. Publish the resident
    // tiles so the compute kernel can untilize them in place.
    input_cb.push_back(tiles_per_core);

    untilized_cb.wait_front(tiles_per_core);
    const uint32_t local_untilized_addr = untilized_cb.get_read_ptr();
    const uint32_t shard_column_offset = sender_id * shard_width_bytes;

    // Width shards are contiguous columns. Stitch only the logical rows into the hub's
    // output shard; tile-padding rows from untilize are dropped.
    for (uint32_t row = 0; row < output_rows; ++row) {
        const uint32_t src_addr = local_untilized_addr + row * shard_width_bytes;
        const uint32_t dst_addr = hub_output_addr + row * full_width_bytes + shard_column_offset;
        noc_async_write(src_addr, get_noc_addr(hub_noc_x, hub_noc_y, dst_addr), shard_width_bytes);
    }
    noc_async_write_barrier();

    const uint32_t gather_semaphore_addr = get_semaphore(gather_semaphore_id);
    noc_semaphore_inc(get_noc_addr(hub_noc_x, hub_noc_y, gather_semaphore_addr), 1);
    noc_async_atomic_barrier();

    untilized_cb.pop_front(tiles_per_core);
}
