// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

// HEIGHT_SHARDED single-core input: the full tensor already lives on this core.
// Untilize if needed, then place the contiguous M x K replica in the hub's output shard.
// When this core is the hub it also multicasts the replica to the rest of the output grid;
// otherwise the hub's receiver kernel does, so the multicast always originates inside the
// destination rectangle.
void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_untilized = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_core = get_compile_time_arg_val(2);
    constexpr uint32_t skip_untilize = get_compile_time_arg_val(3);
    constexpr uint32_t num_slices = get_compile_time_arg_val(4);
    constexpr uint32_t rows_per_slice = get_compile_time_arg_val(5);
    constexpr uint32_t padded_rows_per_slice = get_compile_time_arg_val(6);
    constexpr uint32_t full_width_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t slice_bytes = rows_per_slice * full_width_bytes;
    constexpr uint32_t padded_slice_bytes = padded_rows_per_slice * full_width_bytes;
    constexpr uint32_t full_tensor_bytes = num_slices * slice_bytes;
    constexpr uint32_t bbox_num_cores = get_compile_time_arg_val(8);
    uint32_t mcast_start_x = get_compile_time_arg_val(9);
    uint32_t mcast_start_y = get_compile_time_arg_val(10);
    uint32_t mcast_end_x = get_compile_time_arg_val(11);
    uint32_t mcast_end_y = get_compile_time_arg_val(12);
    constexpr uint32_t is_hub = get_compile_time_arg_val(13);
    constexpr uint32_t gather_semaphore_id = get_compile_time_arg_val(14);

    const uint32_t hub_output_addr = get_arg_val<uint32_t>(0);
    const uint32_t hub_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t hub_noc_y = get_arg_val<uint32_t>(2);

    CircularBuffer input_cb(cb_in);
    uint32_t src_addr;
    if constexpr (skip_untilize) {
        // ROW_MAJOR shards are stored as one page per logical row.
        input_cb.push_back(num_slices * padded_rows_per_slice);
        src_addr = input_cb.get_read_ptr();
    } else {
        CircularBuffer untilized_cb(cb_untilized);
        input_cb.push_back(tiles_per_core);
        untilized_cb.wait_front(tiles_per_core);
        src_addr = untilized_cb.get_read_ptr();
    }

    // Each leading-dim slice holds its logical rows contiguously, followed by tile padding
    // when the source was TILE. Copy one slice at a time to drop the padding; a dense source
    // goes out as a single transfer.
    if constexpr (rows_per_slice == padded_rows_per_slice) {
        noc_async_write(src_addr, get_noc_addr(hub_noc_x, hub_noc_y, hub_output_addr), full_tensor_bytes);
    } else {
        for (uint32_t slice = 0; slice < num_slices; ++slice) {
            noc_async_write(
                src_addr + slice * padded_slice_bytes,
                get_noc_addr(hub_noc_x, hub_noc_y, hub_output_addr + slice * slice_bytes),
                slice_bytes);
        }
    }
    noc_async_write_barrier();

    if constexpr (is_hub) {
        if constexpr (bbox_num_cores > 1) {
            if (noc_index == 1) {
                std::swap(mcast_start_x, mcast_end_x);
                std::swap(mcast_start_y, mcast_end_y);
            }
            const uint64_t mcast_addr =
                get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, hub_output_addr);
            // We already own the assembled output shard, so exclude ourselves from the multicast.
            noc_async_write_multicast(hub_output_addr, mcast_addr, full_tensor_bytes, bbox_num_cores - 1);
            noc_async_write_barrier();
        }
    } else {
        noc_semaphore_inc(get_noc_addr(hub_noc_x, hub_noc_y, get_semaphore(gather_semaphore_id)), 1);
        noc_async_atomic_barrier();
    }
}
