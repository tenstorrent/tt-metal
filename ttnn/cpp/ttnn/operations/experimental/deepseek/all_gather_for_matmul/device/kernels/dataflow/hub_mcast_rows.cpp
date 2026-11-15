// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"

// Row tiles gathered on a hub that holds no shard of its own: once every sender's shard has landed in the hub's
// replica, multicast [offset, offset + bytes) of it to the rest of the output rectangle. Runs on both of the hub's
// RISCs, each with its own part and NoC.
void kernel_main() {
    constexpr uint32_t num_senders = get_compile_time_arg_val(0);
    constexpr uint32_t num_dests = get_compile_time_arg_val(1);
    uint32_t mcast_start_x = get_compile_time_arg_val(2);
    uint32_t mcast_start_y = get_compile_time_arg_val(3);
    uint32_t mcast_end_x = get_compile_time_arg_val(4);
    uint32_t mcast_end_y = get_compile_time_arg_val(5);
    constexpr uint32_t gather_semaphore_id = get_compile_time_arg_val(6);

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t offset = get_arg_val<uint32_t>(1);
    const uint32_t bytes = get_arg_val<uint32_t>(2);

    auto* gather_semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(gather_semaphore_id));
    noc_semaphore_wait_min(gather_semaphore, num_senders);
    if (bytes == 0 || num_dests == 0) {
        return;
    }
    if (noc_index == 1) {
        std::swap(mcast_start_x, mcast_end_x);
        std::swap(mcast_start_y, mcast_end_y);
    }
    const uint64_t mcast_addr =
        get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, output_addr + offset);
    noc_async_write_multicast(output_addr + offset, mcast_addr, bytes, num_dests);
    noc_async_write_barrier();
}
