// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t num_senders = get_compile_time_arg_val(0);
    constexpr uint32_t full_tensor_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t bbox_num_cores = get_compile_time_arg_val(2);
    uint32_t mcast_start_x = get_compile_time_arg_val(3);
    uint32_t mcast_start_y = get_compile_time_arg_val(4);
    uint32_t mcast_end_x = get_compile_time_arg_val(5);
    uint32_t mcast_end_y = get_compile_time_arg_val(6);
    constexpr uint32_t gather_semaphore_id = get_compile_time_arg_val(7);

    const bool is_hub = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);

    // This kernel is deliberately launched on every core in the output CoreRangeSet's bounding
    // box. Only the hub participates; all receiver kernels return without acknowledging the
    // multicast.
    if (!is_hub) {
        return;
    }

    auto* gather_semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(gather_semaphore_id));
    noc_semaphore_wait(gather_semaphore, num_senders);

    if constexpr (bbox_num_cores > 1) {
        if (noc_index == 1) {
            std::swap(mcast_start_x, mcast_end_x);
            std::swap(mcast_start_y, mcast_end_y);
        }

        const uint64_t mcast_addr =
            get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, output_addr);
        // The hub already owns the assembled output shard, so exclude it from the multicast.
        noc_async_write_multicast(output_addr, mcast_addr, full_tensor_bytes, bbox_num_cores - 1);
        noc_async_write_barrier();
    }
}
