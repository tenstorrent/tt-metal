// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "api/dataflow/dataflow_api.h"

// Row tiles ((h x 32) tiles of one tile row): this core's shard already is a contiguous run of the replica, so it
// multicasts [src_offset, src_offset + bytes) of its shard straight to every core of the output rectangle
// (itself included) at dst_offset, with no gather and no hub. The same kernel runs on both RISCs, each taking
// part of the shard on its own NoC.
void kernel_main() {
    uint32_t mcast_start_x = get_compile_time_arg_val(0);
    uint32_t mcast_start_y = get_compile_time_arg_val(1);
    uint32_t mcast_end_x = get_compile_time_arg_val(2);
    uint32_t mcast_end_y = get_compile_time_arg_val(3);
    constexpr uint32_t num_dests = get_compile_time_arg_val(4);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t src_offset = get_arg_val<uint32_t>(2);
    const uint32_t dst_offset = get_arg_val<uint32_t>(3);
    const uint32_t bytes = get_arg_val<uint32_t>(4);
    if (bytes == 0) {
        return;
    }
    if (noc_index == 1) {
        std::swap(mcast_start_x, mcast_end_x);
        std::swap(mcast_start_y, mcast_end_y);
    }
    const uint64_t mcast_addr =
        get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, dst_addr + dst_offset);
    noc_async_write_multicast_loopback_src(src_addr + src_offset, mcast_addr, bytes, num_dests);
    noc_async_write_barrier();
}
