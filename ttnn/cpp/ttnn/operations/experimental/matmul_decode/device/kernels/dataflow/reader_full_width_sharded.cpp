// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

// Gathers width(K)-sharded A onto every core via two-hub gather/broadcast.
void kernel_main() {
    constexpr uint32_t in0_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t full_in0_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t shard_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(5);
    uint32_t mcast_x_start = get_compile_time_arg_val(6);
    uint32_t mcast_y_start = get_compile_time_arg_val(7);
    uint32_t mcast_x_end = get_compile_time_arg_val(8);
    uint32_t mcast_y_end = get_compile_time_arg_val(9);
    constexpr uint32_t stage_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t hub0_noc_x = get_compile_time_arg_val(12);
    constexpr uint32_t hub0_noc_y = get_compile_time_arg_val(13);
    constexpr uint32_t hub1_noc_x = get_compile_time_arg_val(14);
    constexpr uint32_t hub1_noc_y = get_compile_time_arg_val(15);
    constexpr uint32_t split_H = get_compile_time_arg_val(16);
    constexpr uint32_t in1_cb_index = get_compile_time_arg_val(17);
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(18);

    const uint32_t is_sender = get_arg_val<uint32_t>(0);
    const uint32_t sender_id = get_arg_val<uint32_t>(1);
    const uint32_t role = get_arg_val<uint32_t>(2);

    constexpr uint32_t full_num_tiles = num_senders * shard_num_tiles;
    const uint32_t shard_size_bytes = shard_num_tiles * tile_size_bytes;

    // NOC_1 uses an inverted coordinate system.
    if (noc_index == 1) {
        std::swap(mcast_x_start, mcast_x_end);
        std::swap(mcast_y_start, mcast_y_end);
    }

    Noc noc;
    CircularBuffer in0_cb(in0_cb_index);
    CircularBuffer in1_cb(in1_cb_index);
    CircularBuffer full_in0_cb(full_in0_cb_index);
    Semaphore<> stage_sem(stage_sem_id);
    Semaphore<> done_sem(done_sem_id);
    UnicastEndpoint hub;

    in1_cb.reserve_back(in1_num_tiles);
    in1_cb.push_back(in1_num_tiles);
    full_in0_cb.reserve_back(full_num_tiles);

    const bool is_hub0 = (role == 1);
    const bool is_hub1 = (role == 2);

    if (is_sender) {
        const bool owned_by_hub0 = sender_id < split_H;
        const uint32_t hub_x = owned_by_hub0 ? hub0_noc_x : hub1_noc_x;
        const uint32_t hub_y = owned_by_hub0 ? hub0_noc_y : hub1_noc_y;
        const uint32_t dst_offset_bytes = sender_id * shard_size_bytes;

        // full_in0_cb is at the same L1 offset on every core, so the local write ptr is the remote dst addr.
        const uint32_t dst_l1_addr = full_in0_cb.get_write_ptr() + dst_offset_bytes;
        noc.async_write(
            in0_cb, hub, shard_size_bytes, {.offset_bytes = 0}, {.noc_x = hub_x, .noc_y = hub_y, .addr = dst_l1_addr});
        noc.async_write_barrier();
        stage_sem.up(noc, hub_x, hub_y, 1);
        noc.async_atomic_barrier();
    }

    if (is_hub0 || is_hub1) {
        const uint32_t region_first = is_hub0 ? 0 : split_H;
        const uint32_t region_count = is_hub0 ? split_H : (num_senders - split_H);
        const uint32_t region_offset_bytes = region_first * shard_size_bytes;
        const uint32_t region_size_bytes = region_count * shard_size_bytes;

        if (region_count > 0) {
            stage_sem.wait(region_count);

            noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                use<CircularBuffer::AddrSelector::WRITE_PTR>(full_in0_cb),
                full_in0_cb,
                region_size_bytes,
                num_receivers,
                {.offset_bytes = region_offset_bytes},
                {.noc_x_start = mcast_x_start,
                 .noc_y_start = mcast_y_start,
                 .noc_x_end = mcast_x_end,
                 .noc_y_end = mcast_y_end,
                 .offset_bytes = region_offset_bytes});
            noc.async_write_barrier();
        }

        // inc_multicast excludes the sender; self must use atomic NOC up() (local up() can race with the other hub).
        const uint32_t self_noc_x = is_hub0 ? hub0_noc_x : hub1_noc_x;
        const uint32_t self_noc_y = is_hub0 ? hub0_noc_y : hub1_noc_y;
        done_sem.inc_multicast(noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, 1, num_receivers - 1);
        done_sem.up(noc, self_noc_x, self_noc_y, 1);
        noc.async_atomic_barrier();
    }

    done_sem.wait(2);
    full_in0_cb.push_back(full_num_tiles);

    noc.async_write_barrier();
    noc.async_read_barrier();
}
