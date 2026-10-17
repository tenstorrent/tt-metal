// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#ifdef ENABLE_GLOBAL_CB
#include "api/remote_circular_buffer.h"
#endif

// Gathers width(K)-sharded A onto every core via two-hub gather/broadcast.
//
// in1 (weights) arrive one of two ways:
//   - default: the in1 CB is globally allocated over the L1-resident weight shard, so the
//     reader only has to declare its tiles available.
//   - ENABLE_GLOBAL_CB: the weights are pushed into a DRAM-sender GlobalCircularBuffer by the
//     tensor prefetcher. This receiver's [Kc, Nc] slab arrives as num_k_blocks remote pages, each
//     a whole number of K-rows (num_k_blocks == 1 is the whole slab in one page). The reader waits
//     for a page, hands its tiles to compute through the local alias CB, and releases it only
//     after compute signals (via the sync CB) that it has finished reading -- releasing earlier
//     would let the prefetcher overwrite weights still in use. This kernel runs on the merged
//     A/B/output bounding box, so cores outside the B grid have no in1 or sync CB and must skip
//     the handshake entirely.
void kernel_main() {
    // CB indices come in as named args so op fusion can remap them onto the hardware slots it
    // pool-allocates across phases (see models/experimental/ops/descriptors/fusion/docs/op_fusion.md).
    constexpr uint32_t in0_cb_index = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t full_in0_cb_index = get_named_compile_time_arg_val("cb_full_in0");
    constexpr uint32_t in1_cb_index = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t remote_cb_index = get_named_compile_time_arg_val("cb_in1_remote");
    constexpr uint32_t sync_cb_index = get_named_compile_time_arg_val("cb_sync");

    constexpr uint32_t shard_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_senders = get_compile_time_arg_val(2);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(3);
    uint32_t mcast_x_start = get_compile_time_arg_val(4);
    uint32_t mcast_y_start = get_compile_time_arg_val(5);
    uint32_t mcast_x_end = get_compile_time_arg_val(6);
    uint32_t mcast_y_end = get_compile_time_arg_val(7);
    constexpr uint32_t stage_sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t hub0_noc_x = get_compile_time_arg_val(10);
    constexpr uint32_t hub0_noc_y = get_compile_time_arg_val(11);
    constexpr uint32_t hub1_noc_x = get_compile_time_arg_val(12);
    constexpr uint32_t hub1_noc_y = get_compile_time_arg_val(13);
    constexpr uint32_t split_H = get_compile_time_arg_val(14);
    constexpr uint32_t in1_page_tiles = get_compile_time_arg_val(15);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(16);

    const uint32_t is_sender = get_arg_val<uint32_t>(0);
    const uint32_t sender_id = get_arg_val<uint32_t>(1);
    const uint32_t role = get_arg_val<uint32_t>(2);
    const uint32_t is_in1_receiver = get_arg_val<uint32_t>(3);

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

#ifdef ENABLE_GLOBAL_CB
    if (is_in1_receiver) {
        // Publish the first page before the gather so the prefetcher's transfer overlaps it.
        in1_cb.reserve_back(in1_page_tiles);
        experimental::remote_cb_wait_front(remote_cb_index, 1);
        in1_cb.push_back(in1_page_tiles);
    }
#else
    in1_cb.reserve_back(in1_page_tiles);
    in1_cb.push_back(in1_page_tiles);
#endif
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

#ifdef ENABLE_GLOBAL_CB
    if (is_in1_receiver) {
        CircularBuffer sync_cb(sync_cb_index);
        // Page p-1's credit is returned only after page p has been published, so the reader runs a
        // page ahead of the credit return and the remote read pointer lags by one page. That is
        // what makes the wait below mean "the page I still owe a credit for, plus the new one":
        // it can never reach past this weight's own pages into a transfer nobody has queued.
        for (uint32_t page = 1; page < num_k_blocks; ++page) {
            in1_cb.reserve_back(in1_page_tiles);
            experimental::remote_cb_wait_front(remote_cb_index, 2);
            in1_cb.push_back(in1_page_tiles);
            sync_cb.wait_front(1);
            sync_cb.pop_front(1);
            experimental::remote_cb_pop_front(remote_cb_index, 1);
        }
        // Compute signals here once it has finished reading every in1 tile of the last page.
        sync_cb.wait_front(1);
        sync_cb.pop_front(1);
        experimental::remote_cb_pop_front(remote_cb_index, 1);
        // Persist the remote read pointer so the next invocation resumes at the right ring offset.
        experimental::update_remote_cb_config_in_l1(remote_cb_index);
        noc.async_atomic_barrier();
    }
#endif

    noc.async_write_barrier();
    noc.async_read_barrier();
}
