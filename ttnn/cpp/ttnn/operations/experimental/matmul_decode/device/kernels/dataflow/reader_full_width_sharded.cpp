// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

// Full-width-sharded matmul activation (in0 / A) reader.
//
// Input A is width(K)-sharded across only a *subset* of the compute cores
// ("sender" cores), while every compute core holds a slice of B and produces a
// slice of the output.  Each compute core needs the *entire* A matrix to do its
// matmul, so this kernel gathers A onto every core using a two-hub
// gather-then-broadcast scheme so that only two cores ever multicast:
//
//   * Two "hub" cores are chosen at opposite corners of the compute rectangle:
//     hub 0 at the start corner runs on NOC0 (the NOC whose multicast traffic
//     flows toward the end corner) and hub 1 at the end corner runs on NOC1.
//     The K-slices are split into two contiguous halves: hub 0 owns the first
//     `split_H` slices, hub 1 owns the rest.
//
//   1. Every core reserves room for the full A in `full_in0_cb`.
//   2. (Gather) Each sender writes its own K-slice of A into the *owning hub's*
//      `full_in0_cb` at the offset for its slice, using the NOC that matches
//      that hub (so the kernel's NOC is the hub's NOC), then bumps that hub's
//      `stage` semaphore.  A hub that also owns its own slice does an in-place
//      (loopback) copy.
//   3. (Broadcast) Each hub waits until every sender in its half has reported,
//      then multicasts its contiguous half of `full_in0_cb` to all cores
//      (including itself).  After the data lands it increments the `done`
//      semaphore on every core.
//   4. Every core waits for `done` to reach 2 (both hubs finished), then
//      publishes the fully-populated `full_in0_cb` to the compute kernel.
//
// Uses the Device 2.0 data movement API (Noc / CircularBuffer / Semaphore).
void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t in0_cb_index = get_compile_time_arg_val(0);       // this core's sharded A slice (source)
    constexpr uint32_t full_in0_cb_index = get_compile_time_arg_val(1);  // gathered full A (destination)
    constexpr uint32_t shard_num_tiles = get_compile_time_arg_val(2);    // tiles per A slice (M_tiles * inA_K_per_core)
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);    // # of A-holding cores
    constexpr uint32_t num_receivers = get_compile_time_arg_val(5);  // cores in the mcast rectangle (incl. self)
    uint32_t mcast_x_start = get_compile_time_arg_val(6);
    uint32_t mcast_y_start = get_compile_time_arg_val(7);
    uint32_t mcast_x_end = get_compile_time_arg_val(8);
    uint32_t mcast_y_end = get_compile_time_arg_val(9);
    constexpr uint32_t stage_sem_id = get_compile_time_arg_val(10);  // bumped by each sender on its owning hub
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(11);   // bumped by each hub on all cores
    constexpr uint32_t hub0_noc_x = get_compile_time_arg_val(12);    // hub 0 (start corner, NOC0)
    constexpr uint32_t hub0_noc_y = get_compile_time_arg_val(13);
    constexpr uint32_t hub1_noc_x = get_compile_time_arg_val(14);  // hub 1 (end corner, NOC1)
    constexpr uint32_t hub1_noc_y = get_compile_time_arg_val(15);
    constexpr uint32_t split_H = get_compile_time_arg_val(16);        // # of slices owned by hub 0
    constexpr uint32_t in1_cb_index = get_compile_time_arg_val(17);   // this core's sharded B slice
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(18);  // tiles of B resident on this core

    // ---- Runtime args ----
    const uint32_t is_sender = get_arg_val<uint32_t>(0);  // 1 if this core holds a slice of A
    const uint32_t sender_id = get_arg_val<uint32_t>(1);  // K-slice index (valid when is_sender)
    const uint32_t role = get_arg_val<uint32_t>(2);       // 0 = plain core, 1 = hub 0, 2 = hub 1

    constexpr uint32_t full_num_tiles = num_senders * shard_num_tiles;
    const uint32_t shard_size_bytes = shard_num_tiles * tile_size_bytes;

    // NOC_1 uses an inverted coordinate system, so the rectangle corners swap.
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

    // B (in1) is already resident in L1; just publish it to compute.
    in1_cb.reserve_back(in1_num_tiles);
    in1_cb.push_back(in1_num_tiles);

    // Reserve space for the whole A matrix; multicast writes land directly here.
    full_in0_cb.reserve_back(full_num_tiles);

    const bool is_hub0 = (role == 1);
    const bool is_hub1 = (role == 2);

    // ---- Phase 1: gather each sender's slice onto its owning hub ----
    if (is_sender) {
        const bool owned_by_hub0 = sender_id < split_H;
        const uint32_t hub_x = owned_by_hub0 ? hub0_noc_x : hub1_noc_x;
        const uint32_t hub_y = owned_by_hub0 ? hub0_noc_y : hub1_noc_y;
        const uint32_t dst_offset_bytes = sender_id * shard_size_bytes;

        // Unicast this slice into the owning hub's full_in0_cb at this slice's
        // offset.  full_in0_cb is allocated identically on every core, so the
        // local write pointer is also the destination address on the hub.  When
        // this core *is* the owning hub the write is a NoC loopback to its own
        // coordinates (a local copy).  The NoC applies the per-NOC coordinate
        // translation, so the untranslated hub coords are correct on whichever
        // NOC this kernel runs on (host picks the hub-appropriate NOC).
        const uint32_t dst_l1_addr = full_in0_cb.get_write_ptr() + dst_offset_bytes;
        noc.async_write(
            in0_cb, hub, shard_size_bytes, {.offset_bytes = 0}, {.noc_x = hub_x, .noc_y = hub_y, .addr = dst_l1_addr});
        // Ensure the slice has landed on the hub before reporting in.
        noc.async_write_barrier();
        stage_sem.up(noc, hub_x, hub_y, 1);
        noc.async_atomic_barrier();
    }

    // ---- Phase 2: each hub broadcasts its contiguous half to all cores ----
    if (is_hub0 || is_hub1) {
        const uint32_t region_first = is_hub0 ? 0 : split_H;
        const uint32_t region_count = is_hub0 ? split_H : (num_senders - split_H);
        const uint32_t region_offset_bytes = region_first * shard_size_bytes;
        const uint32_t region_size_bytes = region_count * shard_size_bytes;

        if (region_count > 0) {
            // Wait until every sender in this hub's half has delivered its slice.
            stage_sem.wait(region_count);

            noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                use<CircularBuffer::AddrSelector::WRITE_PTR>(full_in0_cb),  // source: this hub's assembled half
                full_in0_cb,                                                // destination: same offset on all cores
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

        // Signal completion to every core: increment the others via multicast
        // (source is excluded, so num_receivers - 1 destinations) plus self.

        const uint32_t self_noc_x = is_hub0 ? hub0_noc_x : hub1_noc_x;
        const uint32_t self_noc_y = is_hub0 ? hub0_noc_y : hub1_noc_y;
        done_sem.inc_multicast(noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, 1, num_receivers - 1);
        done_sem.up(noc, self_noc_x, self_noc_y, 1);
        noc.async_atomic_barrier();
    }

    // Wait until both hubs have finished broadcasting their halves.
    done_sem.wait(2);

    // The full A matrix is now resident on this core; hand it to compute.
    full_in0_cb.push_back(full_num_tiles);

    noc.async_write_barrier();
    noc.async_read_barrier();
}
