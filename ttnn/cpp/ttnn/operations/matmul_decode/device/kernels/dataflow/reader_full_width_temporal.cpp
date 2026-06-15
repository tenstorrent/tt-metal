// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

// deep-plan_14 Lever 2 -- WIDTH-temporal k_stream activation (in0 / A) reader.
//
// The one-shot reader (reader_full_width_sharded.cpp) gathers the FULL A onto every core in
// a single round (num_senders * shard_num_tiles tiles, one push_back). For large-K shapes
// the full-A gather BUSTS L1, so this reader STREAMS A in G_temporal K-slices.
//
// Mapping (deep-plan_13 sec 4.5 / sec 5.3): k_slice_tiles == inA_K_tiles_per_core (each
// sender's whole shard is exactly one temporal slice), so G_temporal == num_senders. Round
// s broadcasts sender s's [M_tiles*32, k_slice*32] slab into a SINGLE-buffered slice CB
// (c_3) on every core, then signals the round complete; every core consumes that slice and
// frees it before the next round's broadcast. a_cores <= out_cores (device-validated; the
// wrapper co-derives a_cores so K_tiles / a_cores == k_slice_tiles).
//
// SINGLE-buffer (not double): the multicast destination address on the slice CB must be
// IDENTICAL across all receiver cores at broadcast time. A single-slot CB pins that address
// (slot 0) deterministically -- the prior round is fully consumed+popped (gated by the
// monotonic `done` semaphore) before the next reserve, so every core's write pointer is back
// at slot 0. (Double-buffering would make the destination address depend on each receiver's
// CB write-pointer state, racing the multicast -- the documented hang hazard; single-buffer
// trades the compute/gather overlap for a hang-safe stable address.)
//
// Semaphores: a single monotonic `done` semaphore. After round s the coordinator sets it to
// s+1 and multicasts; every core waits for done >= s+1 before publishing slice s. The
// `gather` semaphore lets the coordinator know sender s has landed its slice (round s only
// needs the ONE active sender to report -- gather reaches s+1 after round s).
void kernel_main() {
    constexpr uint32_t in0_cb_index = get_compile_time_arg_val(0);       // this core's sharded A slice
    constexpr uint32_t full_in0_cb_index = get_compile_time_arg_val(1);  // double-buffered slice dst
    constexpr uint32_t shard_num_tiles = get_compile_time_arg_val(2);    // M_tiles * k_slice (one slice)
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);     // == G_temporal
    constexpr uint32_t num_receivers = get_compile_time_arg_val(5);   // cores in the mcast rectangle
    uint32_t mcast_x_start = get_compile_time_arg_val(6);
    uint32_t mcast_y_start = get_compile_time_arg_val(7);
    uint32_t mcast_x_end = get_compile_time_arg_val(8);
    uint32_t mcast_y_end = get_compile_time_arg_val(9);
    constexpr uint32_t gather_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t coordinator_noc_x = get_compile_time_arg_val(12);
    constexpr uint32_t coordinator_noc_y = get_compile_time_arg_val(13);
    constexpr uint32_t in1_cb_index = get_compile_time_arg_val(14);
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(15);
    // Number of output DST groups the compute kernel iterates (M_tiles split into
    // rows_per_group chunks when M_tiles*N_tpc > DST cap). The compute re-consumes the full
    // A stream once per group, so the reader re-broadcasts the whole sender sequence
    // num_groups times.
    constexpr uint32_t num_groups = get_compile_time_arg_val(16);

    const uint32_t is_sender = get_arg_val<uint32_t>(0);
    const uint32_t sender_id = get_arg_val<uint32_t>(1);       // K-slice / round index (valid when sender)
    const uint32_t is_coordinator = get_arg_val<uint32_t>(2);

    const uint32_t shard_size_bytes = shard_num_tiles * tile_size_bytes;

    if (noc_index == 1) {
        std::swap(mcast_x_start, mcast_x_end);
        std::swap(mcast_y_start, mcast_y_end);
    }

    Noc noc;
    CircularBuffer in0_cb(in0_cb_index);
    CircularBuffer in1_cb(in1_cb_index);
    CircularBuffer full_in0_cb(full_in0_cb_index);
    Semaphore<> gather_sem(gather_sem_id);
    Semaphore<> done_sem(done_sem_id);

    // B (in1) is already resident in L1; publish it once.
    in1_cb.reserve_back(in1_num_tiles);
    in1_cb.push_back(in1_num_tiles);

    // ---- Temporal round loop: num_groups passes, each one round per sender / K-slice ----
    // `done` is monotonic across the whole call: after global round g (0-indexed) it reaches
    // g+1. Global round = group * num_senders + s.
    uint32_t round = 0;
    for (uint32_t grp = 0; grp < num_groups; ++grp) {
    for (uint32_t s = 0; s < num_senders; ++s, ++round) {
        // Reserve room for ONE slice in the single-buffered slice CB. The CB framework
        // serializes against the compute consumer so the previous slice has been popped.
        full_in0_cb.reserve_back(shard_num_tiles);

        if (is_sender && sender_id == s) {
            // This sender broadcasts its (whole) slice into every core's slice CB at offset 0.
            noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                use<CircularBuffer::AddrSelector::READ_PTR>(in0_cb),
                full_in0_cb,
                shard_size_bytes,
                num_receivers,
                {.offset_bytes = 0},
                {.noc_x_start = mcast_x_start,
                 .noc_y_start = mcast_y_start,
                 .noc_x_end = mcast_x_end,
                 .noc_y_end = mcast_y_end,
                 .offset_bytes = 0});
            noc.async_write_barrier();
            // Report the slice has landed (gather reaches round+1 for THIS global round).
            gather_sem.up(noc, coordinator_noc_x, coordinator_noc_y, 1);
            noc.async_atomic_barrier();
        }

        if (is_coordinator) {
            // Wait for THIS global round's sender to have landed its slice.
            gather_sem.wait(round + 1);
            // Signal the round complete to every core (monotonic: done reaches round+1).
            done_sem.set(round + 1);
            done_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_receivers);
            noc.async_write_barrier();
        }

        // Every core waits for this round's broadcast to be complete, then publishes the slice.
        done_sem.wait(round + 1);
        full_in0_cb.push_back(shard_num_tiles);
    }
    }
}
