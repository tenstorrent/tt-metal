// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

// Full-width-sharded matmul activation (in0 / A) reader -- TEMPORAL K-STREAMING variant
// (plan_5 s3.4). Used for the large-K down projection (K=16384) where the one-shot
// gather reader's full_in0 CB (= M_tiles*K_tiles tiles) OOMs L1. Instead of gathering the
// WHOLE A onto every core once, this reader streams A one K-slice at a time so full_in0
// only ever holds one (CB-double-buffered) slice.
//
// Layout: A is width(K)-sharded across `num_senders` sender cores; sender `s` holds a
// contiguous [M_tiles*32, sender_K_tiles*32] block of A in TILE row-major order in its
// resident in0_cb. K is streamed in slices of `K_slice_tiles` columns. With
// K_slice_tiles <= sender_K_tiles (= K_tiles/num_senders) every stream step is owned by
// EXACTLY ONE sender (the slice never spans a sender boundary), so each step is a
// single-sender broadcast.
//
// Per stream step `s` (s = 0 .. num_steps-1):
//   1. Every core reserve_back(M_tiles*K_slice_tiles) on full_in0_cb. The CB is sized for
//      TWO slices, so reserve_back blocks until the previous-previous slice has been popped
//      by compute -- this is the back-pressure / double-buffering, provided by the CB itself
//      (no hand-rolled 2-slot bookkeeping needed).
//   2. The single owning sender copies its M_tiles row-blocks of this slice from its
//      resident in0_cb into a contiguous [M_tiles, K_slice_tiles] block and multicasts it to
//      every core's full_in0_cb write-ptr (offset 0 of the freshly reserved slot), then
//      barriers and bumps the coordinator's gather_sem by 1.
//   3. The coordinator wait_min's gather_sem to (s+1) (monotonic; one sender per step), then
//      set()s done_sem to (s+1) locally and multicasts it to all cores. MONOTONIC thresholds
//      avoid any per-step semaphore reset / reset-race.
//   4. Every core wait_min's done_sem to (s+1), then push_back(M_tiles*K_slice_tiles).
//
// B (in1) is the resident full-K down weight -- published ONCE outside the stream loop.
//
// full_in0 slice layout (what compute reads): the slot holds M_tiles row-blocks, each of
// K_slice_tiles contiguous K-tiles, i.e. tile (m, kc) -> m*K_slice_tiles + kc. Single-sender
// per step => the sender-major term collapses; compute uses local index m*K_slice_tiles+kc.

void kernel_main() {
    // ---- Compile-time args (must match factory's stream-reader CT arg order) ----
    constexpr uint32_t in0_cb_index = get_compile_time_arg_val(0);       // resident A slice (source)
    constexpr uint32_t full_in0_cb_index = get_compile_time_arg_val(1);  // streamed slice (destination)
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_senders = get_compile_time_arg_val(3);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(4);  // cores in mcast rectangle
    uint32_t mcast_x_start = get_compile_time_arg_val(5);
    uint32_t mcast_y_start = get_compile_time_arg_val(6);
    uint32_t mcast_x_end = get_compile_time_arg_val(7);
    uint32_t mcast_y_end = get_compile_time_arg_val(8);
    constexpr uint32_t gather_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t coordinator_noc_x = get_compile_time_arg_val(11);
    constexpr uint32_t coordinator_noc_y = get_compile_time_arg_val(12);
    constexpr uint32_t in1_cb_index = get_compile_time_arg_val(13);
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t M_tiles = get_compile_time_arg_val(15);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(16);               // total K-tiles
    constexpr uint32_t K_slice_tiles = get_compile_time_arg_val(17);         // K-tiles per stream step
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(18);  // sender_K_tiles

    // ---- Runtime args ----
    const uint32_t is_sender = get_arg_val<uint32_t>(0);
    const uint32_t sender_id = get_arg_val<uint32_t>(1);
    const uint32_t is_coordinator = get_arg_val<uint32_t>(2);
    // is_consumer: this core runs a compute kernel that drains full_in0. Only consumer cores
    // run the per-step reserve_back/push_back loop -- "orphan" bbox cores (not sender, not
    // output) have no consumer, so they would block forever on the 3rd reserve_back. They still
    // receive the multicast (harmless) and the done_sem broadcast, but skip the CB ops.
    const uint32_t is_consumer = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_steps = K_tiles / K_slice_tiles;
    constexpr uint32_t slice_tiles = M_tiles * K_slice_tiles;
    const uint32_t slice_size_bytes = slice_tiles * tile_size_bytes;
    // Steps owned by this sender: steps s with (s*K_slice_tiles) in [sender base, sender base + sender_K).
    constexpr uint32_t steps_per_sender = inA_K_tiles_per_core / K_slice_tiles;
    // Row stride (in tiles) within the resident in0_cb between consecutive M-rows of this sender.
    constexpr uint32_t sender_row_tiles = inA_K_tiles_per_core;

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

    // B (in1) resident; publish once.
    in1_cb.reserve_back(in1_num_tiles);
    in1_cb.push_back(in1_num_tiles);

    const uint32_t my_first_step = is_sender ? sender_id * steps_per_sender : 0;
    const uint32_t my_last_step = my_first_step + steps_per_sender;  // exclusive

    for (uint32_t s = 0; s < num_steps; ++s) {
        // Reserve the next slice slot (blocks until a slot frees -> back-pressure). Only consumer
        // cores have a compute drainer; orphan bbox cores must NOT block on this.
        if (is_consumer) {
            full_in0_cb.reserve_back(slice_tiles);
        }

        const bool i_own_this_step = is_sender && (s >= my_first_step) && (s < my_last_step);
        if (i_own_this_step) {
            // Local K-tile range of this slice within MY resident in0_cb.
            const uint32_t local_k0 = (s - my_first_step) * K_slice_tiles;
            // Copy this sender's [M_tiles, K_slice_tiles] sub-block from in0_cb into a
            // contiguous staging at full_in0_cb's write-ptr, then multicast it. The resident
            // in0_cb holds [M_tiles, inA_K_tiles_per_core] row-major; the destination slot is
            // [M_tiles, K_slice_tiles] row-major. We multicast each M-row separately because the
            // source rows are strided (sender_row_tiles) while the dest rows are packed.
            const uint32_t src_base = in0_cb.get_read_ptr();
            const uint32_t dst_base = full_in0_cb.get_write_ptr();
            for (uint32_t m = 0; m < M_tiles; ++m) {
                const uint32_t src_off = (m * sender_row_tiles + local_k0) * tile_size_bytes;
                const uint32_t dst_off = (m * K_slice_tiles) * tile_size_bytes;
                noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                    use<CircularBuffer::AddrSelector::READ_PTR>(in0_cb),
                    full_in0_cb,
                    K_slice_tiles * tile_size_bytes,
                    num_receivers,
                    {.offset_bytes = src_off},
                    {.noc_x_start = mcast_x_start,
                     .noc_y_start = mcast_y_start,
                     .noc_x_end = mcast_x_end,
                     .noc_y_end = mcast_y_end,
                     .offset_bytes = dst_off});
            }
            noc.async_write_barrier();
            // Report this step's slice landed (monotonic: cumulative gather count == s+1).
            gather_sem.up(noc, coordinator_noc_x, coordinator_noc_y, 1);
            noc.async_atomic_barrier();
        }

        if (is_coordinator) {
            gather_sem.wait_min(s + 1);
            done_sem.set(s + 1);
            done_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_receivers);
            noc.async_write_barrier();
        }

        done_sem.wait_min(s + 1);
        if (is_consumer) {
            full_in0_cb.push_back(slice_tiles);
        }
    }
}
