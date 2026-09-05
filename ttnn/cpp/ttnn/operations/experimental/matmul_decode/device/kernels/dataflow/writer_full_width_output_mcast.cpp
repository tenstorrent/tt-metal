// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#ifdef FUSE_RMS_NORM
#include "full_width_rms_norm_transport.hpp"
#endif

// Replicate each producer's [M, Nc] shard into a full [M, N] buffer on output_core_grid.
//
// Dest tiles are row-major: out[mt, n] at mt * N_tiles + n. Local compute packs mt * Nc + nc, so each
// M-row's Nc tiles are one contiguous unicast into the staging buffer.
//
// Producers unicast into a staging CB on the dest bbox; hub0 (bbox start) then mcasts N-columns
// [0, split_P * Nc) on this kernel's NOC and, when there are two hubs, hub1 (bbox end) mcasts the
// rest. Hub0 is compiled onto NOC0 and hub1 onto NOC1, matching the in0 two-hub gather.
//
// The staging indirection is what keeps this to one multicast sender per NOC. Having each producer
// mcast its own slice instead deadlocks: those mcasts reserve overlapping paths into the same dest
// rectangle, and concurrent path reservations on one NOC circular-wait, so the write barrier never
// drains. The in0 gather in this op splits its two senders across the NOCs for the same reason.
void kernel_main() {
    constexpr uint32_t local_out_cb_index = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t stage_cb_index = get_named_compile_time_arg_val("cb_out_stage");
    constexpr uint32_t dest_cb_index = get_named_compile_time_arg_val("cb_out_full");

    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t Nc_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_dest_cores = get_compile_time_arg_val(4);
    uint32_t mcast_x_start = get_compile_time_arg_val(5);
    uint32_t mcast_y_start = get_compile_time_arg_val(6);
    uint32_t mcast_x_end = get_compile_time_arg_val(7);
    uint32_t mcast_y_end = get_compile_time_arg_val(8);
    constexpr uint32_t stage_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t hub0_noc_x = get_compile_time_arg_val(11);
    constexpr uint32_t hub0_noc_y = get_compile_time_arg_val(12);
    constexpr uint32_t hub1_noc_x = get_compile_time_arg_val(13);
    constexpr uint32_t hub1_noc_y = get_compile_time_arg_val(14);
    constexpr uint32_t split_P = get_compile_time_arg_val(15);
    constexpr uint32_t num_producers = get_compile_time_arg_val(16);
    // 1 or 2. With one hub, split_P is num_producers, so hub0 owns the whole N range.
    constexpr uint32_t num_hubs = get_compile_time_arg_val(17);

    const uint32_t is_producer = get_arg_val<uint32_t>(0);
    const uint32_t n_idx = get_arg_val<uint32_t>(1);
    const uint32_t role = get_arg_val<uint32_t>(2);  // HubRole: 0 plain, 1 hub0, 2 hub1
    // Only dest cores receive the hubs' done bump; a producer outside the dest grid must not wait.
    const uint32_t is_dest = get_arg_val<uint32_t>(3);

#ifdef FUSE_RMS_NORM
    if (is_producer) {
        run_full_width_rms_norm_transport(
            get_arg_val<uint32_t>(4) != 0,
            get_arg_val<uint32_t>(5),
            get_arg_val<uint32_t>(6),
            get_arg_val<uint32_t>(7),
            get_arg_val<uint32_t>(8),
            get_arg_val<uint32_t>(9),
            get_arg_val<uint32_t>(10),
            get_arg_val<uint32_t>(11),
            get_arg_val<uint32_t>(12));
    }
#endif

    constexpr uint32_t local_num_tiles = M_tiles * Nc_tiles;
    constexpr uint32_t full_num_tiles = M_tiles * N_tiles;
    constexpr uint32_t row_bytes = Nc_tiles * tile_size_bytes;

    const bool is_hub0 = role == 1;
    const bool is_hub1 = role == 2;

    if (noc_index == 1) {
        std::swap(mcast_x_start, mcast_x_end);
        std::swap(mcast_y_start, mcast_y_end);
    }

    Noc noc;
    CircularBuffer stage_cb(stage_cb_index);
    Semaphore<> stage_sem(stage_sem_id);
    Semaphore<> done_sem(done_sem_id);
    stage_cb.reserve_back(full_num_tiles);

    if (is_producer) {
        CircularBuffer local_out_cb(local_out_cb_index);
        local_out_cb.wait_front(local_num_tiles);

        const bool to_hub0 = n_idx < split_P;
        const uint32_t hub_x = to_hub0 ? hub0_noc_x : hub1_noc_x;
        const uint32_t hub_y = to_hub0 ? hub0_noc_y : hub1_noc_y;
        UnicastEndpoint hub;
        for (uint32_t mt = 0; mt < M_tiles; ++mt) {
            const uint32_t src_off = mt * Nc_tiles * tile_size_bytes;
            const uint32_t dst_off = (mt * N_tiles + n_idx * Nc_tiles) * tile_size_bytes;
            noc.async_write(
                local_out_cb,
                hub,
                row_bytes,
                {.offset_bytes = src_off},
                {.noc_x = hub_x, .noc_y = hub_y, .addr = stage_cb.get_write_ptr() + dst_off});
        }
        noc.async_write_barrier();
        stage_sem.up(noc, hub_x, hub_y, 1);
        noc.async_atomic_barrier();
        local_out_cb.pop_front(local_num_tiles);
    }

    if (is_hub0 || is_hub1) {
        CircularBuffer dest_cb(dest_cb_index);
        const uint32_t region_first = is_hub0 ? 0 : split_P;
        const uint32_t region_count = is_hub0 ? split_P : (num_producers - split_P);
        const uint32_t region_n_tiles = region_count * Nc_tiles;
        const uint32_t region_row_bytes = region_n_tiles * tile_size_bytes;
        if (region_count > 0) {
            stage_sem.wait(region_count);
            for (uint32_t mt = 0; mt < M_tiles; ++mt) {
                const uint32_t off = (mt * N_tiles + region_first * Nc_tiles) * tile_size_bytes;
                noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                    use<CircularBuffer::AddrSelector::WRITE_PTR>(stage_cb),
                    dest_cb,
                    region_row_bytes,
                    num_dest_cores,
                    {.offset_bytes = off},
                    {.noc_x_start = mcast_x_start,
                     .noc_y_start = mcast_y_start,
                     .noc_x_end = mcast_x_end,
                     .noc_y_end = mcast_y_end,
                     .offset_bytes = off});
            }
            noc.async_write_barrier();
        }
        const uint32_t self_x = is_hub0 ? hub0_noc_x : hub1_noc_x;
        const uint32_t self_y = is_hub0 ? hub0_noc_y : hub1_noc_y;
        done_sem.inc_multicast(noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, 1, num_dest_cores - 1);
        done_sem.up(noc, self_x, self_y, 1);
        noc.async_atomic_barrier();
    }

    if (is_dest) {
        done_sem.wait(num_hubs);
    }
}
