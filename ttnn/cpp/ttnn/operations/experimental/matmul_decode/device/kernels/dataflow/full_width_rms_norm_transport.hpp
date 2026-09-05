// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Stats transport for the fused RMSNorm epilogue, run by the writer RISC on every producer core.
//
// CB ownership is strictly single-producer / single-consumer:
//   cb_rms_local      compute -> writer   (local sum of squares; each producer writes its page to the hub)
//   cb_rms_gathered   writer -> compute   (hub only: producer pages packed into full reduction tiles)
//   cb_rms_reduce_scaler writer -> compute (hub only: SUM reduction scaler)
//   cb_rms_scale_src  compute -> writer   (hub only: the finished scale, handed to the NoC)
//   cb_rms_scale      writer -> compute   (multicast destination on every producer, hub included)
//
// The hub must not consume cb_rms_scale: compute on the hub is a consumer of that CB, and a CB with two
// independent consumers can be popped out from under the writer. Hence the separate cb_rms_scale_src.
inline void run_full_width_rms_norm_transport(
    bool is_hub,
    uint32_t hub_x,
    uint32_t hub_y,
    uint32_t mcast_start_x,
    uint32_t mcast_start_y,
    uint32_t mcast_end_x,
    uint32_t mcast_end_y,
    uint32_t mcast_num_cores,
    uint32_t producer_index) {
    constexpr uint32_t cb_rms_local = get_named_compile_time_arg_val("cb_rms_local");
    constexpr uint32_t cb_rms_gathered = get_named_compile_time_arg_val("cb_rms_gathered");
    constexpr uint32_t cb_rms_scale_src = get_named_compile_time_arg_val("cb_rms_scale_src");
    constexpr uint32_t cb_rms_scale = get_named_compile_time_arg_val("cb_rms_scale");
    constexpr uint32_t cb_rms_reduce_scaler = get_named_compile_time_arg_val("cb_rms_reduce_scaler");
    constexpr uint32_t arrival_sem_id = get_named_compile_time_arg_val("rms_arrival_sem");
    constexpr uint32_t scale_ready_sem_id = get_named_compile_time_arg_val("rms_scale_ready_sem");
    constexpr uint32_t M_tiles = get_named_compile_time_arg_val("rms_m_tiles");
    constexpr uint32_t num_producers = get_named_compile_time_arg_val("rms_num_producers");
    constexpr uint32_t local_tile_size = get_named_compile_time_arg_val("rms_local_tile_size");
    constexpr uint32_t reduce_tile_size = get_named_compile_time_arg_val("rms_reduce_tile_size");
    constexpr uint32_t packed_tiles_per_row = get_named_compile_time_arg_val("rms_packed_tiles_per_row");
    constexpr uint32_t packed_tiles = M_tiles * packed_tiles_per_row;
    constexpr uint32_t packed_bytes_per_row = packed_tiles_per_row * reduce_tile_size;
    static_assert(num_producers * local_tile_size <= packed_bytes_per_row);

    Noc noc;
    CircularBuffer rms_local(cb_rms_local);
    CircularBuffer rms_gathered(cb_rms_gathered);
    CircularBuffer rms_scale_src(cb_rms_scale_src);
    CircularBuffer rms_scale(cb_rms_scale);
    Semaphore<> arrival_sem(arrival_sem_id);
    Semaphore<> scale_ready_sem(scale_ready_sem_id);

    if (noc_index == 1) {
        std::swap(mcast_start_x, mcast_end_x);
        std::swap(mcast_start_y, mcast_end_y);
    }

    // Every producer, hub included, reserves the multicast destination before advertising its local
    // statistic. The arrival semaphore therefore also certifies that the slots the hub is about to write
    // are free, and that the destination write pointer is at the CB base on all receivers.
    rms_scale.reserve_back(M_tiles);

    // Each local statistics page contains one meaningful scalar at [0,0]; REDUCE_SCALAR packing
    // guarantees the rest of that page is zero. Producers write their pages directly into disjoint
    // packed slots on the hub, parallelizing the old hub-issued read loop.
    rms_local.wait_front(M_tiles);
    if (is_hub) {
        rms_gathered.reserve_back(packed_tiles);
        auto* packed_dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rms_gathered.get_write_ptr());
        constexpr uint32_t valid_bytes_per_row = num_producers * local_tile_size;
        for (uint32_t mt = 0; mt < M_tiles; ++mt) {
            const uint32_t tail_start = mt * packed_bytes_per_row + valid_bytes_per_row;
            const uint32_t tail_end = (mt + 1) * packed_bytes_per_row;
            for (uint32_t byte = tail_start; byte < tail_end; byte += sizeof(uint32_t)) {
                packed_dst[byte / sizeof(uint32_t)] = 0;
            }
        }
    }

    UnicastEndpoint hub;
    for (uint32_t mt = 0; mt < M_tiles; ++mt) {
        noc.async_write(
            rms_local,
            hub,
            local_tile_size,
            {.offset_bytes = mt * local_tile_size},
            {.noc_x = hub_x,
             .noc_y = hub_y,
             .addr = rms_gathered.get_write_ptr() + mt * packed_bytes_per_row + producer_index * local_tile_size});
    }
    noc.async_write_barrier();
    arrival_sem.up(noc, hub_x, hub_y, 1);
    noc.async_atomic_barrier();

    if (is_hub) {
        arrival_sem.wait(num_producers);
        arrival_sem.set(0);
        rms_gathered.push_back(packed_tiles);

        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_rms_reduce_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_SCALAR>();

        // Loopback multicast: the hub's own cb_rms_scale is one of the destinations, so the scale reaches
        // every producer through a single transfer and the hub never writes its own destination CB by hand.
        rms_scale_src.wait_front(M_tiles);
        const uint32_t scale_bytes = M_tiles * rms_scale.get_tile_size();
        noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
            use<CircularBuffer::AddrSelector::READ_PTR>(rms_scale_src),
            rms_scale,
            scale_bytes,
            mcast_num_cores,
            {.offset_bytes = 0},
            {.noc_x_start = mcast_start_x,
             .noc_y_start = mcast_start_y,
             .noc_x_end = mcast_end_x,
             .noc_y_end = mcast_end_y,
             .offset_bytes = 0});
        // Blocking barrier, not a flush: the payload must be committed in every receiver's L1 before the
        // readiness flag lets that receiver's compute read it, and before the hub pushes its own copy.
        noc.async_write_barrier();

        if (mcast_num_cores > 1) {
            scale_ready_sem.set(VALID);
            scale_ready_sem.set_multicast(
                noc, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, mcast_num_cores - 1);
            noc.async_write_barrier();
            scale_ready_sem.set(INVALID);
        }
        rms_scale_src.pop_front(M_tiles);
    } else {
        scale_ready_sem.wait(VALID);
        scale_ready_sem.set(INVALID);
    }

    // Single push per core, by the CB's only producer. Compute is the only consumer and pops it.
    rms_scale.push_back(M_tiles);
    rms_local.pop_front(M_tiles);
}
