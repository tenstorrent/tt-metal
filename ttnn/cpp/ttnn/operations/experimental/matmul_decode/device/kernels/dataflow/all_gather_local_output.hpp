// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

#include <cstddef>
#include <cstdint>
#include <array>

// Fabric all-gather of local matmul output N-shard(s). Compute publishes each
// K-complete N-column with push_back(M_tiles) (column-major in cb_out).
//
// Full-width (ag_num_shards == 1): this core wait_fronts cb_out, writes the
// gathered slot locally, and fabric-mcasts onto the matching worker of every
// other device.
//
// Partial-width (ag_num_shards > 1): only N-block 0 opens fabric. It sends its
// own cb_out, then fabric-sends the other N-block shards from cb_ag_staging
// (those bases copy here via forward_partial_shard_to_sender). Dest NOC coords
// for shard s>0 follow this core's my_x/my_y args.
//
// Runtime args starting at ag_rt_arg_base:
//   [0] output buffer address
//   [1] this core's noc0 x
//   [2] this core's noc0 y
//   [3 .. 2+2*(num_shards-1)] extra shard dest noc x,y  (num_shards > 1)
//   next: num_connections
//   then: fabric RoutingPlaneConnectionManager args
//
// Open EDM as soon as the sender writer starts. Opening after K-reduce drops
// start-barrier incs that already arrived at this chip's ethernet.
inline void all_gather_open_connections(tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection) {
    using namespace tt::tt_fabric::linear::experimental;

    constexpr uint32_t ag_rt_arg_base = get_named_compile_time_arg_val("ag_rt_arg_base");
    constexpr uint32_t num_shards = get_named_compile_time_arg_val("ag_num_shards");
    uint32_t arg_idx = ag_rt_arg_base + 1 + 2 * num_shards;
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
    size_t fabric_arg_idx = arg_idx;
    DPRINT("[AGW] before open_connections nconn={}\n", num_connections);
    open_connections(fabric_connection, num_connections, fabric_arg_idx);
    DPRINT("[AGW] after open_connections\n");
}

inline void all_gather_local_output(tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection) {
    using namespace tt::tt_fabric::linear::experimental;

    constexpr uint32_t cb_out_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t M_tiles = get_named_compile_time_arg_val("ag_M_tiles");
    constexpr uint32_t N_tiles_per_core = get_named_compile_time_arg_val("ag_N_tiles_per_core");
    constexpr uint32_t ring_index = get_named_compile_time_arg_val("ag_ring_index");
    constexpr uint32_t ring_size = get_named_compile_time_arg_val("ag_ring_size");
    constexpr uint32_t out_ready_sem_addr = get_named_compile_time_arg_val("ag_out_ready_sem_id");
    constexpr uint32_t barrier_sem_addr = get_named_compile_time_arg_val("ag_barrier_sem_id");
    constexpr uint32_t start_distance_in_hops_forward = get_named_compile_time_arg_val("ag_start_fwd");
    constexpr uint32_t range_hops_forward = get_named_compile_time_arg_val("ag_range_fwd");
    constexpr uint32_t start_distance_in_hops_backward = get_named_compile_time_arg_val("ag_start_bwd");
    constexpr uint32_t range_hops_backward = get_named_compile_time_arg_val("ag_range_bwd");
    constexpr uint32_t ag_rt_arg_base = get_named_compile_time_arg_val("ag_rt_arg_base");
    constexpr uint32_t num_shards = get_named_compile_time_arg_val("ag_num_shards");
    constexpr uint32_t shard_sem_id = get_named_compile_time_arg_val("ag_shard_sem_id");
    constexpr uint32_t staging_cb_id = get_named_compile_time_arg_val("ag_staging_cb");
    constexpr uint32_t gathered_n_tiles = N_tiles_per_core * ring_size;
    constexpr uint32_t block_num_tiles = M_tiles * N_tiles_per_core;

    uint32_t arg_idx = ag_rt_arg_base;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    std::array<uint32_t, num_shards> shard_noc_x{};
    std::array<uint32_t, num_shards> shard_noc_y{};
    shard_noc_x[0] = get_arg_val<uint32_t>(arg_idx++);
    shard_noc_y[0] = get_arg_val<uint32_t>(arg_idx++);
    if constexpr (num_shards > 1) {
        for (uint32_t s = 1; s < num_shards; ++s) {
            shard_noc_x[s] = get_arg_val<uint32_t>(arg_idx++);
            shard_noc_y[s] = get_arg_val<uint32_t>(arg_idx++);
        }
    }
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);

    CircularBuffer out_cb(cb_out_id);
    const uint32_t tile_size = out_cb.get_tile_size();
    DPRINT(
        "[AGW] enter ring_idx={} ring_size={} nconn={} nshards={} start_fwd={} range_fwd={} start_bwd={} "
        "range_bwd={} M={} Npc={} out_addr={:x} noc=({},{}) ready_sem={:x} barrier_sem={:x}\n",
        ring_index,
        ring_size,
        num_connections,
        num_shards,
        start_distance_in_hops_forward,
        range_hops_forward,
        start_distance_in_hops_backward,
        range_hops_backward,
        M_tiles,
        N_tiles_per_core,
        out_addr,
        shard_noc_x[0],
        shard_noc_y[0],
        out_ready_sem_addr,
        barrier_sem_addr);

    std::array starts = {
        static_cast<uint8_t>(start_distance_in_hops_forward), static_cast<uint8_t>(start_distance_in_hops_backward)};
    std::array ranges = {static_cast<uint8_t>(range_hops_forward), static_cast<uint8_t>(range_hops_backward)};
    if (ranges[0] == 0) {
        starts[0] = starts[1];
        ranges[0] = ranges[1];
    }

    auto write_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    auto sem_route_id = PacketHeaderPool::allocate_header_n(num_connections);

    fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        fabric_connection, write_route_id, starts.data(), ranges.data(), nullptr, static_cast<uint16_t>(tile_size));

    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection,
        sem_route_id,
        starts.data(),
        ranges.data(),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});

    // Fabric packets always carry a NOC_0 address (remote EDM writes on NOC_0). Local
    // noc_async_write / noc_semaphore_inc must use this kernel's NOC (the writer is on
    // NOC_1); encoding a NOC_0 dest on a NOC_1 kernel hangs.
    // Two semaphores (same split as broadcast_rm_writer): a fast chip can send its
    // completion inc while others are still on the start barrier. Sharing one sem
    // means that inc is counted as a start signal and then wiped by the reset.
    const uint64_t fabric_barrier_noc_addr = safe_get_noc_addr(shard_noc_x[0], shard_noc_y[0], barrier_sem_addr, 0);
    const uint64_t fabric_ready_noc_addr = safe_get_noc_addr(shard_noc_x[0], shard_noc_y[0], out_ready_sem_addr, 0);
    const uint64_t local_ready_noc_addr = safe_get_noc_addr(shard_noc_x[0], shard_noc_y[0], out_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* barrier_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr);
    volatile tt_l1_ptr uint32_t* ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_addr);

    constexpr uint32_t num_remote_targets = range_hops_forward + range_hops_backward;
    DPRINT("[AGW] start_barrier mcast-inc remotes={} sem={}\n", num_remote_targets, *barrier_ptr);
    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection, sem_route_id, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{fabric_barrier_noc_addr, 0});
    DPRINT("[AGW] before start_barrier wait_min need={} sem={}\n", num_remote_targets, *barrier_ptr);
    noc_semaphore_wait_min(barrier_ptr, num_remote_targets);
    DPRINT("[AGW] after start_barrier wait_min sem={}\n", *barrier_ptr);
    noc_semaphore_set(barrier_ptr, 0);

    Noc noc;
    auto send_column = [&](uint32_t l1_read_addr, uint32_t dest_x, uint32_t dest_y, uint32_t bw, bool do_local) {
        for (uint32_t mt = 0; mt < M_tiles; ++mt) {
            const uint32_t tile_id = mt * gathered_n_tiles + ring_index * N_tiles_per_core + bw;
            const uint32_t dest_l1 = out_addr + tile_id * tile_size;
            const uint64_t fabric_dest_noc_addr = safe_get_noc_addr(dest_x, dest_y, dest_l1, 0);
            if (do_local) {
                const uint64_t local_dest_noc_addr = safe_get_noc_addr(dest_x, dest_y, dest_l1);
                noc_async_write(l1_read_addr, local_dest_noc_addr, tile_size);
                noc.async_writes_flushed();
            }
            fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                fabric_connection,
                write_route_id,
                l1_read_addr,
                tt::tt_fabric::NocUnicastCommandHeader{fabric_dest_noc_addr},
                static_cast<uint16_t>(0u));
            l1_read_addr += tile_size;
        }
        noc.async_writes_flushed();
    };

    for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
        DPRINT("[AGW] before wait_front col={} M={}\n", bw, M_tiles);
        out_cb.wait_front(M_tiles);
        DPRINT("[AGW] after wait_front col={}\n", bw);
        send_column(out_cb.get_read_ptr(), shard_noc_x[0], shard_noc_y[0], bw, true);
        out_cb.pop_front(M_tiles);
    }

    if constexpr (num_shards > 1) {
        CircularBuffer staging_cb(staging_cb_id);
        volatile tt_l1_ptr uint32_t* shard_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(shard_sem_id));
        DPRINT("[AGW] before shard wait_min need={} sem={}\n", num_shards - 1, *shard_ptr);
        noc_semaphore_wait_min(shard_ptr, num_shards - 1);
        DPRINT("[AGW] after shard wait_min sem={}\n", *shard_ptr);
        noc_semaphore_set(shard_ptr, 0);
        const uint32_t staging_base = staging_cb.get_write_ptr();
        const uint32_t block_size_bytes = block_num_tiles * tile_size;
        for (uint32_t s = 1; s < num_shards; ++s) {
            DPRINT("[AGW] send staged shard={} dest=({},{})\n", s, shard_noc_x[s], shard_noc_y[s]);
            uint32_t l1_read_addr = staging_base + s * block_size_bytes;
            for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
                send_column(l1_read_addr, shard_noc_x[s], shard_noc_y[s], bw, false);
                l1_read_addr += M_tiles * tile_size;
            }
        }
    }

    DPRINT("[AGW] completion mcast-inc sem={}\n", *ready_ptr);
    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection, sem_route_id, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{fabric_ready_noc_addr, 0});
    noc_semaphore_inc(local_ready_noc_addr, 1);
    DPRINT("[AGW] before completion wait_min need={} sem={}\n", ring_size, *ready_ptr);
    noc_semaphore_wait_min(ready_ptr, ring_size);
    DPRINT("[AGW] after completion wait_min sem={}\n", *ready_ptr);
    noc_semaphore_set(ready_ptr, 0);

    DPRINT("[AGW] before close_connections\n");
    close_connections(fabric_connection);
    noc.async_write_barrier();
    DPRINT("[AGW] done\n");
}

// Other partial-width N-block bases: local write of this core's gathered shard, copy the
// packed cb_out block to the fabric sender's staging slot, then credit the sender.
inline void forward_partial_shard_to_sender() {
    constexpr uint32_t cb_out_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t M_tiles = get_named_compile_time_arg_val("ag_M_tiles");
    constexpr uint32_t N_tiles_per_core = get_named_compile_time_arg_val("ag_N_tiles_per_core");
    constexpr uint32_t ring_index = get_named_compile_time_arg_val("ag_ring_index");
    constexpr uint32_t ring_size = get_named_compile_time_arg_val("ag_ring_size");
    constexpr uint32_t ag_rt_arg_base = get_named_compile_time_arg_val("ag_rt_arg_base");
    constexpr uint32_t shard_sem_id = get_named_compile_time_arg_val("ag_shard_sem_id");
    constexpr uint32_t staging_cb_id = get_named_compile_time_arg_val("ag_staging_cb");
    constexpr uint32_t gathered_n_tiles = N_tiles_per_core * ring_size;
    constexpr uint32_t block_num_tiles = M_tiles * N_tiles_per_core;

    const uint32_t n_idx = get_arg_val<uint32_t>(4);
    uint32_t arg_idx = ag_rt_arg_base;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sender_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sender_noc_y = get_arg_val<uint32_t>(arg_idx++);

    DPRINT(
        "[FWD] enter n_idx={} ring_idx={} M={} Npc={} out_addr={:x} noc=({},{}) sender=({},{})\n",
        n_idx,
        ring_index,
        M_tiles,
        N_tiles_per_core,
        out_addr,
        my_noc_x,
        my_noc_y,
        sender_noc_x,
        sender_noc_y);

    CircularBuffer out_cb(cb_out_id);
    CircularBuffer staging_cb(staging_cb_id);
    const uint32_t tile_size = out_cb.get_tile_size();
    const uint32_t block_size_bytes = block_num_tiles * tile_size;

    DPRINT("[FWD] before wait_front tiles={}\n", block_num_tiles);
    out_cb.wait_front(block_num_tiles);
    DPRINT("[FWD] after wait_front\n");
    uint32_t l1_read_addr = out_cb.get_read_ptr();
    Noc noc;
    for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
        for (uint32_t mt = 0; mt < M_tiles; ++mt) {
            const uint32_t tile_id = mt * gathered_n_tiles + ring_index * N_tiles_per_core + bw;
            const uint32_t dest_l1 = out_addr + tile_id * tile_size;
            noc_async_write(l1_read_addr, safe_get_noc_addr(my_noc_x, my_noc_y, dest_l1), tile_size);
            l1_read_addr += tile_size;
        }
    }
    noc.async_write_barrier();

    const uint32_t staging_dst = staging_cb.get_write_ptr() + n_idx * block_size_bytes;
    noc_async_write(
        out_cb.get_read_ptr(), safe_get_noc_addr(sender_noc_x, sender_noc_y, staging_dst), block_size_bytes);
    noc.async_write_barrier();
    out_cb.pop_front(block_num_tiles);

    DPRINT("[FWD] before shard sem inc sender=({},{})\n", sender_noc_x, sender_noc_y);
    noc_semaphore_inc(safe_get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(shard_sem_id)), 1);
    noc.async_atomic_barrier();
    DPRINT("[FWD] done n_idx={}\n", n_idx);
}
