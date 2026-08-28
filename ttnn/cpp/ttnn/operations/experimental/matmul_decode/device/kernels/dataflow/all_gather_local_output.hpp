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
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"

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
//   [1] primary output shard's noc0 x
//   [2] primary output shard's noc0 y
//   [3] tile offset within the primary output shard
//   [4] this worker's noc0 x
//   [5] this worker's noc0 y
//   [6 .. 5+2*(num_shards-1)] extra shard dest noc x,y  (num_shards > 1)
//   next: num_connections
//   then: fabric RoutingPlaneConnectionManager args
//
// Open EDM as soon as the sender writer starts. Opening after K-reduce drops
// start-barrier incs that already arrived at this chip's ethernet.
template <uint32_t NumBuffers>
struct AllGatherMuxConnection {
    bool valid = false;
    bool termination_master = false;
    uint32_t mux_x = 0, mux_y = 0;
    uint32_t channel_base = 0, connection_info = 0, handshake = 0, flow_control = 0, buffer_index = 0, channel_id = 0;
    uint32_t termination_sync = 0, mux_status = 0, local_flow_control = 0, local_teardown = 0, local_buffer_index = 0;
    uint32_t termination_master_x = 0, termination_master_y = 0, num_clients = 0;
    tt::tt_fabric::WorkerToFabricMuxSender<NumBuffers> sender;
};

template <uint32_t NumBuffers>
inline AllGatherMuxConnection<NumBuffers> all_gather_parse_mux_connection(uint32_t& arg_idx) {
    AllGatherMuxConnection<NumBuffers> c;
    c.valid = get_arg_val<uint32_t>(arg_idx++) != 0;
    c.termination_master = get_arg_val<uint32_t>(arg_idx++) != 0;
    c.mux_x = get_arg_val<uint32_t>(arg_idx++);
    c.mux_y = get_arg_val<uint32_t>(arg_idx++);
    c.channel_base = get_arg_val<uint32_t>(arg_idx++);
    c.connection_info = get_arg_val<uint32_t>(arg_idx++);
    c.handshake = get_arg_val<uint32_t>(arg_idx++);
    c.flow_control = get_arg_val<uint32_t>(arg_idx++);
    c.buffer_index = get_arg_val<uint32_t>(arg_idx++);
    c.channel_id = get_arg_val<uint32_t>(arg_idx++);
    c.termination_sync = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    c.mux_status = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    c.local_flow_control = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    c.local_teardown = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    c.local_buffer_index = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    c.termination_master_x = get_arg_val<uint32_t>(arg_idx++);
    c.termination_master_y = get_arg_val<uint32_t>(arg_idx++);
    c.num_clients = get_arg_val<uint32_t>(arg_idx++);
    return c;
}

template <uint32_t NumBuffers>
inline void all_gather_open_mux_connections(AllGatherMuxConnection<NumBuffers>* connections) {
    for (uint32_t dir = 0; dir < 2; ++dir) {
        auto& c = connections[dir];
        if (!c.valid) {
            continue;
        }
        c.sender = tt::tt_fabric::build_connection_to_fabric_endpoint<NumBuffers>(
            c.mux_x,
            c.mux_y,
            c.channel_id,
            NumBuffers,
            get_named_compile_time_arg_val("ag_mux_buffer_size"),
            c.channel_base,
            c.connection_info,
            c.handshake,
            c.flow_control,
            c.buffer_index,
            c.local_flow_control,
            c.local_teardown,
            c.local_buffer_index);
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            c.mux_x, c.mux_y, get_named_compile_time_arg_val("ag_mux_status"), c.mux_status);
        tt::tt_fabric::fabric_client_connect(c.sender);
    }
}

inline void all_gather_open_connections(tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection) {
    using namespace tt::tt_fabric::linear::experimental;

    constexpr uint32_t ag_rt_arg_base = get_named_compile_time_arg_val("ag_rt_arg_base");
    constexpr uint32_t num_shards = get_named_compile_time_arg_val("ag_num_shards");
    uint32_t arg_idx = ag_rt_arg_base + 4 + 2 * num_shards;
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
    size_t fabric_arg_idx = arg_idx;
    DPRINT("[AGW] before open_connections nconn={}\n", num_connections);
    open_connections(fabric_connection, num_connections, fabric_arg_idx);
    DPRINT("[AGW] after open_connections\n");
}

template <uint32_t NumBuffers>
inline void all_gather_local_output(
    tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection,
    AllGatherMuxConnection<NumBuffers>* mux_connections = nullptr) {
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
    constexpr bool use_mux = get_named_compile_time_arg_val("ag_use_mux") != 0;
    constexpr uint32_t mux_buffer_size = get_named_compile_time_arg_val("ag_mux_buffer_size");
    constexpr uint32_t gathered_n_tiles = N_tiles_per_core * ring_size;
    constexpr uint32_t block_num_tiles = M_tiles * N_tiles_per_core;

    uint32_t arg_idx = ag_rt_arg_base;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    std::array<uint32_t, num_shards> shard_noc_x{};
    std::array<uint32_t, num_shards> shard_noc_y{};
    shard_noc_x[0] = get_arg_val<uint32_t>(arg_idx++);
    shard_noc_y[0] = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_tile_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_noc_y = get_arg_val<uint32_t>(arg_idx++);
    if constexpr (num_shards > 1) {
        for (uint32_t s = 1; s < num_shards; ++s) {
            shard_noc_x[s] = get_arg_val<uint32_t>(arg_idx++);
            shard_noc_y[s] = get_arg_val<uint32_t>(arg_idx++);
        }
    }
    uint32_t num_connections = 0;
    if constexpr (use_mux) {
        arg_idx += 2 * 17;
    } else {
        num_connections = get_arg_val<uint32_t>(arg_idx++);
    }

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
    if constexpr (!use_mux) {
        if (ranges[0] == 0) {
            starts[0] = starts[1];
            ranges[0] = ranges[1];
        }
    }

    uint8_t write_route_id = 0;
    uint8_t sem_route_id = 0;
    if constexpr (!use_mux) {
        write_route_id = PacketHeaderPool::allocate_header_n(num_connections);
        sem_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    }
    std::array<volatile PACKET_HEADER_TYPE*, 2> mux_write_headers{};
    std::array<volatile PACKET_HEADER_TYPE*, 2> mux_sem_headers{};
    if constexpr (use_mux) {
        for (uint32_t dir = 0; dir < 2; ++dir) {
            mux_write_headers[dir] = PacketHeaderPool::allocate_header();
            mux_sem_headers[dir] = PacketHeaderPool::allocate_header();
        }
    }
    auto send_atomic_inc = [&](uint32_t dir, uint64_t dest) {
        if constexpr (use_mux) {
            if (mux_connections[dir].valid) {
                // A multicast packet injected through a worker mux can be
                // replicated only on the first forwarding hop. Send one
                // unicast packet for each hop so every remote writer receives
                // exactly one barrier increment.
                for (uint32_t hop = 1; hop <= ranges[dir]; ++hop) {
                    fabric_unicast_noc_unicast_atomic_inc(
                        &mux_connections[dir].sender,
                        mux_sem_headers[dir],
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{dest, 1},
                        static_cast<uint8_t>(hop));
                    // The packet header is reused for the next hop. Wait until
                    // the NOC has copied it into the mux staging slot before
                    // changing its route fields.
                    noc_async_writes_flushed();
                }
            }
        } else {
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_connection, sem_route_id, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{dest, 0});
        }
    };

    if constexpr (!use_mux) {
        fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            fabric_connection, write_route_id, starts.data(), ranges.data(), nullptr, static_cast<uint16_t>(tile_size));
    }

    if constexpr (!use_mux) {
        fabric_multicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            fabric_connection,
            sem_route_id,
            starts.data(),
            ranges.data(),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});
    }

    // Fabric packets always carry a NOC_0 address (remote EDM writes on NOC_0). Local
    // noc_async_write / noc_semaphore_inc must use this kernel's NOC (the writer is on
    // NOC_1); encoding a NOC_0 dest on a NOC_1 kernel hangs.
    // Two semaphores (same split as broadcast_rm_writer): a fast chip can send its
    // completion inc while others are still on the start barrier. Sharing one sem
    // means that inc is counted as a start signal and then wiped by the reset.
    const uint64_t fabric_barrier_noc_addr = safe_get_noc_addr(worker_noc_x, worker_noc_y, barrier_sem_addr, 0);
    const uint64_t fabric_ready_noc_addr = safe_get_noc_addr(worker_noc_x, worker_noc_y, out_ready_sem_addr, 0);
    const uint64_t local_ready_noc_addr = safe_get_noc_addr(worker_noc_x, worker_noc_y, out_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* barrier_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr);
    volatile tt_l1_ptr uint32_t* ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_addr);

    constexpr uint32_t num_remote_targets = range_hops_forward + range_hops_backward;
    DPRINT("[AGW] start_barrier mcast-inc remotes={} sem={}\n", num_remote_targets, *barrier_ptr);
    if constexpr (use_mux) {
        send_atomic_inc(0, fabric_barrier_noc_addr);
        send_atomic_inc(1, fabric_barrier_noc_addr);
    } else {
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection,
            sem_route_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{fabric_barrier_noc_addr, 0});
    }
    DPRINT("[AGW] before start_barrier wait_min need={} sem={}\n", num_remote_targets, *barrier_ptr);
    noc_semaphore_wait_min(barrier_ptr, num_remote_targets);
    DPRINT("[AGW] after start_barrier wait_min sem={}\n", *barrier_ptr);
    noc_semaphore_set(barrier_ptr, 0);

    Noc noc;
    auto send_column = [&](uint32_t l1_read_addr, uint32_t dest_x, uint32_t dest_y, uint32_t bw, bool do_local) {
        for (uint32_t mt = 0; mt < M_tiles; ++mt) {
            const uint32_t tile_id = mt * gathered_n_tiles + output_tile_offset + bw;
            const uint32_t dest_l1 = out_addr + tile_id * tile_size;
            const uint64_t fabric_dest_noc_addr = safe_get_noc_addr(dest_x, dest_y, dest_l1, 0);
            if (do_local) {
                const uint64_t local_dest_noc_addr = safe_get_noc_addr(dest_x, dest_y, dest_l1);
                noc_async_write(l1_read_addr, local_dest_noc_addr, tile_size);
                noc.async_writes_flushed();
            }
            if constexpr (use_mux) {
                for (uint32_t dir = 0; dir < 2; ++dir) {
                    if (mux_connections[dir].valid) {
                        // A worker mux has one outgoing fabric connection.
                        // Explicit unicast packets make each remote chip's
                        // destination deterministic and avoid relying on
                        // multicast replication after mux injection.
                        for (uint32_t hop = 1; hop <= ranges[dir]; ++hop) {
                            fabric_unicast_noc_unicast_write(
                                &mux_connections[dir].sender,
                                mux_write_headers[dir],
                                l1_read_addr,
                                tile_size,
                                tt::tt_fabric::NocUnicastCommandHeader{fabric_dest_noc_addr},
                                static_cast<uint8_t>(hop));
                            // Protect the reused header and source tile until
                            // their copies into the mux staging slot complete.
                            noc.async_writes_flushed();
                        }
                    }
                }
            } else {
                fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                    fabric_connection,
                    write_route_id,
                    l1_read_addr,
                    tt::tt_fabric::NocUnicastCommandHeader{fabric_dest_noc_addr},
                    static_cast<uint16_t>(0u));
            }
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
    if constexpr (use_mux) {
        send_atomic_inc(0, fabric_ready_noc_addr);
        send_atomic_inc(1, fabric_ready_noc_addr);
    } else {
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection, sem_route_id, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{fabric_ready_noc_addr, 0});
    }
    noc_semaphore_inc(local_ready_noc_addr, 1);
    DPRINT("[AGW] before completion wait_min need={} sem={}\n", ring_size, *ready_ptr);
    noc_semaphore_wait_min(ready_ptr, ring_size);
    DPRINT("[AGW] after completion wait_min sem={}\n", *ready_ptr);
    noc_semaphore_set(ready_ptr, 0);

    DPRINT("[AGW] before close_connections\n");
    if constexpr (use_mux) {
        for (uint32_t dir = 0; dir < 2; ++dir) {
            if (!mux_connections[dir].valid) {
                continue;
            }
            tt::tt_fabric::fabric_client_disconnect(mux_connections[dir].sender);
            if (mux_connections[dir].termination_master) {
                auto* sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mux_connections[dir].termination_sync);
                noc_semaphore_wait(sync_ptr, mux_connections[dir].num_clients - 1);
                noc_inline_dw_write(
                    safe_get_noc_addr(
                        mux_connections[dir].mux_x,
                        mux_connections[dir].mux_y,
                        get_named_compile_time_arg_val("ag_mux_termination")),
                    tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE);
                noc_async_write_barrier();
            } else {
                noc_semaphore_inc(
                    safe_get_noc_addr(
                        mux_connections[dir].termination_master_x,
                        mux_connections[dir].termination_master_y,
                        mux_connections[dir].termination_sync),
                    1);
                noc_async_atomic_barrier();
            }
        }
    } else {
        close_connections(fabric_connection);
    }
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
