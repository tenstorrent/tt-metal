// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include <cstdint>

#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/mesh/api.h"
namespace fabric_api = tt::tt_fabric::mesh::experimental;
using FabricRange = tt::tt_fabric::mesh::experimental::MeshMcastRange;
#else
#include "tt_metal/fabric/hw/inc/linear/api.h"
namespace fabric_api = tt::tt_fabric::linear::experimental;
using FabricRange = uint8_t;  // a 1D connection is one line: its single nonzero hop count
#endif

// The fused all-reduce on one fabric sender core, on a single DM processor so it also fits the router
// cores, whose other one runs the router. Its link's output cores stage their partials here (at the
// scratch's address on this core, in their run order) and bump sem_ar_ready once each; it then sends
// each partial as one packet down every branch -- E/W lines, plus N/S spines with E/W teeth when
// reducing over a 2D mesh -- to the owning core's slot for this chip on every other reduced chip,
// where the same packet bumps that core's out_ready.

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t num_connections = get_compile_time_arg_val(1);
constexpr uint32_t core_bytes = get_compile_time_arg_val(2);
constexpr uint32_t sem_ar_ready_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_scratch_id = get_compile_time_arg_val(4);

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    // Per connection, the hops per physical direction (E, W, N, S).
    FabricRange ranges[num_connections];
#ifndef FABRIC_2D
    uint8_t starts[num_connections];
#endif
    for (uint32_t c = 0; c < num_connections; ++c) {
        const uint8_t e = get_arg_val<uint32_t>(arg_idx++);
        const uint8_t w = get_arg_val<uint32_t>(arg_idx++);
        const uint8_t n = get_arg_val<uint32_t>(arg_idx++);
        const uint8_t s = get_arg_val<uint32_t>(arg_idx++);
#ifdef FABRIC_2D
        ranges[c] = FabricRange{e, w, n, s};
#else
        ranges[c] = e + w + n + s;
        starts[c] = 1;
#endif
    }
    // out_ready is a GlobalSemaphore at one address on every output core.
    const uint32_t out_ready_addr = get_common_arg_val<uint32_t>(0);

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_idx);

    // One header per connection, preset with everything but the two destination addresses.
    const uint8_t route_id = PacketHeaderPool::allocate_header_n(num_connections);
    fabric_api::fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<
        UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush |
        UnicastFusedAtomicIncUpdateMask::PayloadSize>(
        fabric_connection,
        route_id,
#ifndef FABRIC_2D
        starts,
#endif
        ranges,
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{0, 0, 1, false},
        core_bytes);

    // The scratch sits at one address on the output cores and here (cb_scratch is bound to it on both).
    const uint32_t staging_addr = get_write_ptr(cb_scratch_id);
    const uint32_t slot_addr = staging_addr + my_chip_id * core_bytes;

    Semaphore<> ar_ready(sem_ar_ready_id);
    ar_ready.wait(num_cores);
    ar_ready.set(0);

    Noc noc;
    for (uint32_t core = 0; core < num_cores; ++core) {
        // The headers are rewritten in place, so the previous packet's must have left.
        noc.async_writes_flushed();
        fabric_api::fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<
            UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
            fabric_connection,
            route_id,
            staging_addr + core * core_bytes,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                safe_get_noc_addr(core_noc_x[core], core_noc_y[core], slot_addr, 0),
                safe_get_noc_addr(core_noc_x[core], core_noc_y[core], out_ready_addr, 0),
                1,
                false});
    }
    noc.async_write_barrier();
    close_connections(fabric_connection);
}
