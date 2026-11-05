// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include <cstdint>

namespace fabric_api = tt::tt_fabric::mesh::experimental;

// The persistent op's reduce to the root chip, on one fabric sender core per link (see persistent.h for
// the iteration protocol). Its link's output cores bump sem_ar_ready once per iteration each. The senders
// are their own stage, a row behind the worker grid: they report ready on FE_SEM_READY (the control core's
// reduce-stage count), which the grid's go for iteration i waits on for iteration i - 2.
//
// Off the root, they have staged their partials here first (at the scratch's address, in their run
// order, in the iteration's half of the staging), and the sender unicasts each one to the root's matching
// output core, into the slot for this
// chip in the iteration's half of the root's scratch -- (iter & 1) * ring_size + my_chip -- with the
// same packet bumping that core's out_ready[iter & 1]. A half is reused two iterations later, once the
// root has summed it: the root's sender multicasts one credit per iteration back to every chip, so
// iteration i waits for i - 1 credits (from iteration 2 on), which in steady state are long there.
//
// On the root, the bumps mean the link's output cores have summed the iteration, and the sender
// multicasts the credit, down the whole-mesh branches, to this core's credit semaphore on every chip.
//
// Compile-time args:
//   0: my_chip   1: is_root   2: num_connections (1 off the root: the route to it; the branches on it)
//   3: core_bytes (one output core's partial)   4: sem_ar_ready   5: ring_size   6: sem_go
//   7: cb_scratch (this core's staging, at the scratch address)   8: staging_slots (output cores per link,
//      the stride of the staging's two halves)
//
// Common runtime args:
//   0: scratch address   1: out_ready[0]   2: out_ready[1]   3: credit (GlobalSemaphore, on the senders)
//
// Runtime args: num_cores, their NoC xs, their NoC ys, then on the root the hops (E, W, N, S) of each
// branch, then the fabric connections.

constexpr uint32_t my_chip = get_compile_time_arg_val(0);
constexpr bool is_root = get_compile_time_arg_val(1) != 0;
constexpr uint32_t num_connections = get_compile_time_arg_val(2);
constexpr uint32_t core_bytes = get_compile_time_arg_val(3);
constexpr uint32_t sem_ar_ready_id = get_compile_time_arg_val(4);
constexpr uint32_t ring_size = get_compile_time_arg_val(5);
constexpr uint32_t sem_go_id = get_compile_time_arg_val(6);
constexpr uint32_t cb_scratch_id = get_compile_time_arg_val(7);
constexpr uint32_t staging_slots = get_compile_time_arg_val(8);
constexpr uint32_t kStopBit = 0x80000000u;

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    fabric_api::MeshMcastRange ranges[num_connections];
    if constexpr (is_root) {
        for (uint32_t c = 0; c < num_connections; ++c) {
            const uint8_t e = get_arg_val<uint32_t>(arg_idx++);
            const uint8_t w = get_arg_val<uint32_t>(arg_idx++);
            const uint8_t n = get_arg_val<uint32_t>(arg_idx++);
            const uint8_t s = get_arg_val<uint32_t>(arg_idx++);
            ranges[c] = fabric_api::MeshMcastRange{e, w, n, s};
        }
    }
    const uint32_t scratch_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_ready_addr[2] = {get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2)};
    const uint32_t credit_addr = get_common_arg_val<uint32_t>(3);
    auto* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_addr);

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_idx);
    const uint8_t route_id = PacketHeaderPool::allocate_header_n(num_connections);
    if constexpr (!is_root) {
        fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
            UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush |
            UnicastFusedAtomicIncUpdateMask::PayloadSize>(
            fabric_connection,
            route_id,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{0, 0, 1, false},
            core_bytes);
    }
    const uint32_t staging_addr = get_write_ptr(cb_scratch_id);

    Noc noc;
    Semaphore<> ar_ready(sem_ar_ready_id);
    auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go_id));
    uint32_t iter = 0;
    for (;; ++iter) {
        if (iter > 0) {
            noc_semaphore_inc(get_noc_addr(FE_CTRL_X, FE_CTRL_Y, get_semaphore(FE_SEM_READY)), 1);
            noc_async_atomic_barrier();
        }
        uint32_t v;
        do {
            invalidate_l1_cache();
            v = *go;
        } while ((v & ~kStopBit) < iter + 1u);
        if ((v & kStopBit) != 0u) {
            break;
        }

        {
            DeviceZoneScopedN("FE_RED_PARTIALS_WAIT");
            ar_ready.wait_min(num_cores * (iter + 1u));
        }
        DeviceZoneScopedN("FE_RED_SEND");
        if constexpr (is_root) {
            noc.async_writes_flushed();
            fabric_api::fabric_multicast_noc_unicast_atomic_inc(
                fabric_connection,
                route_id,
                ranges,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    safe_get_noc_addr(my_x[noc_index], my_y[noc_index], credit_addr, 0), 1, true});
        } else {
            if (iter >= 2u) {
                noc_semaphore_wait_min(credit, iter - 1u);
            }
            const uint32_t parity = iter & 1u;
            const uint32_t slot_addr = scratch_addr + (parity * ring_size + my_chip) * core_bytes;
            const uint32_t staged_addr = staging_addr + parity * staging_slots * core_bytes;
            for (uint32_t core = 0; core < num_cores; ++core) {
                // The headers are rewritten in place, so the previous packet's must have left.
                noc.async_writes_flushed();
                fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
                    UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
                    fabric_connection,
                    route_id,
                    staged_addr + core * core_bytes,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        safe_get_noc_addr(core_noc_x[core], core_noc_y[core], slot_addr, 0),
                        safe_get_noc_addr(core_noc_x[core], core_noc_y[core], out_ready_addr[parity], 0),
                        1,
                        false});
            }
        }
        // The staging is refilled next iteration, so every packet has to have left it.
        noc.async_writes_flushed();
    }
    noc.async_write_barrier();
    close_connections(fabric_connection);
    if constexpr (!is_root) {
        // The root sent one credit per iteration; the last ones may still be in flight.
        noc_semaphore_wait_min(credit, iter);
        *credit = 0;
    }
}
