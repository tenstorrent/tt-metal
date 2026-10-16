// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — port sender (BRISC, NoC1): fence_ready + send_unit (adapted from fabric_reduce_scatter's
// port sender).
//
// fence_ready: tell the peer's opposite port (which writes into this chip) that this chip has started the call, then,
// if this port sends anything, wait for the same word from the downstream chip and consume it.
// send_unit: every chunk of cb_port_chunks (in the reader's order) is one fabric packet into the downstream matmul
// core's zone slot (same logical core, same staging address on every chip); a core's last chunk carries the fused
// increment of that core's arrival counter (one ordered sender per counter: this port). Headers rotate over `group`
// pre-routed headers; flush once per batch, popping in the reader's batch boundaries.
//
// Raw-API note (design "Helpers considered and rejected"): the CCL helper library is unmeasured and routes by 1D hop
// count with a classic step schedule; the measured raw sender of fabric_reduce_scatter is the reference.

#include <cstdint>
#ifdef MRS_ZONES  // perf instrumentation only
#include <tools/profiler/kernel_profiler.hpp>
#define MRS_ZONE(name) DeviceZoneScopedN(name)
#else
#define MRS_ZONE(name)
#endif
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

// One hop: 2D fabrics route by destination fabric node, 1D fabrics by hop count (ROUTING_MODE from the build).
inline void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

void kernel_main() {
    constexpr uint32_t cb_port_chunks = get_compile_time_arg_val(0);
    constexpr uint32_t P = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t group = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;
    constexpr uint32_t chunks_per_core = (P + chunk_tiles - 1) / chunk_tiles;
    constexpr uint32_t num_headers = group;
    constexpr uint32_t slot_bytes = P * tile_bytes;

    size_t arg = 0;
    const uint32_t zone_base = get_arg_val<uint32_t>(arg++);     // downstream zone of this direction (staging + offset)
    const uint32_t arrival_addr = get_arg_val<uint32_t>(arg++);  // global arr_d on the downstream matmul cores
    const uint32_t ready_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t send_ready = get_arg_val<uint32_t>(arg++);
    const uint32_t ready_x = get_arg_val<uint32_t>(arg++);
    const uint32_t ready_y = get_arg_val<uint32_t>(arg++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint32_t num_cores = get_arg_val<uint32_t>(arg++);
    const uint32_t num_units = get_arg_val<uint32_t>(arg++);
    const uint32_t cores_idx = arg;
    const uint32_t units_idx = cores_idx + num_cores;
    arg = units_idx + num_units;  // fabric connection args follow

    if (num_units == 0 && !send_ready) {
        return;
    }
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
    for (uint32_t h = 0; h < num_headers; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
    }
    conn.open();

    // fence_ready
    if (send_ready) {
        hdrs[0]->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
    }
    if (num_units > 0) {
        MRS_ZONE("READY_WAIT");
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
        noc_semaphore_wait_min(ready, 1);
        noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
        noc_async_atomic_barrier();
    }

    // send_unit, mirroring the reader's walk and batch boundaries
    uint32_t rpos = 0, u = 0, lim = group, h = 0;
    auto pop_batch = [&]() {
        noc_async_writes_flushed();  // sources + headers of the in-flight packets have left L1
        cb_pop_front(cb_port_chunks, u * chunk_tiles);
        rpos += u;
        u = 0;
        lim = group - rpos % group;
    };
    for (uint32_t un = 0; un < num_units; ++un) {
        MRS_ZONE("SEND_UNIT");
        const uint32_t unit = get_arg_val<uint32_t>(units_idx + un);
        const uint32_t half_mask = (unit >> 8) & 0x3;
        const uint32_t slot0 = (unit >> 16) & 0xFF;
        const uint32_t slot1 = (unit >> 24) & 0xFF;
        const uint32_t first_half = (unit >> 10) & 0x1;
        for (uint32_t walk = 0; walk < 2 * num_cores; ++walk) {  // the reader's walk: first_half's cores first
            const uint32_t hh = first_half ^ (walk >= num_cores ? 1u : 0u);
            const uint32_t i = walk >= num_cores ? walk - num_cores : walk;
            const uint32_t cw = get_arg_val<uint32_t>(cores_idx + i);
            const uint32_t hf = (cw >> 16) & 0xF;
            if (!((half_mask >> hh) & 1) || hf != hh) {
                continue;
            }
            const uint32_t x = cw & 0xFF, y = (cw >> 8) & 0xFF;
            const uint32_t dst_base = zone_base + (hf ? slot1 : slot0) * slot_bytes;
            const uint64_t ctr = get_noc_addr(x, y, arrival_addr);
            for (uint32_t ci = 0; ci < chunks_per_core; ++ci) {
                {
                    MRS_ZONE("CHUNK_WAIT");
                    cb_wait_front(cb_port_chunks, chunk_tiles * (u + 1));
                }
                const uint32_t src = get_read_ptr(cb_port_chunks) + u * chunk_bytes;
                const uint32_t n = (P - ci * chunk_tiles) < chunk_tiles ? (P - ci * chunk_tiles) : chunk_tiles;
                const uint32_t bytes = n * tile_bytes;
                const uint64_t dst = get_noc_addr(x, y, dst_base + ci * chunk_bytes);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == num_headers) ? 0 : h + 1;
                if (ci + 1 == chunks_per_core) {  // a core's last chunk carries its arrival increment
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, ctr, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                if (++u == lim) {
                    pop_batch();
                }
            }
        }
        if (u > 0) {
            pop_batch();  // end of unit (the reader flushed there too)
        }
    }
    conn.close();
}
