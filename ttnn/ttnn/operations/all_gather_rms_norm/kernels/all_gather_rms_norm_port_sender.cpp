// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm port sender (BRISC / NoC1) — TRANSPORT (adapted from examples/fabric_all_gather).
//
// ready_fence: this program running on this chip means every earlier command here is done, so the neighbour's
//   port that writes into this chip may start: send it one ready increment; then send nothing until this port's
//   own ready (from the neighbour it writes into) has arrived, and consume it (each call sends / consumes exactly
//   one ready per sending port: the counters are 0 at every call boundary).
// send_entry: every packet of cb_port_stage goes to the SAME core and cb_x offset on the neighbour (mirror layout);
//   the last packet of every entry is a fused write + atomic increment of the downstream port's arrival counter
//   (one increment per entry ~ INC_EVERY packets: the receiving router stalls until each increment's write landed).
//   Batches of `group` packets per flush (sources + headers left L1), then the batch is popped.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "all_gather_rms_norm_common.hpp"

using namespace tt::tt_fabric;
using namespace agrms_cb;

constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
constexpr uint32_t chunk_pages = get_compile_time_arg_val(1);
constexpr uint32_t group = get_compile_time_arg_val(2);
constexpr uint32_t chunk_bytes = chunk_pages * tile_bytes;
constexpr uint32_t num_headers = group;

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

FORCE_INLINE uint32_t arg(uint32_t i) { return get_arg_val<uint32_t>(i); }

void kernel_main() {
    const uint32_t arrival_addr = arg(0);
    const uint32_t ready_addr = arg(1);
    const uint32_t send_ready = arg(2);
    const uint32_t ready_x = arg(3);
    const uint32_t ready_y = arg(4);
    if (arg(9) == 0) {
        return;  // idle port on this chip, or receive-only (no connection)
    }
    const uint32_t peer_x = arg(5);
    const uint32_t peer_y = arg(6);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(arg(7));
    const uint16_t dst_chip_id = static_cast<uint16_t>(arg(8));
    uint32_t chunks_left = arg(10);
    const uint32_t num_out = arg(11);
    uint32_t a = 12;
    uint32_t fab = a;
    for (uint32_t e = 0; e < num_out; ++e) {
        fab += 1 + 2 * arg(fab);
    }
    size_t fa = fab;
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(fa);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
    for (uint32_t h = 0; h < num_headers; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
    }
    conn.open();
    if (send_ready) {
        hdrs[0]->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
    }
    if (num_out > 0) {
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
        noc_semaphore_wait_min(ready, 1);
        noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
        noc_async_atomic_barrier();
    }
    const uint32_t x_base = get_write_ptr(cb_x);
    const uint64_t arrival_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
    uint32_t h = 0, unflushed = 0;
    for (uint32_t e = 0; e < num_out; ++e) {
        const uint32_t n_runs = arg(a);
        a += 1;
        for (uint32_t r = 0; r < n_runs; ++r) {
            const uint32_t xy = arg(a + 2 * r);
            const uint32_t offn = arg(a + 2 * r + 1);
            const uint32_t off = offn & 0xFFFF;
            const uint32_t n = offn >> 16;
            for (uint32_t t = 0; t < n; t += chunk_pages) {
                const uint32_t pieces = (n - t) < chunk_pages ? (n - t) : chunk_pages;
                cb_wait_front(cb_stage, chunk_pages * (unflushed + 1));
                const uint32_t src = get_read_ptr(cb_stage) + unflushed * chunk_bytes;
                const uint32_t bytes = pieces * tile_bytes;
                const uint64_t dst = get_noc_addr(xy & 0xFFFF, xy >> 16, x_base + (off + t) * tile_bytes);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == num_headers) ? 0 : h + 1;
                const bool entry_last = (r + 1 == n_runs) && (t + chunk_pages >= n);
                if (entry_last) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                --chunks_left;
                if (++unflushed == group || chunks_left == 0) {
                    noc_async_writes_flushed();
                    cb_pop_front(cb_stage, chunk_pages * unflushed);
                    unflushed = 0;
                }
            }
        }
        a += 2 * n_runs;
    }
    conn.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
