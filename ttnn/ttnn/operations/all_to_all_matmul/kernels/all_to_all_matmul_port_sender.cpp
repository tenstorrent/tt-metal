// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_to_all_matmul port sender (RISCV_0 / NoC1), one per (direction, link) port core, on its own fabric
// connection (routing plane = link).
//
//   send_readies  header-only atomic incs of the ready counter of every port that writes into this chip (this
//                 program running means this chip's previous call, including the compute that read its landing
//                 slots, is done)
//   wait_readies  one ready per destination chip, then re-arm (-n)
//   send_entry    per entry: ceil(slot_tiles / packet_tiles) packets from cb_port_chunks straight into the destination
//                 compute core's landing slot (same L1 address on every chip); route by destination node
//                 (fabric_set_unicast_route: the routing table's own route, 1 or 2 hops); the entry's last packet is
//                 fused with an increment of the slot's arrival counter. One flush + pop per batch (mirrors the
//                 reader's batch rule).
// Packet headers: one ring of hdrs_per_dest pre-routed headers per destination chip; a header is rewritten only after
// a flush.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#if defined(ABLATE_ZONES)
#include "tools/profiler/kernel_profiler.hpp"
#define A2AMM_ZONE(n) DeviceZoneScopedN(n)
#else
#define A2AMM_ZONE(n)
#endif
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

namespace {
inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "all_to_all_matmul routes by destination node: a 2D fabric config is required"
#endif
}
constexpr uint32_t READY_WORDS = 3;  // destination index, target port x, y
constexpr uint32_t ENTRY_WORDS = 5;  // destination index, core x, y, slot address, arrival address
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_port_chunks = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t packet_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t batch_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t slot_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t max_headers = get_compile_time_arg_val(5);
    static_assert(max_headers >= 1 && max_headers <= 12, "packet header pool: <= 12 headers per RISC");
    constexpr uint32_t entry_chunks = (slot_tiles + packet_tiles - 1) / packet_tiles;
    constexpr uint32_t packet_bytes = packet_tiles * tile_bytes;

    size_t a = 0;
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_ready_wait = get_arg_val<uint32_t>(a++);
    const bool has_conn = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t n_dest = get_arg_val<uint32_t>(a++);
    const uint32_t hdrs_per_dest = get_arg_val<uint32_t>(a++);
    const uint32_t dest_idx = a;
    a += 2 * n_dest;
    const uint32_t n_ready_out = get_arg_val<uint32_t>(a++);
    const uint32_t ready_idx = a;
    a += READY_WORDS * n_ready_out;
    const uint32_t n_entries = get_arg_val<uint32_t>(a++);
    const uint32_t entry_idx = a;
    a += ENTRY_WORDS * n_entries;

    if (!has_conn) {
        return;
    }
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[max_headers];
    ASSERT(n_dest * hdrs_per_dest <= max_headers);
    for (uint32_t d = 0; d < n_dest; ++d) {
        const uint16_t chip = static_cast<uint16_t>(get_arg_val<uint32_t>(dest_idx + 2 * d));
        const uint16_t mesh = static_cast<uint16_t>(get_arg_val<uint32_t>(dest_idx + 2 * d + 1));
        for (uint32_t i = 0; i < hdrs_per_dest; ++i) {
            hdrs[d * hdrs_per_dest + i] = PacketHeaderPool::allocate_header();
            route_to(hdrs[d * hdrs_per_dest + i], chip, mesh);
        }
    }
    conn.open();

    // ---- ready fence ----
    {
        A2AMM_ZONE("psd_fence");
        for (uint32_t k = 0; k < n_ready_out; ++k) {
            const uint32_t t = ready_idx + k * READY_WORDS;
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[get_arg_val<uint32_t>(t) * hdrs_per_dest];
            hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                get_noc_addr(get_arg_val<uint32_t>(t + 1), get_arg_val<uint32_t>(t + 2), ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
        }
        if (n_ready_wait > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, n_ready_wait);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - n_ready_wait);
            noc_async_atomic_barrier();
        }
    }
    // ---- entries ----
    A2AMM_ZONE("psd_send");
    uint32_t used[max_headers];  // headers of each destination used since the last flush
    for (uint32_t d = 0; d < n_dest; ++d) {
        used[d] = 0;
    }
    auto flushed = [&]() {
        noc_async_writes_flushed();  // packet sources + headers have left L1
        for (uint32_t d = 0; d < n_dest; ++d) {
            used[d] = 0;
        }
    };
    uint32_t pos = 0;
    for (uint32_t e = 0; e < n_entries; ++e) {
        const uint32_t rec = entry_idx + e * ENTRY_WORDS;
        const uint32_t d = get_arg_val<uint32_t>(rec);
        const uint32_t x = get_arg_val<uint32_t>(rec + 1);
        const uint32_t y = get_arg_val<uint32_t>(rec + 2);
        const uint32_t slot_addr = get_arg_val<uint32_t>(rec + 3);
        const uint64_t arrival_noc = get_noc_addr(x, y, get_arg_val<uint32_t>(rec + 4));
        uint32_t tile = 0;
        for (uint32_t c = 0; c < entry_chunks;) {
            const uint32_t room = batch_chunks - pos % batch_chunks;
            const uint32_t need = (entry_chunks - c) < room ? (entry_chunks - c) : room;
            cb_wait_front(cb_port_chunks, need);
            const uint32_t rp = get_read_ptr(cb_port_chunks);
            for (uint32_t i = 0; i < need; ++i) {
                const uint32_t left = slot_tiles - tile;
                const uint32_t bytes = (left < packet_tiles ? left : packet_tiles) * tile_bytes;
                if (used[d] == hdrs_per_dest) {
                    flushed();
                }
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[d * hdrs_per_dest + used[d]++];
                const uint64_t dst = get_noc_addr(x, y, slot_addr + tile * tile_bytes);
                if (c + i + 1 == entry_chunks) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(rp + i * packet_bytes, bytes, reinterpret_cast<uint32_t>(hdr));
                tile += packet_tiles;
            }
            flushed();
            cb_pop_front(cb_port_chunks, need);
            pos += need;
            c += need;
        }
    }
    noc_async_writes_flushed();
    conn.close();
}
