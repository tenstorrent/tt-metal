// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem port sender (row 0, RISCV_0 / NoC1): `ready_fence_block` + the send half of `send_segment_block`.
//
// Helper note: kernel_lib/ccl FabricStream builds 1-D hop-count routes only (ccl_helpers_dataflow.hpp
// unicast_route); this port's one connection also carries the antipodal (2-hop) destination, which on the snake
// ring must be routed by fabric node through the 2D routing table (fabric_set_unicast_route(hdr, chip, mesh)).
// One mechanism on the shared connection: the raw fabric API, as fabric_ring_exchange's port.cpp.
//
//   ready fence   one header-only atomic inc to every port that writes into this chip through my (direction, link);
//                 then wait for one ready from every chip I send to (they have started this call: their landing is
//                 free and their counters are re-armed)
//   segments      the reader's walk: per remote block of my direction, per row, per head: the hw tiles staged in
//                 cb_port_stage go to (destination chip, destination core, landing slot of the head-row), in packets
//                 of <= pkt_tiles tiles (pkt_tiles from the router's max payload); the last packet of the last head
//                 of a destination core in a row is fused with an increment of that core's arrive[p] by the tiles
//                 of all those heads (one in-order stream per (core, source): the per-row threshold is exact)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/ttnn/operations/q_stem/perf_experiments/qb_startup/q_stem_v/kernels/q_stem_common.hpp"

using namespace tt::tt_fabric;

namespace {
inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    static_assert(false, "q_stem needs a 2D fabric (FABRIC_2D_TORUS_XY): destinations are addressed by fabric node");
#endif
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_port_stage = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t pkt_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(3);
    constexpr uint32_t hdr_per_dest = get_compile_time_arg_val(4);
    constexpr uint32_t B = get_compile_time_arg_val(5);
    constexpr uint32_t qrt = get_compile_time_arg_val(6);
    constexpr uint32_t m = get_compile_time_arg_val(7);
    constexpr uint32_t G = get_compile_time_arg_val(8);
    constexpr uint32_t hw = get_compile_time_arg_val(9);
    constexpr uint32_t seg_bytes = hw * tile_bytes;
    constexpr uint32_t max_headers = get_compile_time_arg_val(10);  // PacketHeaderPool cap (host MAX_HEADERS)
    constexpr uint32_t ts = get_compile_time_arg_val(11);           // rows >= ts land after the local heads (+ jump)
    constexpr uint32_t head_words = 2 + 3 * G;  // dest core, inc value, (slot base, row stride, jump) per destination

    size_t a = 0;
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t my_dir = get_arg_val<uint32_t>(a++);
    const uint32_t landing_addr = get_arg_val<uint32_t>(a++);
    const uint32_t arrive_p_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_dest = get_arg_val<uint32_t>(a++);
    const uint32_t dests_idx = a;
    a += 2 * n_dest;
    uint32_t dest_of_chip[G];
    for (uint32_t i = 0; i < G; ++i) {
        dest_of_chip[i] = get_arg_val<uint32_t>(a++);
    }
    uint32_t dirmap[G];
    for (uint32_t i = 0; i < G; ++i) {
        dirmap[i] = get_arg_val<uint32_t>(a++);
    }
    uint32_t rorder[G];
    for (uint32_t i = 0; i + 1 < G; ++i) {
        rorder[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t n_heads = get_arg_val<uint32_t>(a++);
    const uint32_t heads_idx = a;
    a += n_heads * head_words;
    const uint32_t n_ready_out = get_arg_val<uint32_t>(a++);
    const uint32_t ready_idx = a;
    a += 3 * n_ready_out;
    const uint32_t n_ready_wait = get_arg_val<uint32_t>(a++);
    const bool has_conn = get_arg_val<uint32_t>(a++) != 0;
    if (!has_conn) {
        return;
    }

    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
    ASSERT(n_dest * hdr_per_dest <= max_headers);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[max_headers];
    for (uint32_t k = 0; k < n_dest; ++k) {
        const uint16_t chip = static_cast<uint16_t>(get_arg_val<uint32_t>(dests_idx + 2 * k));
        const uint16_t mesh = static_cast<uint16_t>(get_arg_val<uint32_t>(dests_idx + 2 * k + 1));
        for (uint32_t r = 0; r < hdr_per_dest; ++r) {
            hdrs[k * hdr_per_dest + r] = PacketHeaderPool::allocate_header();
            route_to(hdrs[k * hdr_per_dest + r], chip, mesh);
        }
    }
    conn.open();

    // A header is rewritten (command fields only; the route stays) only after the NoC has read its last packet.
    uint32_t dirty = 0;
    uint32_t rot[G];
    for (uint32_t i = 0; i < G; ++i) {
        rot[i] = 0;
    }
    auto next_header = [&](uint32_t k) {
        const uint32_t idx = k * hdr_per_dest + rot[k];
        rot[k] = rot[k] + 1 == hdr_per_dest ? 0 : rot[k] + 1;
        if (dirty & (1u << idx)) {
            noc_async_writes_flushed();
            dirty = 0;
        }
        dirty |= 1u << idx;
        return hdrs[idx];
    };

    // ---- ready fence ----------------------------------------------------------------------------------------------
    {
    MaybeDeviceZoneScope("PORT_FENCE");
    for (uint32_t k = 0; k < n_ready_out; ++k) {
        const uint32_t t = ready_idx + 3 * k;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = next_header(get_arg_val<uint32_t>(t));
        hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
            get_noc_addr(get_arg_val<uint32_t>(t + 1), get_arg_val<uint32_t>(t + 2), ready_addr, 0), 1, true});
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

    // ---- segments -------------------------------------------------------------------------------------------------
    uint32_t total = 0;
    if (n_heads > 0) {
        q_stem::for_each_block(B, qrt, m, G, p, rorder, [&](uint32_t, uint32_t, uint32_t d, uint32_t) {
            if (d != p && dirmap[d] == my_dir) {
                total += m * n_heads;
            }
        });
    }
    uint32_t sent = 0, unflushed = 0;
    MaybeDeviceZoneScope("PORT_SEND");  // segments: waits on the port reader's pulls + fabric sends
    if (total > 0) {
        q_stem::for_each_block(B, qrt, m, G, p, rorder, [&](uint32_t, uint32_t b, uint32_t d, uint32_t r0) {
            if (d == p || dirmap[d] != my_dir) {
                return;
            }
            const uint32_t k = dest_of_chip[d];
            for (uint32_t i = 0; i < m; ++i) {
                const uint32_t row = b * qrt + r0 + i;
                for (uint32_t e = 0; e < n_heads; ++e) {
                    const uint32_t t = heads_idx + e * head_words;
                    const uint32_t dc = get_arg_val<uint32_t>(t);
                    const uint32_t inc = get_arg_val<uint32_t>(t + 1);
                    const uint32_t slot = get_arg_val<uint32_t>(t + 2 + 3 * d) +
                                          row * get_arg_val<uint32_t>(t + 3 + 3 * d) +
                                          (row >= ts ? get_arg_val<uint32_t>(t + 4 + 3 * d) : 0);
                    const uint32_t dx = q_stem::unpack_x(dc), dy = q_stem::unpack_y(dc);
                    const uint64_t arrive_noc = get_noc_addr(dx, dy, arrive_p_addr, 0);
                    cb_wait_front(cb_port_stage, hw * (unflushed + 1));
                    const uint32_t src = get_read_ptr(cb_port_stage) + unflushed * seg_bytes;
                    for (uint32_t off = 0; off < hw; off += pkt_tiles) {
                        const uint32_t n = (hw - off) < pkt_tiles ? hw - off : pkt_tiles;
                        const uint64_t dst = get_noc_addr(dx, dy, landing_addr + (slot + off) * tile_bytes, 0);
                        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = next_header(k);
#ifdef ABLATE_SEND_PAYLOAD
                        if (off + n == hw && inc > 0) {
                            hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{arrive_noc, inc, true});
                            conn.wait_for_empty_write_slot();
                            conn.send_payload_flush_non_blocking_from_address(
                                reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
                        }
                        continue;
#endif
                        if (off + n == hw && inc > 0) {
                            hdr->to_noc_fused_unicast_write_atomic_inc(
                                NocUnicastAtomicIncFusedCommandHeader{dst, arrive_noc, inc, true}, n * tile_bytes);
                        } else {
                            hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, n * tile_bytes);
                        }
                        conn.wait_for_empty_write_slot();
                        conn.send_current_slot_non_blocking(
                            src + off * tile_bytes, n * tile_bytes, reinterpret_cast<uint32_t>(hdr));
                    }
                    ++sent;
                    if (++unflushed == batch || sent == total) {
                        noc_async_writes_flushed();  // payloads + headers of the in-flight segments have left L1
                        dirty = 0;
                        cb_pop_front(cb_port_stage, hw * unflushed);
                        unflushed = 0;
                    }
                }
            }
        });
    }
    noc_async_writes_flushed();
    conn.close();
}
