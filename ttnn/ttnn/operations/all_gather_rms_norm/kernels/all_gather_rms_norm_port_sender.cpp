// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm port sender (BRISC / NoC1), one per (direction, link); the only producer on its router
// channel. Per call:
//   fabric_fence (TRANSPORT)      a header-only ready increment to the peer port that writes into this chip; before
//                                 its first send, wait for its own peer's ready (one sent / one consumed per call)
//   exchange_stats (TRANSPORT, link-0 ports = hubs)
//                                 once every active worker wrote its records into my table (records_local), send
//                                 the own stats slot, then relay each upstream slot once it landed (one fused
//                                 increment of the peer hub's stats_arrived per slot, on its last packet)
//   announce_stats_ready (OVERLAP, hubs)
//                                 once my table holds every slot (own + all expected arrivals): one increment of
//                                 every active worker's stats_ready; re-arm stats_arrived
//   gather_normalized_slices (TRANSPORT)
//                                 the chunks the port reader stages, each one packet into the same pages of the
//                                 downstream chip's destination (final output pages, or R_U staging): the routers
//                                 write them, no receive-side kernel. Every inc_every-th chunk and each entry's
//                                 last carry a fused increment of the downstream port's arrival counter.
// fabric_all_gather's sender + fabric_ring_exchange's stats round, ported.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "agrn_port_common.hpp"

using namespace tt::tt_fabric;
using namespace agrn_port;

namespace {
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
}  // namespace

void kernel_main() {
    constexpr auto dst_args = TensorAccessorArgs<PORT_CT_ARGS>();
    constexpr uint32_t num_headers = group;
    static_assert(num_headers >= 1 && num_headers <= 12, "packet header pool: <= 12 headers per RISC");

    size_t a = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);  // downstream port's counter: same address
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++);
    const uint32_t ready_xy = get_arg_val<uint32_t>(a++);
    const uint32_t peer_xy = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const bool has_conn = get_arg_val<uint32_t>(a++) != 0;
    const bool is_hub = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t table = get_arg_val<uint32_t>(a++);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t stats_arrived_addr = get_arg_val<uint32_t>(a++);
    const uint32_t stats_expect_in = get_arg_val<uint32_t>(a++);
    const uint32_t n_active = get_arg_val<uint32_t>(a++);
    const uint32_t n_stats = get_arg_val<uint32_t>(a++);
    const uint32_t stats_idx = a;
    a += n_stats;
    const uint32_t n_entries = get_arg_val<uint32_t>(a++);
    const uint32_t entries_idx = a;  // per entry: slot | part << 16, chunks
    a += 2 * n_entries;
    const uint32_t n_notify = is_hub ? n_active : 0;
    const uint32_t workers_idx = a;
    a += n_notify;

    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* stats_arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_arrived_addr);

    if (has_conn) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
        }
        conn.open();
        // ---- fabric_fence ----
        if (send_ready) {
            hdrs[0]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_xy & 0xFFFF, ready_xy >> 16, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (n_entries > 0 || n_stats > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            AGRN_ZONE("ps_fence_wait");
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
        // ---- exchange_stats ----
        if (is_hub && n_stats > 0) {
            volatile tt_l1_ptr uint32_t* records =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_records_local));
            {
                AGRN_ZONE("ps_rec_wait");
                noc_semaphore_wait_min(records, n_active);
            }
            AGRN_ZONE("ps_stats_send");
            const uint64_t peer_stats_arrived = get_noc_addr(peer_xy & 0xFFFF, peer_xy >> 16, stats_arrived_addr);
            for (uint32_t k = 0; k < n_stats; ++k) {
                if (k > 0) {
                    noc_semaphore_wait_min(stats_arrived, k);  // upstream's slot k - 1 landed in my table
                }
                const uint32_t q = get_arg_val<uint32_t>(stats_idx + k);
                const uint32_t src = table + q * slot_bytes;
                for (uint32_t off = 0; off < slot_bytes; off += payload_bytes) {
                    const uint32_t n = (slot_bytes - off) < payload_bytes ? slot_bytes - off : payload_bytes;
                    const uint64_t d = get_noc_addr(peer_xy & 0xFFFF, peer_xy >> 16, src + off);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[0];
                    if (off + n == slot_bytes) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{d, peer_stats_arrived, 1, true}, n);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{d}, n);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(src + off, n, reinterpret_cast<uint32_t>(hdr));
                    noc_async_writes_flushed();  // header 0 is rewritten by the next packet
                }
            }
        }
        // ---- announce_stats_ready ----
        if (is_hub) {
            {
                AGRN_ZONE("ps_stats_wait");
                noc_semaphore_wait_min(stats_arrived, stats_expect_in);
            }
            const uint32_t sem = get_semaphore(agrn::sem_stats_ready);
            for (uint32_t i = 0; i < n_notify; ++i) {
                const uint32_t xy = get_arg_val<uint32_t>(workers_idx + i);
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, sem), 1);
            }
            if (stats_expect_in > 0) {
                noc_semaphore_inc(get_noc_addr(stats_arrived_addr), 0u - stats_expect_in);
            }
            noc_async_atomic_barrier();
        }
        // ---- gather_normalized_slices ----
        const uint64_t arrival_noc = get_noc_addr(peer_xy & 0xFFFF, peer_xy >> 16, arrival_addr);
        uint32_t sent = 0;
        uint32_t unflushed = 0;
        uint32_t chunks_left = 0;
        for (uint32_t k = 0; k < n_entries; ++k) {
            chunks_left += get_arg_val<uint32_t>(entries_idx + 2 * k + 1);
        }
        // A header is rewritten (command fields only; the route stays) once the NoC has read its previous packet.
        uint32_t h_next = 0, dirty = 0;
        auto next_header = [&]() {
            const uint32_t hk = h_next;
            h_next = (h_next + 1 == num_headers) ? 0 : h_next + 1;
            if (dirty & (1u << hk)) {
                noc_async_writes_flushed();
                dirty = 0;
            }
            dirty |= 1u << hk;
            return hdrs[hk];
        };
        AGRN_ZONE("ps_bulk");
        for (uint32_t k = 0; k < n_entries; ++k) {
            const uint32_t e = get_arg_val<uint32_t>(entries_idx + 2 * k);
            const uint32_t entry_total = get_arg_val<uint32_t>(entries_idx + 2 * k + 1);
            uint32_t entry_sent = 0;
            for_each_packet(e & 0xFFFF, e >> 16, first_bank, bank_stride, [&](const Packet& pk) {
                cb_wait_front(cb_chunks, run_pages * (unflushed + 1));
                const uint32_t src = get_read_ptr(cb_chunks) + unflushed * chunk_bytes;
                const uint32_t bytes = pk.pages * page_bytes;
                uint64_t d[4];
                uint16_t sz[4];
                for (uint32_t ri = 0; ri < pk.nr; ++ri) {
                    d[ri] = dst.get_noc_addr(pk.first[ri], 0, 0);  // NoC0 coordinates
                    sz[ri] = static_cast<uint16_t>(pk.n[ri] * page_bytes);
                }
                ++entry_sent;
                // The increment lands only after the packet's writes (flush), stalling the receiving router, so only
                // every inc_every-th packet and each entry's last carry one (fabric_all_gather).
                const bool inc = (++sent % inc_every) == 0 || entry_sent == entry_total;
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = next_header();
                bool extra_inc = false;
                if (pk.nr == 1) {
                    if (inc) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{d[0], arrival_noc, 1, true}, bytes);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{d[0]}, bytes);
                    }
                } else if (pk.nr == 2 && inc) {
                    hdr->to_noc_fused_unicast_scatter_write_atomic_inc(
                        NocUnicastScatterAtomicIncFusedCommandHeader({d[0], d[1]}, arrival_noc, {sz[0]}, 1, true),
                        bytes);
                } else {
                    hdr->to_noc_unicast_scatter_write(
                        NocUnicastScatterCommandHeader(d, sz, static_cast<uint8_t>(pk.nr)), bytes);
                    extra_inc = inc;
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                if (extra_inc) {
                    // > 2 runs: a header-only increment after the scatter write (flush: lands after the writes)
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* ih = next_header();
                    ih->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{arrival_noc, 1, true});
                    conn.wait_for_empty_write_slot();
                    conn.send_payload_flush_blocking_from_address(
                        reinterpret_cast<uint32_t>(ih), sizeof(PACKET_HEADER_TYPE));
                }
                --chunks_left;
                // pops stay on group boundaries of the 2-group CB (the read offset never wraps), plus the last
                if (++unflushed == group || chunks_left == 0) {
                    noc_async_writes_flushed();  // sources + headers of the in-flight packets have left L1
                    dirty = 0;
                    cb_pop_front(cb_chunks, run_pages * unflushed);
                    unflushed = 0;
                }
            });
        }
        noc_async_writes_flushed();
        conn.close();
    } else if (is_hub) {
        // receive-only hub (an end of a line): still announces its table complete
        noc_semaphore_wait_min(stats_arrived, stats_expect_in);
        const uint32_t sem = get_semaphore(agrn::sem_stats_ready);
        for (uint32_t i = 0; i < n_notify; ++i) {
            const uint32_t xy = get_arg_val<uint32_t>(workers_idx + i);
            noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, sem), 1);
        }
        if (stats_expect_in > 0) {
            noc_semaphore_inc(get_noc_addr(stats_arrived_addr), 0u - stats_expect_in);
        }
        noc_async_atomic_barrier();
    }
}
