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

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "agrn_port_common.hpp"
#if defined(AGRN_ABL_TIMING)
#include "api/debug/device_print.h"
#define TT_(...) __VA_ARGS__
#else
#define TT_(...)
#endif

using namespace tt::tt_fabric;
using namespace agrn_port;

// Interleaved destination with few banks (DRAM): per-bank precomputed router-side addresses (see kernel_main).
constexpr bool fast_banks = shard_w == 0 && nb <= 16;
#if defined(AGRN_ABL_DST)
constexpr bool abl_dst = true;
#else
constexpr bool abl_dst = false;
#endif

constexpr uint8_t scatter_enc =
    expand_encoding_to_all_chunks(NocScatterWriteChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE);
static_assert(
    sizeof(NocUnicastScatterCommandHeader) == 40 && offsetof(NocUnicastScatterCommandHeader, chunk_size) == 32 &&
        offsetof(NocUnicastScatterCommandHeader, chunk_count) == 38 &&
        offsetof(NocUnicastScatterCommandHeader, chunk_encoding) == 39,
    "scatter command field layout (word stores below)");

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

    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    const uint32_t table = get_write_ptr(agrn::cb_stats_table);  // same address on every core of every chip
    volatile tt_l1_ptr uint32_t* stats_arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_arrived_addr);

    if (has_conn) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
        }
        // Router-side (edm-local) NoC address of every destination bank's first page, and the bank page pitch: a
        // bulk packet header then takes its run addresses as base + offset, with none of the per-chunk address
        // re-encoding of the header builders (the bulk is BRISC-bound on per-packet work, Refinement 2).
        uint64_t bank_base[fast_banks ? nb : 1];
        uint32_t bank_pitch = 0;
        if constexpr (fast_banks) {
            volatile tt_l1_ptr PACKET_HEADER_TYPE* sh = hdrs[0];
            for (uint32_t b = 0; b < nb; ++b) {
                sh->to_noc_unicast_write(NocUnicastCommandHeader{dst.get_noc_addr(b, 0, 0)}, 0);
                bank_base[b] = sh->command_fields.unicast_write.noc_address;
            }
            bank_pitch = static_cast<uint32_t>(dst.get_noc_addr(nb, 0, 0) - dst.get_noc_addr(0, 0, 0));
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
            // the hub's reader multicasts stats-ready over the program grid (NoC0, see the port reader)
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_stats_done)) = 1;
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
        constexpr uint32_t n_slots = 2 * group;
        const uint32_t desc_base = get_read_ptr(cb_desc);
        uint32_t taken = 0;  // packets taken from the port CB (= the reader's issue index)
        TT_(uint32_t t_cb[2] = {0, 0}; uint32_t t_slot[2] = {0, 0}; uint32_t t_fl[2] = {0, 0};
            uint32_t n_pk[2] = {0, 0};
            uint32_t t_e[2] = {0, 0};
            uint32_t t_addr[2] = {0, 0};
            uint32_t t_hdr[2] = {0, 0};
            uint32_t t_send[2] = {0, 0};)
        for (uint32_t k = 0; k < n_entries; ++k) {
            TT_(uint32_t te = get_timestamp_32b();)
            const uint32_t e = get_arg_val<uint32_t>(entries_idx + 2 * k);
            const uint32_t entry_total = get_arg_val<uint32_t>(entries_idx + 2 * k + 1);
            uint32_t entry_sent = 0;
            if (k == 1) {
                AGRN_ZONE("ps_relay_start");  // marker
            }
#if defined(AGRN_ABL_MINSEND)
            if (k == 0) {  // perf ablation only: entry_total full-size packets, minimal CPU, into peer L1, no reader
                const uint32_t base = get_read_ptr(cb_chunks);
                for (uint32_t i = 0; i < entry_total; ++i) {
                    const bool inc = (++sent % inc_every) == 0 || i + 1 == entry_total;
                    const uint64_t d0 =
                        get_noc_addr(peer_xy & 0xFFFF, peer_xy >> 16, base + (i % (2 * group)) * chunk_bytes);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = next_header();
                    if (inc) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{d0, arrival_noc, 1, true}, chunk_bytes);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{d0}, chunk_bytes);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(base, chunk_bytes, reinterpret_cast<uint32_t>(hdr));
                    if ((i + 1) % group == 0) {
                        noc_async_writes_flushed();
                        dirty = 0;
                    }
                }
                chunks_left -= entry_total;
                TT_(n_pk[0] = entry_total; t_e[0] += get_timestamp_32b() - te;)
                continue;
            }
#endif
            (void)e;
            for (uint32_t pi = 0; pi < entry_total; ++pi) {
                TT_(uint32_t t0 = get_timestamp_32b();)
                cb_wait_front(cb_chunks, run_pages * (unflushed + 1));
                TT_(t_cb[k > 0] += get_timestamp_32b() - t0;)
                invalidate_l1_cache();  // the slot's descriptor was written by the reader (NCRISC)
                Packet pk;
                {
                    volatile tt_l1_ptr PacketDesc* dsc =
                        reinterpret_cast<volatile tt_l1_ptr PacketDesc*>(desc_base + (taken % n_slots) * DESC_BYTES);
                    pk.nr = dsc->nr;
                    pk.pages = dsc->pages;
                    for (uint32_t ri = 0; ri < pk.nr; ++ri) {
                        pk.first[ri] = dsc->first[ri];
                        pk.n[ri] = dsc->n[ri];
                    }
                }
                ++taken;
                const uint32_t src = get_read_ptr(cb_chunks) + unflushed * chunk_bytes;
                const uint32_t bytes = pk.pages * page_bytes;
                uint64_t d[4];
                uint16_t sz[4];
                uint32_t pk_nr_override = 0;
                TT_(uint32_t ta = get_timestamp_32b();)
                const bool inc = (sent + 1) % inc_every == 0 || entry_sent + 1 == entry_total;
                for (uint32_t ri = 0; ri < pk.nr; ++ri) {
                    if (!fast_banks || abl_dst || inc) {
                        d[ri] = dst.get_noc_addr(pk.first[ri], 0, 0);  // NoC0 coordinates (header builders)
                    }
                    sz[ri] = static_cast<uint16_t>(pk.n[ri] * page_bytes);
                }
                TT_(t_addr[k > 0] += get_timestamp_32b() - ta; ta = get_timestamp_32b();)
#if defined(AGRN_ABL_DST)
                {  // perf ablation only: land the packet in the peer port's L1 (data is garbage)
                    uint32_t la = get_read_ptr(cb_chunks) + (sent % (2 * group)) * chunk_bytes;
                    for (uint32_t ri = 0; ri < pk.nr; ++ri) {
                        d[ri] = get_noc_addr(peer_xy & 0xFFFF, peer_xy >> 16, la);
                        la += sz[ri];
                    }
                }
#if AGRN_ABL_DST == 1
                pk_nr_override = 1;
#endif
#endif
                ++entry_sent;
                (void)pk_nr_override;
                // The increment lands only after the packet's writes (flush), stalling the receiving router, so only
                // every inc_every-th packet and each entry's last carry one (fabric_all_gather).
                ++sent;
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = next_header();
                bool extra_inc = false;
                const uint32_t pnr = pk_nr_override ? pk_nr_override : pk.nr;
                if (fast_banks && !abl_dst && !inc) {
                    // the header builders' fields, written directly from the precomputed bank addresses
                    if (pnr == 1) {
                        hdr->noc_send_type = NOC_UNICAST_WRITE;
                        hdr->command_fields.unicast_write.noc_address =
                            bank_base[pk.first[0] % nb] + (pk.first[0] / nb) * bank_pitch;
                    } else {
                        // the scatter command fields as ten 32-bit stores (layout pinned by the static_asserts above)
                        hdr->noc_send_type = NOC_UNICAST_SCATTER_WRITE;
                        volatile tt_l1_ptr uint32_t* w =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&hdr->command_fields.unicast_scatter_write);
                        for (uint32_t ri = 0; ri < NOC_SCATTER_WRITE_MAX_CHUNKS; ++ri) {
                            const uint64_t a =
                                ri < pnr ? bank_base[pk.first[ri] % nb] + (pk.first[ri] / nb) * bank_pitch : 0;
                            w[2 * ri] = static_cast<uint32_t>(a);
                            w[2 * ri + 1] = static_cast<uint32_t>(a >> 32);
                        }
                        const uint32_t s1 = pnr > 2 ? sz[1] : 0, s2 = pnr > 3 ? sz[2] : 0;
                        w[8] = static_cast<uint32_t>(sz[0]) | (s1 << 16);
                        w[9] = s2 | (pnr << 16) | (static_cast<uint32_t>(scatter_enc) << 24);
                    }
                    hdr->payload_size_bytes = static_cast<uint16_t>(bytes);
                } else if (pnr == 1) {
                    if (inc) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{d[0], arrival_noc, 1, true}, bytes);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{d[0]}, bytes);
                    }
                } else if (pnr == 2 && inc) {
                    hdr->to_noc_fused_unicast_scatter_write_atomic_inc(
                        NocUnicastScatterAtomicIncFusedCommandHeader({d[0], d[1]}, arrival_noc, {sz[0]}, 1, true),
                        bytes);
                } else {
                    hdr->to_noc_unicast_scatter_write(
                        NocUnicastScatterCommandHeader(d, sz, static_cast<uint8_t>(pnr)), bytes);
                    extra_inc = inc;
                }
                TT_(uint32_t t1 = get_timestamp_32b(); t_hdr[k > 0] += t1 - ta;)
                conn.wait_for_empty_write_slot();
                TT_(t_slot[k > 0] += get_timestamp_32b() - t1; t1 = get_timestamp_32b();)
                conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                TT_(t_send[k > 0] += get_timestamp_32b() - t1;)
                TT_(++n_pk[k > 0];)
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
                    TT_(uint32_t t2 = get_timestamp_32b();)
                    noc_async_writes_flushed();
                    TT_(t_fl[k > 0] +=
                        get_timestamp_32b() - t2;)  // sources + headers of the in-flight packets have left L1
                    dirty = 0;
                    cb_pop_front(cb_chunks, run_pages * unflushed);
                    unflushed = 0;
                }
            }
            TT_(t_e[k > 0] += get_timestamp_32b() - te;)
        }
        TT_(DEVICE_PRINT(
                "AGRN_T own n={} tot={} cb={} slot={} fl={} | relay n={} tot={} cb={} slot={} fl={} addr={} hdr={} "
                "send={}\n",
                n_pk[0],
                t_e[0],
                t_cb[0],
                t_slot[0],
                t_fl[0],
                n_pk[1],
                t_e[1],
                t_cb[1],
                t_slot[1],
                t_fl[1],
                t_addr[0],
                t_hdr[0],
                t_send[0]);)
        noc_async_writes_flushed();
        conn.close();
    } else if (is_hub) {
        // receive-only hub (an end of a line): its table is complete once the own records and every expected slot
        // landed; its reader announces it
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_records_local)), n_active);
        noc_semaphore_wait_min(stats_arrived, stats_expect_in);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_stats_done)) = 1;
        if (stats_expect_in > 0) {
            noc_semaphore_inc(get_noc_addr(stats_arrived_addr), 0u - stats_expect_in);
        }
        noc_async_atomic_barrier();
    }
}
