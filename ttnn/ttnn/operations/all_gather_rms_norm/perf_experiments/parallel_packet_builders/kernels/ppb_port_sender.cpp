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
//                                 last carry an increment of the downstream port's arrival counter. The bulk is
//                                 bound by per-packet work on this core: the runs come from the reader (cb_desc, no
//                                 second walk) and an interleaved destination's command fields are written directly
//                                 from per-bank router-side addresses (no per-chunk address re-encoding).
// fabric_all_gather's sender + fabric_ring_exchange's stats round, ported.

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "agrn_port_common.hpp"
#include "ppb_common.hpp"
#if defined(PPB_DEBUG)
#include "api/debug/dprint.h"
#endif

using namespace tt::tt_fabric;
using namespace agrn_port;

// Interleaved destination with few banks (DRAM): per-bank precomputed router-side addresses (see kernel_main).
constexpr bool fast_banks = shard_w == 0 && nb <= 16;

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
    const uint32_t n_entries_args = get_arg_val<uint32_t>(a++);
#if defined(AGRN_ABL_NO_BULK)
    const uint32_t n_entries = 0;  // ablation: bulk stubbed (output garbage; perf ablation only)
#else
    const uint32_t n_entries = n_entries_args;
#endif
    const uint32_t entries_idx = a;  // per entry: slot | part << 16, chunks
    a += 2 * n_entries_args;
    // parallel_packet_builders: H helpers build the bulk packets into my ring (cb_chunks), D ring slots per helper
    const uint32_t ppb_H = get_arg_val<uint32_t>(a++);
    const uint32_t ppb_D = get_arg_val<uint32_t>(a++);
    const uint32_t ppb_pitch = get_arg_val<uint32_t>(a++);
    const uint32_t ppb_helpers_idx = a;  // H helper NoC coords (x | y << 16)
    a += ppb_H;

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
        if (n_entries_args > 0 || n_stats > 0) {
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
            // (the reader multicasts the table, then re-arms stats_arrived)
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_stats_done)) = 1;
        }
        // ---- gather_normalized_slices (forward only: the helpers built every packet) ----
        const uint64_t arrival_noc = get_noc_addr(peer_xy & 0xFFFF, peer_xy >> 16, arrival_addr);
        uint32_t n_tot = 0;
        for (uint32_t k = 0; k < n_entries; ++k) {
            n_tot += get_arg_val<uint32_t>(entries_idx + 2 * k + 1);
        }
        const uint32_t n_own = n_entries > 0 ? get_arg_val<uint32_t>(entries_idx + 1) : 0;
        (void)n_own;
        AGRN_ZONE("ps_bulk");
        const uint32_t ring = get_write_ptr(cb_chunks);
        const uint32_t credit_sem = get_semaphore(ppb::SEM_CREDIT);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* ih = hdrs[0];  // header-only increments (> 2-run packets with inc)
        uint32_t h = 0, j = 0;
#if defined(PPB_DEBUG)
        uint32_t t_snd[2] = {0, 0}, t_fl[2] = {0, 0}, t_rdy[2] = {0, 0}, t_slot[2] = {0, 0}, t_all[2] = {0, 0},
                 t_first = 0, t_relay0 = 0;
        uint32_t t_prev = get_timestamp_32b();
#endif
        for (uint32_t i = 0; i < n_tot; ++i) {
#if defined(PPB_DEBUG)
            const uint32_t ph = i < n_own ? 0 : 1;
            const uint32_t t0 = get_timestamp_32b();
            if (i == n_own) {
                t_relay0 = t0;
            }
#endif
            volatile tt_l1_ptr uint32_t* rdy =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ppb::SEM_READY0 + h));
#if defined(PPB_DEBUG)
            uint32_t spins = 0;
#endif
            do {
                invalidate_l1_cache();
#if defined(PPB_DEBUG)
                if (++spins == 20000000) {
                    auto rv = [&](uint32_t hh) {
                        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ppb::SEM_READY0 + hh));
                    };
                    DPRINT(
                        "PPB stall i {} h {} j {} n_tot {} rdy {} {} {} {} addr {}\n",
                        i,
                        h,
                        j,
                        n_tot,
                        rv(0),
                        rv(1),
                        rv(2),
                        rv(3),
                        get_semaphore(ppb::SEM_READY0));
                }
#endif
            } while (*rdy < j + 1);
#if defined(PPB_DEBUG)
            const uint32_t t1 = get_timestamp_32b();
            t_rdy[ph] += t1 - t0;
            if (i == 0) {
                t_first = t1;
            }
#endif
#if defined(AGRN_PROFILE_ZONES)
            if (i == 0) {
                AGRN_ZONE("ps_own_pkt0");
            }
            if (i == n_own) {
                AGRN_ZONE("ps_relay_start");
            }
#endif
            const uint32_t slot = ring + ((j % ppb_D) * ppb_H + h) * ppb_pitch;
            volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
            const uint32_t bytes = meta[0];
            const bool extra = meta[1] != 0;
            conn.wait_for_empty_write_slot();
#if defined(PPB_DEBUG)
            t_slot[ph] += get_timestamp_32b() - t1;
#endif
#if defined(PPB_DEBUG)
            const uint32_t t2 = get_timestamp_32b();
#endif
            conn.send_payload_non_blocking_from_address(slot + ppb::META_BYTES, sizeof(PACKET_HEADER_TYPE) + bytes);
#if defined(PPB_DEBUG)
            t_snd[ph] += get_timestamp_32b() - t2;
#endif
            if (extra) {
                ih->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{arrival_noc, 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(ih), sizeof(PACKET_HEADER_TYPE));
            }
            if (++h == ppb_H || i + 1 == n_tot) {
                // the ring slots of this round have left L1: free them (credit = my packets sent, only values the
                // helper will wait for: <= its count - D)
#if defined(PPB_DEBUG)
                const uint32_t tf = get_timestamp_32b();
#endif
                noc_async_writes_flushed();
#if defined(PPB_DEBUG)
                t_fl[ph] += get_timestamp_32b() - tf;
#endif
                for (uint32_t hh = 0; hh < h; ++hh) {
                    const uint32_t n_h = (n_tot - hh + ppb_H - 1) / ppb_H;
                    if (j + 1 + ppb_D <= n_h) {
                        const uint32_t xy = get_arg_val<uint32_t>(ppb_helpers_idx + hh);
                        noc_inline_dw_write(get_noc_addr(xy & 0xFFFF, xy >> 16, credit_sem), j + 1);
                    }
                }
                h = 0;
                ++j;
            }
#if defined(PPB_DEBUG)
            const uint32_t t3 = get_timestamp_32b();
            t_all[ph] += t3 - t0;
            t_prev = t3;
#endif
        }
#if defined(PPB_DEBUG)
        DPRINT(
            "PPB timing n_own {} n_tot {} own: all {} rdy {} slot {} | relay: all {} rdy {} slot {} | snd {} {} fl {} "
            "{} nbuf {} bufsz {} | own span {} end {}\n",
            n_own,
            n_tot,
            t_all[0],
            t_rdy[0],
            t_slot[0],
            t_all[1],
            t_rdy[1],
            t_slot[1],
            t_snd[0],
            t_snd[1],
            t_fl[0],
            t_fl[1],
            (uint32_t)conn.num_buffers_per_channel,
            (uint32_t)conn.buffer_size_bytes,
            t_relay0 - t_first,
            t_prev - t_first);
#endif
        noc_async_write_barrier();
        conn.close();
    } else if (is_hub) {
        // receive-only hub (an end of a line): its table is complete once the own records and every expected slot
        // landed; its reader announces it
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_records_local)), n_active);
        noc_semaphore_wait_min(stats_arrived, stats_expect_in);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(agrn::sem_stats_done)) = 1;
    }
}
