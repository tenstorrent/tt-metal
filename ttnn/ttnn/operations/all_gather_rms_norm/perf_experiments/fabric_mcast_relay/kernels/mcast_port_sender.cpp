// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// fabric_mcast_relay bench — CANDIDATE port sender (BRISC / NoC1): the op's port sender's bulk path with the
// store-and-forward relay replaced by 2D chip-level fabric multicast.
//
// Raw fabric API (no helper exists for this): the route of each packet header is set with tt_fabric_api.h
// fabric_set_mcast_route (the route setter mesh/api.h fabric_multicast_noc_* calls), and the command fields are
// written exactly as the op's sender does (direct field stores from per-bank router addresses). mesh/api.h's
// fabric_multicast_noc_* wrappers re-set the route and re-encode the command on every packet, which is the per-packet
// port-CPU work the op's Refinement 2 removed; here the route is set once per header per range change.
//
// Per port (direction d, link l), one entry: the own lane (part 0), read by the op's UNCHANGED port reader, sent ONCE
// in up to two segments split at packet `seg_split` (the walk's half-lane bank boundary: part 1 prefix, part 2 suffix),
// each with a host-computed MeshMcastRange spec (spec_a / spec_b). A spec writes the ring's next 1 or 2 chips along d:
// a straight range when both hops go the same physical direction (a torus axis), or an L = one N/S spine hop + one E/W
// branch hop. An E/W-then-N/S path is NOT expressible (the 2D router only turns from the N/S spine to E/W branches);
// on a QuietBox-2 "1x4 ring" (physically a 2x2 square, every hop turns) each chip has exactly one expressible
// direction, so that port sends its whole lane to 2 chips and the other port to 1 (busiest link 2 lanes, not 1.5).
// The routers write every chip of a range (same NoC address, same increment); no port relays, no DRAM read-back.
//
// Arrival increments: one fused increment on the LAST packet of each non-empty segment, landing on the lane's counter
// core (the same core on every chip: chip 0's (d, l) port core), polled over the NoC by this sender at the end.
//
// Fence: before its first packet the port waits one ready from EVERY chip it writes into (the far chip must have
// started this call before a router writes into it). Readies are unicasts routed by the fabric table (a 2-hop one for
// the far chip), sent by whichever port of the far chip sits on the table's first hop.

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "agrn_port_common.hpp"
#if defined(FMR_DPRINT)
#include "api/debug/dprint.h"
#endif

using namespace tt::tt_fabric;
using namespace agrn_port;

#if !((defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D))
#error "chip-level mesh multicast needs a 2D fabric (1D: linear/api.h start_distance / range instead)"
#endif

constexpr bool fast_banks = shard_w == 0 && nb <= 16;

constexpr uint8_t scatter_enc =
    expand_encoding_to_all_chunks(NocScatterWriteChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE);
static_assert(
    sizeof(NocUnicastScatterCommandHeader) == 40 && offsetof(NocUnicastScatterCommandHeader, chunk_size) == 32 &&
        offsetof(NocUnicastScatterCommandHeader, chunk_count) == 38 &&
        offsetof(NocUnicastScatterCommandHeader, chunk_encoding) == 39,
    "scatter command field layout (word stores below)");

namespace {
// spec = e | w << 8 | n << 16 | s << 24 (MeshMcastRange hop counts from the sender; host-computed from the physical
// directions of the ring's links: a straight range along one direction, or an L = one N/S spine hop + one E/W branch
// hop — the only turn a 2D multicast route can take)
inline void set_spec_route(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip, uint16_t mesh, uint32_t spec) {
    fabric_set_mcast_route(hdr, chip, mesh, spec & 0xFF, (spec >> 8) & 0xFF, (spec >> 16) & 0xFF, (spec >> 24) & 0xFF);
}
}  // namespace

void kernel_main() {
    constexpr auto dst_args = TensorAccessorArgs<PORT_CT_ARGS>();
    constexpr uint32_t num_headers = group;
    static_assert(num_headers >= 1 && num_headers <= 11, "packet header pool: <= 12 headers per RISC (+1 ready)");

    size_t a = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);  // downstream ports' counter: same address, same core
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ready_expect = get_arg_val<uint32_t>(a++);  // readies I wait for (one per chip I write into)
    // counter core of this lane (d, l): the SAME core on every chip (a multicast increment lands at one NoC address on
    // every chip of its range), holding the arrival counter at arrival_addr
    const uint32_t peer_xy = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const bool has_conn = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t n_pkts = get_arg_val<uint32_t>(a++);
    const uint32_t seg_split = get_arg_val<uint32_t>(a++);
    const uint32_t spec_a = get_arg_val<uint32_t>(a++);
    const uint32_t spec_b = get_arg_val<uint32_t>(a++);
    // readies this port sends (unicast, routed by the fabric's table; the host puts each on the port whose link is
    // the table's first hop): per ready the destination chip and the port core it lands on
    const uint32_t n_readies = get_arg_val<uint32_t>(a++);
    const uint32_t readies_idx = a;
    a += 2 * n_readies;
    const uint32_t wait_expect = get_arg_val<uint32_t>(a++);  // arrivals into this chip's lane-(d, l) counter

    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    if (!has_conn) {
        return;
    }
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
    uint32_t hdr_spec[num_headers];
    for (uint32_t h = 0; h < num_headers; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        hdr_spec[h] = 0;
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* rh = PacketHeaderPool::allocate_header();  // readies
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
    // ---- fabric_fence: a ready to every port writing into this chip, then one from every chip I write into ----
    for (uint32_t i = 0; i < n_readies; ++i) {
        const uint32_t chip = get_arg_val<uint32_t>(readies_idx + 2 * i);
        const uint32_t xy = get_arg_val<uint32_t>(readies_idx + 2 * i + 1);
        (void)fabric_set_unicast_route(rh, static_cast<uint16_t>(chip), dst_mesh_id);
        rh->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{get_noc_addr(xy & 0xFFFF, xy >> 16, ready_addr), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(rh), sizeof(PACKET_HEADER_TYPE));
#if defined(FMR_DPRINT)
        DPRINT("FMRD ready to chip {} rb {},{}\n", chip, (uint32_t)rh->route_buffer[0], (uint32_t)rh->route_buffer[1]);
#endif
    }
    if (ready_expect > 0) {
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
        AGRN_ZONE("ps_fence_wait");
        noc_semaphore_wait_min(ready, ready_expect);
        noc_semaphore_inc(get_noc_addr(ready_addr), 0u - ready_expect);
        noc_async_atomic_barrier();
    }
    // ---- gather_normalized_slices: one walk, two multicast segments ----
    const uint64_t arrival_noc = get_noc_addr(peer_xy & 0xFFFF, peer_xy >> 16, arrival_addr);
    uint32_t unflushed = 0;
    uint32_t h_next = 0, dirty = 0;
    auto next_header = [&](uint32_t spec) {
        const uint32_t hk = h_next;
        h_next = (h_next + 1 == num_headers) ? 0 : h_next + 1;
        if (dirty & (1u << hk)) {
            noc_async_writes_flushed();
            dirty = 0;
        }
        dirty |= 1u << hk;
        if (hdr_spec[hk] != spec) {
            set_spec_route(hdrs[hk], dst_chip_id, dst_mesh_id, spec);
            hdr_spec[hk] = spec;
        }
        return hdrs[hk];
    };
    auto fill_header = [&](volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, const Packet& pk, bool inc) -> bool {
        const uint32_t bytes = pk.pages * page_bytes;
        uint16_t sz[4];
        for (uint32_t ri = 0; ri < pk.nr; ++ri) {
            sz[ri] = static_cast<uint16_t>(pk.n[ri] * page_bytes);
        }
        if (fast_banks && !inc) {
            if (pk.nr == 1) {
                hdr->noc_send_type = NOC_UNICAST_WRITE;
                hdr->command_fields.unicast_write.noc_address =
                    bank_base[pk.first[0] % nb] + (pk.first[0] / nb) * bank_pitch;
            } else {
                hdr->noc_send_type = NOC_UNICAST_SCATTER_WRITE;
                volatile tt_l1_ptr uint32_t* w =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&hdr->command_fields.unicast_scatter_write);
                for (uint32_t ri = 0; ri < NOC_SCATTER_WRITE_MAX_CHUNKS; ++ri) {
                    const uint64_t ad =
                        ri < pk.nr ? bank_base[pk.first[ri] % nb] + (pk.first[ri] / nb) * bank_pitch : 0;
                    w[2 * ri] = static_cast<uint32_t>(ad);
                    w[2 * ri + 1] = static_cast<uint32_t>(ad >> 32);
                }
                const uint32_t s1 = pk.nr > 2 ? sz[1] : 0, s2 = pk.nr > 3 ? sz[2] : 0;
                w[8] = static_cast<uint32_t>(sz[0]) | (s1 << 16);
                w[9] = s2 | (pk.nr << 16) | (static_cast<uint32_t>(scatter_enc) << 24);
            }
            hdr->payload_size_bytes = static_cast<uint16_t>(bytes);
            return false;
        }
        uint64_t d[4];
        for (uint32_t ri = 0; ri < pk.nr; ++ri) {
            d[ri] = dst.get_noc_addr(pk.first[ri], 0, 0);
        }
        if (pk.nr == 1) {
            if (inc) {
                hdr->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{d[0], arrival_noc, 1, true}, bytes);
            } else {
                hdr->to_noc_unicast_write(NocUnicastCommandHeader{d[0]}, bytes);
            }
            return false;
        }
        if (pk.nr == 2 && inc) {
            hdr->to_noc_fused_unicast_scatter_write_atomic_inc(
                NocUnicastScatterAtomicIncFusedCommandHeader({d[0], d[1]}, arrival_noc, {sz[0]}, 1, true), bytes);
            return false;
        }
        hdr->to_noc_unicast_scatter_write(NocUnicastScatterCommandHeader(d, sz, static_cast<uint8_t>(pk.nr)), bytes);
        return inc;
    };

    AGRN_ZONE("ps_bulk");
    constexpr uint32_t n_slots = 2 * group;
    const uint32_t desc_base = get_read_ptr(cb_desc);
    for (uint32_t pi = 0; pi < n_pkts; ++pi) {
        cb_wait_front(cb_chunks, run_pages * (unflushed + 1));
        invalidate_l1_cache();
        Packet pk;
        {
            volatile tt_l1_ptr PacketDesc* dsc =
                reinterpret_cast<volatile tt_l1_ptr PacketDesc*>(desc_base + (pi % n_slots) * DESC_BYTES);
            pk.nr = dsc->nr;
            pk.pages = dsc->pages;
            for (uint32_t ri = 0; ri < pk.nr; ++ri) {
                pk.first[ri] = dsc->first[ri];
                pk.n[ri] = dsc->n[ri];
            }
        }
        const uint32_t src = get_read_ptr(cb_chunks) + unflushed * chunk_bytes;
        const uint32_t spec = pi < seg_split ? spec_a : spec_b;
        const bool inc = pi + 1 == seg_split || pi + 1 == n_pkts;  // last packet of a segment
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = next_header(spec);
        const bool extra_inc = fill_header(hdr, pk, inc);
        conn.wait_for_empty_write_slot();
        conn.send_current_slot_non_blocking(src, pk.pages * page_bytes, reinterpret_cast<uint32_t>(hdr));
        if (extra_inc) {
            volatile tt_l1_ptr PACKET_HEADER_TYPE* ih = next_header(spec);
            ih->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{arrival_noc, 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(ih), sizeof(PACKET_HEADER_TYPE));
        }
        if (++unflushed == group || pi + 1 == n_pkts) {
            noc_async_writes_flushed();
            dirty = 0;
            cb_pop_front(cb_chunks, run_pages * unflushed);
            unflushed = 0;
        }
    }
    noc_async_writes_flushed();
    conn.close();
    // every arrival of this chip's lane landed (the counter may sit on another core: polled over the NoC), then re-arm
    if (wait_expect > 0) {
        AGRN_ZONE("ps_wait_all");
        const uint64_t ctr = get_noc_addr(peer_xy & 0xFFFF, peer_xy >> 16, arrival_addr);
        volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(desc_base);
        while (true) {
            noc_async_read(ctr, desc_base, 4);
            noc_async_read_barrier();
            invalidate_l1_cache();
            if (*scratch >= wait_expect) {
                break;
            }
        }
        noc_semaphore_inc(ctr, 0u - wait_expect);
        noc_async_atomic_barrier();
    }
}
