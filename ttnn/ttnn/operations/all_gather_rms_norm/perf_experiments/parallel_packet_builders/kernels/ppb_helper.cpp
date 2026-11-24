// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// parallel_packet_builders: helper core h of a (direction, link) port (NCRISC / NoC0). Raw dataflow, no helper library:
// this is the op's port reader + the header half of its port sender, split across H cores so that the per-packet CPU
// work (walk, run reads, header command fields) runs in parallel; the port core only forwards finished packets.
//
// For every packet i of the port's walk with i % H == h (j = i / H):
//   reads its runs (own entry: the own final pages / staging; relay: after the arrival threshold, forwarded by the port
//   NCRISC into SEM_MIRROR) into a local slot laid out [meta][header][payload] (payload 64 B aligned), fills the
//   header's command fields (route set once per local slot), and — once the port freed ring slot (j % D) * H + h
//   (SEM_CREDIT >= j - D + 1) — writes meta + header + payload with ONE NoC write, flushes, and publishes j + 1 in the
//   port's SEM_READY0 + h (inline write behind the data on the same path: the fabric worker adapter's own ordering).
// Reads are pipelined over P local slots (one transaction id each), retired eagerly once landed.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "agrn_port_common.hpp"
#include "ppb_common.hpp"
#if defined(PPB_DEBUG)
#include "api/debug/dprint.h"
#endif

using namespace tt::tt_fabric;
using namespace agrn_port;

constexpr bool fast_banks = shard_w == 0 && nb <= 16;
constexpr uint32_t HDR = sizeof(PACKET_HEADER_TYPE);
static_assert(HDR % 16 == 0 && ppb::META_BYTES + HDR <= 128, "header size (bench HDR_REGION)");
constexpr uint32_t PAY_OFF = ((ppb::META_BYTES + HDR + 63) / 64) * 64;  // payload offset in a local slot
constexpr uint32_t META_OFF = PAY_OFF - HDR - ppb::META_BYTES;
constexpr uint8_t scatter_enc =
    expand_encoding_to_all_chunks(NocScatterWriteChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE);

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
    size_t a = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t H = get_arg_val<uint32_t>(a++);
    const uint32_t h = get_arg_val<uint32_t>(a++);
    const uint32_t D = get_arg_val<uint32_t>(a++);
    const uint32_t P = get_arg_val<uint32_t>(a++);  // local slots (reads in flight), <= D, <= 15
    const uint32_t port_xy = get_arg_val<uint32_t>(a++);
    const uint32_t port_ring = get_write_ptr(get_arg_val<uint32_t>(a++));  // port ring CB: same address on every core
    const uint32_t port_pitch = get_arg_val<uint32_t>(a++);
    const uint32_t local_pitch = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);  // downstream port's arrival counter
    const uint32_t peer_xy = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const bool wait_own = get_arg_val<uint32_t>(a++) != 0;  // own entry waits SEM_OWN (the port forwards readiness)
    const uint32_t n_entries = get_arg_val<uint32_t>(a++);
    const uint32_t entries_idx = a;  // per entry: q | part << 16, A, B, packets
    a += 4 * n_entries;
    // staged own entry (op: Perf 1 staging on the workers); 0 in the isolated bench
    const bool st_on = get_arg_val<uint32_t>(a++) != 0;
    uint32_t st_pages = 0, st_nact = 1, st_nch = 0, st_lane = 0, st_L = 1, st_cp = 1, st_tbl = 0;
    if (st_on) {
        st_pages = get_arg_val<uint32_t>(a++);
        st_nact = get_arg_val<uint32_t>(a++);
        st_nch = get_arg_val<uint32_t>(a++);
        st_lane = get_arg_val<uint32_t>(a++);
        st_L = get_arg_val<uint32_t>(a++);
        st_cp = get_arg_val<uint32_t>(a++);
        st_tbl = a;  // st_nact worker NoC coords
        a += st_nact;
    }

    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    const uint32_t px = port_xy & 0xFFFF, py = port_xy >> 16;
    const uint64_t ready_noc = get_noc_addr(px, py, get_semaphore(ppb::SEM_READY0 + h));
    const uint64_t arrival_noc = get_noc_addr(peer_xy & 0xFFFF, peer_xy >> 16, arrival_addr);
    volatile tt_l1_ptr uint32_t* mirror =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ppb::SEM_MIRROR));
    volatile tt_l1_ptr uint32_t* credit =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ppb::SEM_CREDIT));
    volatile tt_l1_ptr uint32_t* own_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ppb::SEM_OWN));
    const uint32_t local_base = (get_write_ptr(cb_chunks) + 63) & ~63u;
    const uint32_t stage_base = stage_on ? get_write_ptr(cb_stage) : 0u;

    // route once per local slot (only the command fields change per packet)
    for (uint32_t s = 0; s < P; ++s) {
        route_one_hop(
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(local_base + s * local_pitch + PAY_OFF - HDR),
            dst_chip_id,
            dst_mesh_id);
    }
    // router-side NoC address of every destination bank's first page + the bank page pitch (as the op's sender)
    uint64_t bank_base[fast_banks ? nb : 1];
    uint32_t bank_pitch = 0;
    uint64_t rd_bank_base[fast_banks ? nb : 1];
    uint32_t rd_bank_pitch = 0;
    if constexpr (fast_banks) {
        volatile tt_l1_ptr PACKET_HEADER_TYPE* sh =
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(local_base + PAY_OFF - HDR);
        for (uint32_t b = 0; b < nb; ++b) {
            sh->to_noc_unicast_write(NocUnicastCommandHeader{dst.get_noc_addr(b, 0, 0)}, 0);
            bank_base[b] = sh->command_fields.unicast_write.noc_address;
            rd_bank_base[b] = dst.get_noc_addr(b);
        }
        bank_pitch = static_cast<uint32_t>(dst.get_noc_addr(nb, 0, 0) - dst.get_noc_addr(0, 0, 0));
        rd_bank_pitch = static_cast<uint32_t>(dst.get_noc_addr(nb) - dst.get_noc_addr(0));
    }

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

    uint32_t issued = 0, retired = 0;  // my packets (j)
    auto credit_ok = [&](uint32_t j) {
        if (j < D) {
            return true;
        }
        invalidate_l1_cache();
        return *credit >= j - D + 1;
    };
    auto retire = [&]() {
        const uint32_t j = retired;
        const uint32_t ls = local_base + (j % P) * local_pitch;
        noc_async_read_barrier_with_trid(1 + j % P);
        while (!credit_ok(j)) {
        }
        const uint32_t bytes = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ls + META_OFF)[0];
        const uint32_t pslot = port_ring + ((j % D) * H + h) * port_pitch;
        noc_async_write(ls + META_OFF, get_noc_addr(px, py, pslot), ppb::META_BYTES + HDR + bytes);
        noc_async_writes_flushed();
        noc_inline_dw_write(ready_noc, j + 1);
        ++retired;
    };
    auto retire_ready = [&]() {
        while (retired < issued && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + retired % P) &&
               credit_ok(retired)) {
            retire();
        }
    };

    uint32_t i = 0;  // global packet index of the port walk
    uint32_t mirror_seen = 0;
    uint32_t own_lvl = wait_own ? 0u : 2u;  // SEM_OWN seen: 1 = staged prefix ready, 2 = own pages written
    for (uint32_t k = 0; k < n_entries; ++k) {
        const uint32_t e = get_arg_val<uint32_t>(entries_idx + 4 * k);
        const uint32_t A = get_arg_val<uint32_t>(entries_idx + 4 * k + 1);
        const uint32_t B = get_arg_val<uint32_t>(entries_idx + 4 * k + 2);
        const uint32_t total = get_arg_val<uint32_t>(entries_idx + 4 * k + 3);
        const uint32_t q = e & 0xFFFF, part = e >> 16;
        uint32_t c = 0;        // packet index in the entry
        uint32_t s_pages = 0;  // own entry: pages walked (staging position)
        for_each_packet(q, part, first_bank, bank_stride, [&](const Packet& pk) {
            const uint32_t my_c = c++;
            const uint32_t gi = i++;
            const uint32_t st_s = s_pages;
            if (k == 0) {
                s_pages += pk.pages;
            }
            if (gi % H != h) {
                return;
            }
            const bool inc = ((gi + 1) % inc_every) == 0 || my_c + 1 == total;
            if (issued - retired == P) {
                retire();
            }
            const bool staged = stage_on && k == 0 && st_on && st_s + pk.pages <= st_pages;
            const uint32_t own_need = staged ? 1u : 2u;
            if (k == 0 && own_lvl < own_need) {
                while (retired < issued) {
                    retire();
                }
                noc_semaphore_wait_min(own_sem, own_need);
                own_lvl = *own_sem;
            }
            if (k > 0) {
                const uint32_t need = A + (B + my_c) / inc_every;
                if (mirror_seen < need) {
                    invalidate_l1_cache();
                    mirror_seen = *mirror;
                    if (mirror_seen < need) {
                        // the arrivals may depend on my packets already read: hand them to the port first
                        while (retired < issued) {
                            retire();
                        }
                        noc_semaphore_wait_min(mirror, need);
                        mirror_seen = *mirror;
                    }
                }
            }
            const uint32_t ls = local_base + (issued % P) * local_pitch;
            uint32_t l1 = ls + PAY_OFF;
            noc_async_read_set_trid(1 + issued % P);
            if (staged) {
                uint32_t s0 = st_s;
                const uint32_t s1 = st_s + pk.pages;
                const uint32_t n_ch = st_L * st_nch;
                while (s0 < s1) {
                    const uint32_t cc = s0 / st_cp;
                    const uint32_t ce = (cc + 1) * st_cp;
                    const uint32_t en = ce < s1 ? ce : s1;
                    const uint32_t g = st_lane * st_nch + cc;
                    const uint32_t core = n_ch <= st_nact ? g * st_nact / n_ch : g % st_nact;
                    const uint32_t slot = n_ch <= st_nact ? 0u : g / st_nact;
                    const uint32_t xy = get_arg_val<uint32_t>(st_tbl + core);
                    const uint32_t src = stage_base + (slot * st_cp + (s0 - cc * st_cp)) * page_bytes;
                    noc_async_read(get_noc_addr(xy & 0xFFFF, xy >> 16, src), l1, (en - s0) * page_bytes);
                    l1 += (en - s0) * page_bytes;
                    s0 = en;
                }
            } else {
                for (uint32_t ri = 0; ri < pk.nr; ++ri) {
                    const uint32_t n = pk.n[ri];
                    if constexpr (fast_banks) {
                        noc_async_read(
                            rd_bank_base[pk.first[ri] % nb] + (pk.first[ri] / nb) * rd_bank_pitch, l1, n * page_bytes);
                    } else {
                        noc_async_read(dst.get_noc_addr(pk.first[ri]), l1, n * page_bytes);
                    }
                    l1 += n * page_bytes;
                }
            }
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr =
                reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(ls + PAY_OFF - HDR);
            const bool extra = fill_header(hdr, pk, inc);
            volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ls + META_OFF);
            meta[0] = pk.pages * page_bytes;
            meta[1] = extra ? 1u : 0u;
            ++issued;
            retire_ready();
        });
    }
    while (retired < issued) {
        retire();
    }
#if defined(PPB_DEBUG)
    DPRINT(
        "PPB helper {} done i {} issued {} ready_addr {} port {} {}\n",
        h,
        i,
        issued,
        get_semaphore(ppb::SEM_READY0 + h),
        px,
        py);
#endif
    noc_async_read_set_trid(0);
    noc_async_write_barrier();
}
