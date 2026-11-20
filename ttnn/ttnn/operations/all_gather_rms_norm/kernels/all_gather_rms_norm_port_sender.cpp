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
// Header ring (hdr_ring > 0, TRANSPORT, Refinement 6): with 1-tile packets the sender's per-packet RISC work (walk
//   the runs, build the header, issue) is the transport's rate, not the link. Every packet's header is built into
//   its own slot of an L1 ring (cb_port_hdr, routed once per slot) AHEAD of its send — while the sender waits for
//   the ready fence, a stage slot or a router credit — so the send itself only issues. A slot is rebuilt once the
//   packet that used it was flushed. Stateful sends (`stateful`): the router's write / credit command buffers are
//   programmed once after the fence; router credits are cached (one poll per credit refill, not per packet).
//   hdr_ring == 0: one header per batch position built right before its send (the original sender).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "all_gather_rms_norm_common.hpp"
#include "all_gather_rms_norm_port_pack.hpp"

using namespace tt::tt_fabric;
using namespace agrms_cb;

constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
constexpr uint32_t chunk_pages = get_compile_time_arg_val(1);
constexpr uint32_t group = get_compile_time_arg_val(2);
constexpr uint32_t chunk_bytes = chunk_pages * tile_bytes;
constexpr bool pack = get_compile_time_arg_val(3) != 0;  // packets span runs of an entry (fabric scatter write)
constexpr uint32_t num_data_headers = group;             // reused after the batch flush
constexpr uint32_t hdr_ring = get_compile_time_arg_val(4);   // header ring slots in cb_port_hdr (0 = per-batch headers)
constexpr bool stateful = get_compile_time_arg_val(5) != 0;  // stateful router sends + cached credits
constexpr uint32_t hdr_stride = get_compile_time_arg_val(6);  // bytes per header ring slot
// slot templates: 1 = the RISC copies the routed header into a slot at its first build (inside the waits), 0 = DMA of
// the template into every slot after the connection opens (a write barrier before the fence wait)
constexpr bool hdr_copy = get_compile_time_arg_val(7) != 0;
static_assert(hdr_ring == 0 || sizeof(PACKET_HEADER_TYPE) <= hdr_stride, "header ring slot too small");
static_assert(hdr_ring == 0 || hdr_ring >= group, "a header slot is reused only after its packet was flushed");

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

// Walks every packet of every outgoing entry in send order (runs given as runtime args, see the port packer).
template <bool pk, uint32_t cp>
struct PacketWalk {
    uint32_t e = 0;
    uint32_t num_out;
    uint32_t a;  // arg index of the current entry's run count
    agrms_pack::EntryPacker<pk, cp> packer{0, 0};

    PacketWalk(uint32_t first_arg, uint32_t n) : num_out(n), a(first_arg) {
        if (num_out > 0) {
            packer = agrms_pack::EntryPacker<pk, cp>{a + 1, arg(a)};
        }
    }
    bool done() const { return e >= num_out; }
    uint32_t next(agrms_pack::Piece* p, uint32_t& tiles, bool& entry_last) {
        const uint32_t np = packer.next(p, tiles, entry_last);
        if (entry_last) {
            a += 1 + 2 * arg(a);
            if (++e < num_out) {
                packer = agrms_pack::EntryPacker<pk, cp>{a + 1, arg(a)};
            }
        }
        return np;
    }
};

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
    const uint32_t num_out = arg(10);
    constexpr uint32_t first_entry = 11;
    uint32_t fab = first_entry;
    for (uint32_t e = 0; e < num_out; ++e) {
        fab += 1 + 2 * arg(fab);
    }
    size_t fa = fab;
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(fa);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_data_headers];
    for (uint32_t h = 0; h < num_data_headers; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
    }
    const uint32_t x_base = get_write_ptr(cb_x);
    const uint64_t arrival_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
    // entry-end increments after a >2-piece scatter: a constant header (re-sent while an earlier send of it may still
    // be in flight: the NoC only reads it)
    volatile tt_l1_ptr PACKET_HEADER_TYPE* inc_hdr = hdrs[0];
    if constexpr (pack) {
        inc_hdr = PacketHeaderPool::allocate_header();
        route_one_hop(inc_hdr, dst_chip_id, dst_mesh_id);
        inc_hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{arrival_noc, 1, true});
    }
    auto dst_of = [&](const agrms_pack::Piece& p) {
        return get_noc_addr(p.xy & 0xFFFF, p.xy >> 16, x_base + p.off * tile_bytes);
    };
    // Header of one packet; returns its payload bytes | (a separate entry-end increment follows) << 31.
    auto build = [&](volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr,
                     const agrms_pack::Piece* pc,
                     uint32_t np,
                     uint32_t tiles,
                     bool entry_last) -> uint32_t {
        const uint32_t bytes = tiles * tile_bytes;
        uint32_t separate_inc = 0;
#ifdef AGRMS_ABL
        if constexpr ((AGRMS_ABL & 4) != 0) {  // perf probe: one contiguous write per packet on the peer port core
            const uint64_t d = get_noc_addr(peer_x, peer_y, x_base);
            if (entry_last) {
                hdr->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{d, arrival_noc, 1, true}, bytes);
            } else {
                hdr->to_noc_unicast_write(NocUnicastCommandHeader{d}, bytes);
            }
            return bytes;
        }
#endif
        if (np == 1) {
            if (entry_last) {
                hdr->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{dst_of(pc[0]), arrival_noc, 1, true}, bytes);
            } else {
                hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst_of(pc[0])}, bytes);
            }
        } else if (np == 2 && entry_last) {
            hdr->to_noc_fused_unicast_scatter_write_atomic_inc(
                NocUnicastScatterAtomicIncFusedCommandHeader(
                    {dst_of(pc[0]), dst_of(pc[1])},
                    arrival_noc,
                    {static_cast<uint16_t>(pc[0].n * tile_bytes)},
                    1,
                    true),
                bytes);
        } else {
            uint64_t dsts[agrms_pack::MAX_PIECES];
            uint16_t sizes[agrms_pack::MAX_PIECES - 1];
            for (uint32_t i = 0; i < np; ++i) {
                dsts[i] = dst_of(pc[i]);
                if (i + 1 < np) {
                    sizes[i] = static_cast<uint16_t>(pc[i].n * tile_bytes);
                }
            }
            hdr->to_noc_unicast_scatter_write(
                NocUnicastScatterCommandHeader(dsts, sizes, static_cast<uint8_t>(np)), bytes);
            separate_inc = entry_last ? 1u : 0u;  // the fused scatter + inc form carries 2 write chunks only
        }
        return bytes | (separate_inc << 31);
    };

    PacketWalk<pack, chunk_pages> walk(first_entry, num_out);
    agrms_pack::Piece pc[agrms_pack::MAX_PIECES];
    // header ring state: packets built / sent / flushed (slot of packet i = i % hdr_ring)
    const uint32_t ring_base = hdr_ring > 0 ? get_write_ptr(cb_port_hdr) : 0;
    uint32_t meta[hdr_ring > 0 ? hdr_ring : 1];
    uint32_t built = 0, sent = 0, flushed = 0;
    auto slot_hdr = [&](uint32_t i) {
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            ring_base + (i % (hdr_ring > 0 ? hdr_ring : 1)) * hdr_stride);
    };
    // build the next packet's header if its slot is free; false when nothing could be built
    auto build_ahead = [&]() -> bool {
        if constexpr (hdr_ring > 0) {
            if (walk.done() || built >= flushed + hdr_ring) {
                return false;
            }
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = slot_hdr(built);
            if (hdr_copy && built < hdr_ring) {
                // a slot's route never changes: copy it once from the routed pool header (computing it costs more)
                constexpr uint32_t words = sizeof(PACKET_HEADER_TYPE) / sizeof(uint32_t);
                static_assert(sizeof(PACKET_HEADER_TYPE) % sizeof(uint32_t) == 0);
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdr);
                volatile tt_l1_ptr uint32_t* t =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdrs[num_data_headers - 1]);
                for (uint32_t w = 0; w < words; ++w) {
                    d[w] = t[w];
                }
            }
            uint32_t tiles;
            bool entry_last;
            const uint32_t np = walk.next(pc, tiles, entry_last);
            meta[built % hdr_ring] = build(hdr, pc, np, tiles, entry_last);
            ++built;
            return true;
        } else {
            return false;
        }
    };

    {
        AGZ("ps_open");
        conn.open();
    }
    if constexpr (hdr_ring > 0 && !hdr_copy) {
        // every slot gets the routed header by DMA (a slot's route never changes; computing or RISC-copying it per
        // slot costs more than a packet's whole send); waited for before the first build
        const uint32_t tmpl = reinterpret_cast<uint32_t>(hdrs[num_data_headers - 1]);
        for (uint32_t i = 0; i < hdr_ring; ++i) {
            noc_async_write(tmpl, get_noc_addr(ring_base + i * hdr_stride), sizeof(PACKET_HEADER_TYPE));
        }
    }
    {
        AGZ("ps_fence");
        if (send_ready) {
            hdrs[0]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if constexpr (hdr_ring > 0 && !hdr_copy) {
            noc_async_write_barrier();  // the slot templates (and the ready) have left / landed
        }
        if (num_out > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            while (true) {  // the fence wait builds headers ahead
                invalidate_l1_cache();
                if (*ready >= 1) {
                    break;
                }
                build_ahead();  // (none here measured +0.64 us on the indexer case)
            }
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
    }
    AGZ("ps_send");
    if constexpr (stateful) {
        conn.setup_stateful_send_cmd_bufs();
    }
    uint32_t credits = 0;  // stateful: router slots known free
    auto take_credit = [&]() {
        if constexpr (stateful) {
            while (credits == 0) {
                credits = conn.get_num_free_write_slots();
                if (credits == 0) {
                    build_ahead();
                }
            }
            --credits;
        } else {
            conn.wait_for_empty_write_slot();
        }
    };
    auto issue = [&](uint32_t src, uint32_t bytes, volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr) {
        take_credit();
        if constexpr (stateful) {
            conn.send_current_slot_stateful_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
        } else {
            conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
        }
    };
    auto issue_inc = [&]() {
        take_credit();
        if constexpr (stateful) {
            conn.send_current_slot_stateful_non_blocking_from_address(
                reinterpret_cast<uint32_t>(inc_hdr), sizeof(PACKET_HEADER_TYPE));
        } else {
            conn.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(inc_hdr), sizeof(PACKET_HEADER_TYPE));
        }
    };
    uint32_t h = 0, unflushed = 0;
    while (!walk.done() || sent < built) {
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr;
        uint32_t m;
        if constexpr (hdr_ring > 0) {
            if (sent == built) {
                build_ahead();  // always succeeds: hdr_ring >= group > unflushed
            }
            hdr = slot_hdr(sent);
            m = meta[sent % hdr_ring];
        } else {
            uint32_t tiles;
            bool entry_last;
            const uint32_t np = walk.next(pc, tiles, entry_last);
            hdr = hdrs[h];
            h = (h + 1 == num_data_headers) ? 0 : h + 1;
            m = build(hdr, pc, np, tiles, entry_last);
        }
        while (!cb_pages_available_at_front(cb_stage, chunk_pages * (unflushed + 1))) {
            build_ahead();
        }
        const uint32_t src = get_read_ptr(cb_stage) + unflushed * chunk_bytes;
        issue(src, m & 0x7FFFFFFFu, hdr);
        if ((m >> 31) != 0) {
            // header-only increment, flushed by the receiving router after every earlier write of the channel
            issue_inc();
        }
        ++sent;
        if (++unflushed == group) {
            noc_async_writes_flushed();
            cb_pop_front(cb_stage, chunk_pages * unflushed);
            flushed = sent;
            unflushed = 0;
        }
    }
    if (unflushed > 0) {
        noc_async_writes_flushed();
        cb_pop_front(cb_stage, chunk_pages * unflushed);
    }
    {
        AGZ("ps_close");
        conn.close();
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    }
}
