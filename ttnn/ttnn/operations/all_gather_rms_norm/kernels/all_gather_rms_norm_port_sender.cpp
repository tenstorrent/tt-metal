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
#include "all_gather_rms_norm_port_pack.hpp"

using namespace tt::tt_fabric;
using namespace agrms_cb;

constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
constexpr uint32_t chunk_pages = get_compile_time_arg_val(1);
constexpr uint32_t group = get_compile_time_arg_val(2);
constexpr uint32_t chunk_bytes = chunk_pages * tile_bytes;
constexpr bool pack = get_compile_time_arg_val(3) != 0;  // packets span runs of an entry (fabric scatter write)
constexpr uint32_t num_data_headers = group;             // reused after the batch flush

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
    const uint32_t num_out = arg(10);
    uint32_t a = 11;
    uint32_t fab = a;
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
    volatile tt_l1_ptr PACKET_HEADER_TYPE* inc_hdr = hdrs[0];  // entry-end increments after a >2-piece scatter
    if constexpr (pack) {
        inc_hdr = PacketHeaderPool::allocate_header();
        route_one_hop(inc_hdr, dst_chip_id, dst_mesh_id);
    }
    conn.open();
    {
        AGZ("ps_fence");
        if (send_ready) {
            hdrs[0]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (num_out > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
    }
    AGZ("ps_send");
    const uint32_t x_base = get_write_ptr(cb_x);
    const uint64_t arrival_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
    auto dst_of = [&](const agrms_pack::Piece& p) {
        return get_noc_addr(p.xy & 0xFFFF, p.xy >> 16, x_base + p.off * tile_bytes);
    };
    uint32_t h = 0, unflushed = 0;
    bool inc_hdr_busy = false;
    agrms_pack::Piece pc[agrms_pack::MAX_PIECES];
    for (uint32_t e = 0; e < num_out; ++e) {
        const uint32_t n_runs = arg(a);
        a += 1;
        agrms_pack::EntryPacker<pack, chunk_pages> packer{a, n_runs};
        while (!packer.done()) {
            uint32_t tiles;
            bool entry_last;
            const uint32_t np = packer.next(pc, tiles, entry_last);
            cb_wait_front(cb_stage, chunk_pages * (unflushed + 1));
            const uint32_t src = get_read_ptr(cb_stage) + unflushed * chunk_bytes;
            const uint32_t bytes = tiles * tile_bytes;
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
            h = (h + 1 == num_data_headers) ? 0 : h + 1;
            bool separate_inc = false;
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
                separate_inc = entry_last;  // the fused scatter + inc form carries 2 write chunks only
            }
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
            if (separate_inc) {
                // header-only increment, flushed by the receiving router after every earlier write of the channel
                if (inc_hdr_busy) {
                    noc_async_writes_flushed();
                }
                inc_hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{arrival_noc, 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_non_blocking_from_address(
                    reinterpret_cast<uint32_t>(inc_hdr), sizeof(PACKET_HEADER_TYPE));
                inc_hdr_busy = true;
            }
            if (++unflushed == group) {
                noc_async_writes_flushed();
                cb_pop_front(cb_stage, chunk_pages * unflushed);
                unflushed = 0;
                inc_hdr_busy = false;
            }
        }
        a += 2 * n_runs;
    }
    if (unflushed > 0) {
        noc_async_writes_flushed();
        cb_pop_front(cb_stage, chunk_pages * unflushed);
    }
    conn.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
