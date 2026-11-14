// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post feed port sender (RISCV_0, NoC1), regime R4 feed-DRAM-staged.
//
// TRANSPORT from the fabric examples (fabric_all_gather's sender, fabric_ring_exchange's direct 2-D sends):
//   ready fence  this program running means every earlier command on this chip is done (the previous call's feed
//                output has been read), so every peer may now write into it: one header-only atomic inc of the
//                ready counter of the port that writes into this chip, per peer reached through my direction; then
//                no data packet until every destination of mine said the same (ready >= #destinations).
//   sends        the chunks the reader staged (this link's share of the shared walk), each one packet written
//                straight into the destination's feed output (same address on every chip), addressed by
//                destination node (2-D routing, multi-hop through the fabric); the last chunk of a destination
//                stream carries the fused increment of the destination port's arrival counter (a header-only
//                increment if the stream is empty). Up to `group` packets in flight, one header each.
//   drain        wait until everything addressed to this core arrived (one increment per incoming stream), then
//                subtract what was consumed from both counters: back at 0 after every call (trace-safe).
// T % 32 != 0 (Refinement 2b): my own block goes through the ports too (destination me, hops 0: local NoC writes
// into my D), and the pieces of shared D tile rows go to the landing port (lx, ly) of the destination, one fused
// write + arrival increment each (local writes + one increment for my own). The landing port waits for its streams
// and every piece, merges each shared tile from its contributors' compact blocks and writes it to its D: one chip,
// one core writes each D tile.
//
// Raw fabric API (WorkerToFabricEdmSender + PacketHeaderPool), not ccl_helpers_dataflow.hpp: the helper's
// connection policy builds from the C++ FabricConnectionManager argument block and routes by hop count (1-D), while
// this op's Python program descriptor gets its connection arguments from ttnn.setup_fabric_connection and routes
// by destination node (2-D, the far chip reached directly) — the examples' measured path.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/ttnn/operations/mhc_post/kernels/mhc_post_feed_walk.hpp"

using namespace tt::tt_fabric;

namespace {
inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id, uint16_t hops) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)hops;
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), hops);
#endif
}
constexpr uint32_t DEST_WORDS = 4;   // position d, chip id, mesh id, hops
constexpr uint32_t READY_WORDS = 5;  // chip id, mesh id, hops, target port x, y
constexpr uint32_t MAX_DEST = 8;
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_chunks = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t run_pages = get_compile_time_arg_val(2);
    constexpr uint32_t num_banks = get_compile_time_arg_val(3);
    constexpr uint32_t links = get_compile_time_arg_val(4);
    constexpr uint32_t wave_rows = get_compile_time_arg_val(5);
    constexpr uint32_t max_rows = get_compile_time_arg_val(6);
    constexpr uint32_t group = get_compile_time_arg_val(7);        // packets in flight (one header each)
    constexpr uint32_t ring_chunks = get_compile_time_arg_val(8);  // chunk slots of the ring
    constexpr uint32_t cb_land = get_compile_time_arg_val(9);      // landing of shared D tile rows (Refinement 2b)
    constexpr auto d_args = TensorAccessorArgs<10>();
    static_assert(page_bytes == 2048, "the feed moves bf16 tiles");
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    static_assert(group >= 1 && group + 1 <= 12, "PacketHeaderPool: <= 12 headers per RISC");

    size_t a = 0;
    mhc_feed::Geom g;
    g.c_tiles = get_arg_val<uint32_t>(a++);
    g.blk_tiles = get_arg_val<uint32_t>(a++);
    g.ht = get_arg_val<uint32_t>(a++);
    g.g = get_arg_val<uint32_t>(a++);
    g.me = get_arg_val<uint32_t>(a++);
    g.r = get_arg_val<uint32_t>(a++);
    g.rows_base = get_arg_val<uint32_t>(a++);
    g.rows_extra = get_arg_val<uint32_t>(a++);
    g.t_rows = get_arg_val<uint32_t>(a++);
    g.htd = get_arg_val<uint32_t>(a++);
    g.nimg = get_arg_val<uint32_t>(a++);
    g.nsh = get_arg_val<uint32_t>(a++);
    const uint32_t link = get_arg_val<uint32_t>(a++);
    const uint32_t d_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    const uint32_t my_x = get_arg_val<uint32_t>(a++);  // this port's virtual coords = the destination port's
    const uint32_t my_y = get_arg_val<uint32_t>(a++);
    const uint32_t expect_in = get_arg_val<uint32_t>(a++);  // streams arriving at this core
    const uint32_t land_x = get_arg_val<uint32_t>(a++);     // the landing port (the same core on every chip)
    const uint32_t land_y = get_arg_val<uint32_t>(a++);
    const bool is_landing = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t n_dest = get_arg_val<uint32_t>(a++);   // destinations, mine (hops 0) included
    const uint32_t n_ready = get_arg_val<uint32_t>(a++);  // remote destinations = readies to send and to wait for
    const bool has_conn = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t dest_idx = a;
    a += n_dest * DEST_WORDS;
    const uint32_t ready_idx = a;
    a += n_ready * READY_WORDS;

    const auto out = TensorAccessor(d_args, d_addr, page_bytes);

    const uint32_t land_base = g.nsh > 0 ? get_write_ptr(cb_land) : 0;
    const uint64_t land_arrival = get_noc_addr(land_x, land_y, arrival_addr);
    const uint32_t ring_base = get_read_ptr(cb_chunks);  // nothing consumed yet: the ring's base
    uint32_t self_pieces = 0;
    if (n_dest > 0) {
        WorkerToFabricEdmSender conn;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[group];
        uint32_t hdr_dest[group];
        volatile tt_l1_ptr PACKET_HEADER_TYPE* ctl = nullptr;  // readies, empty incs
        if (has_conn) {
            conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
            for (uint32_t h = 0; h < group; ++h) {
                hdrs[h] = PacketHeaderPool::allocate_header();
                hdr_dest[h] = 0xFFFFFFFFu;
            }
            ctl = PacketHeaderPool::allocate_header();
            MaybeDeviceZoneScope("s_ready");
            conn.open();

            for (uint32_t k = 0; k < n_ready; ++k) {
                const uint32_t t = ready_idx + k * READY_WORDS;
                route_to(
                    ctl,
                    static_cast<uint16_t>(get_arg_val<uint32_t>(t)),
                    static_cast<uint16_t>(get_arg_val<uint32_t>(t + 1)),
                    static_cast<uint16_t>(get_arg_val<uint32_t>(t + 2)));
                ctl->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                    get_noc_addr(get_arg_val<uint32_t>(t + 3), get_arg_val<uint32_t>(t + 4), ready_addr), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(ctl), sizeof(PACKET_HEADER_TYPE));
            }
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, n_ready);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - n_ready);
            noc_async_atomic_barrier();
        }
        auto is_local = [&](uint32_t k) { return get_arg_val<uint32_t>(dest_idx + k * DEST_WORDS + 3) == 0; };

        // my D-page chunks per remote destination stream (the last one carries the arrival increment)
        uint32_t total[MAX_DEST];
        for (uint32_t k = 0; k < n_dest; ++k) {
            total[k] = 0;
        }
        for (uint32_t i0 = 0, w = 0; i0 < max_rows; i0 += wave_rows, ++w) {
            uint32_t ci = 0, pages = 0;  // the walk is the same for every destination
            mhc_feed::for_each_item<num_banks, run_pages, wave_rows>(
                g, w, ci, [&](uint32_t c, const mhc_feed::Item& it) {
                    pages += (!it.piece && c % links == link) ? 1 : 0;
                });
            for (uint32_t k = 0; k < n_dest; ++k) {
                total[k] += pages;
            }
        }

        const uint64_t arrival_noc = get_noc_addr(my_x, my_y, arrival_addr);
        uint32_t slot = 0, unflushed = 0;
        uint32_t sent[MAX_DEST];
        for (uint32_t k = 0; k < n_dest; ++k) {
            sent[k] = 0;
        }
        auto flush = [&]() {
            noc_async_writes_flushed();  // sources + headers of the in-flight packets have left L1
            for (uint32_t u = 0; u < unflushed; ++u) {
                cb_pop_front(cb_chunks, run_pages);
            }
            unflushed = 0;
        };
        for (uint32_t i0 = 0, w = 0; i0 < max_rows; i0 += wave_rows, ++w) {
            MaybeDeviceZoneScope("s_send");  // ring waits (reader-bound) + fabric slot waits + sends of one wave
            for (uint32_t k = 0; k < n_dest; ++k) {
                const uint32_t t = dest_idx + k * DEST_WORDS;
                const bool local = is_local(k);
                uint32_t ci = 0;
                mhc_feed::for_each_item<num_banks, run_pages, wave_rows>(
                    g, w, ci, [&](uint32_t c, const mhc_feed::Item& it) {
                        if (c % links != link) {
                            return;
                        }
                        cb_wait_front(cb_chunks, run_pages * (unflushed + 1));
                        const uint32_t src = ring_base + slot * chunk_bytes;
                        slot = (slot + 1 == ring_chunks) ? 0 : slot + 1;
                        uint64_t dst;
                        uint32_t bytes;
                        if (it.piece) {
                            dst = get_noc_addr(land_x, land_y, land_base + mhc_feed::piece_offset(g, it));
                            bytes = mhc_feed::piece_bytes(it);
                        } else {
                            dst = out.get_noc_addr(it.j, 0, 0);  // the same page on the destination chip
                            bytes = it.n * page_bytes;
                        }
                        if (local) {
                            noc_async_write(src, dst, bytes);
                            self_pieces += it.piece ? 1 : 0;
                        } else {
                            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[unflushed];  // free: flushed before reuse
                            if (hdr_dest[unflushed] != k) {
                                route_to(
                                    hdr,
                                    static_cast<uint16_t>(get_arg_val<uint32_t>(t + 1)),
                                    static_cast<uint16_t>(get_arg_val<uint32_t>(t + 2)),
                                    static_cast<uint16_t>(get_arg_val<uint32_t>(t + 3)));
                                hdr_dest[unflushed] = k;
                            }
                            if (it.piece) {
                                hdr->to_noc_fused_unicast_write_atomic_inc(
                                    NocUnicastAtomicIncFusedCommandHeader{dst, land_arrival, 1, true}, bytes);
                            } else if (++sent[k] == total[k]) {
                                hdr->to_noc_fused_unicast_write_atomic_inc(
                                    NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, bytes);
                            } else {
                                hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                            }
                            conn.wait_for_empty_write_slot();
                            conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                        }
                        if (++unflushed == group) {
                            flush();
                        }
                    });
            }
        }
        flush();
        if (has_conn) {
            for (uint32_t k = 0; k < n_dest; ++k) {  // an empty remote stream still counts once at its destination
                if (total[k] == 0 && !is_local(k)) {
                    const uint32_t t = dest_idx + k * DEST_WORDS;
                    route_to(
                        ctl,
                        static_cast<uint16_t>(get_arg_val<uint32_t>(t + 1)),
                        static_cast<uint16_t>(get_arg_val<uint32_t>(t + 2)),
                        static_cast<uint16_t>(get_arg_val<uint32_t>(t + 3)));
                    ctl->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{arrival_noc, 1, true});
                    conn.wait_for_empty_write_slot();
                    conn.send_payload_flush_blocking_from_address(
                        reinterpret_cast<uint32_t>(ctl), sizeof(PACKET_HEADER_TYPE));
                }
            }
            noc_async_writes_flushed();
            conn.close();
        }
    }
    noc_async_write_barrier();  // my local D pages and pieces have landed
    if (self_pieces > 0) {
        noc_semaphore_inc(land_arrival, self_pieces);
    }

    // everything addressed to this core has landed (each stream's last packet incremented after its data; the
    // landing port also counts every piece); re-arm
    const uint32_t expect = expect_in + (is_landing ? mhc_feed::landing_pieces<run_pages>(g) : 0);
    if (expect > 0) {
        MaybeDeviceZoneScope("s_arrival");
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect);
        noc_async_atomic_barrier();
    }

    // landing port: merge every shared D tile from its contributors' compact blocks (the drained ring is the
    // scratch) and write it to my D
    if (is_landing && g.nsh > 0) {
        MaybeDeviceZoneScope("s_landing");
        constexpr uint32_t cap = ring_chunks * run_pages;  // tiles a batch
        for (uint32_t img = 0; img < g.nimg; ++img) {
            for (uint32_t row = 0, s = 0; row < g.htd; ++row) {
                if (!mhc_feed::shared_row(g, row)) {
                    continue;
                }
                const uint32_t blk_row = land_base + (img * g.nsh + s++) * g.blk_tiles * page_bytes;
                for (uint32_t c0 = 0; c0 < g.blk_tiles; c0 += cap) {
                    const uint32_t c1 = mhc_feed::min_u(c0 + cap, g.blk_tiles);
                    for (uint32_t c = c0; c < c1; ++c) {
                        const uint32_t tile = ring_base + (c - c0) * page_bytes;
                        for (uint32_t q = mhc_feed::first_chip(g, row); q <= mhc_feed::last_chip(g, row); ++q) {
                            const uint32_t ra = mhc_feed::rows_a(g, q, row), rb = mhc_feed::rows_b(g, q, row);
                            const uint32_t col = blk_row + ra * mhc_feed::TILE_ROW_BYTES * g.blk_tiles +
                                                 c * (rb - ra) * mhc_feed::TILE_ROW_BYTES;
                            mhc_feed::copy_rows(
                                ra,
                                ra,
                                rb - ra,
                                [tile](uint32_t r, uint32_t h) { return mhc_feed::tile_row_addr(tile, r, h); },
                                [col, ra, rb](uint32_t r, uint32_t h) {
                                    return mhc_feed::compact_row_addr(col, ra, rb, r, h);
                                });
                        }
                    }
                    noc_async_read_barrier();
                    for (uint32_t c = c0; c < c1; ++c) {
                        noc_async_write(
                            ring_base + (c - c0) * page_bytes,
                            out.get_noc_addr((img * g.htd + row) * g.blk_tiles + c, 0, 0),
                            page_bytes);
                    }
                    noc_async_write_barrier();
                }
            }
        }
    }
}
