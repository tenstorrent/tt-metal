// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 fabric_tx writer (BRISC) — send_block + handshakes for one (direction, link) stream.
//
// One tx core = one DIRECT router connection = one or more destinations ("routes"): link 0 carries the
// neighbour chunk; link 1 carries the antipode half plus the last DIRECT_OVERFLOW_HEADS heads of the
// neighbour chunk (Perf 1: balances the per-worker-channel packet load, no fabric mux). Per route: its own
// dst chip, its own armed packet headers (inc, unicast tail, a scatter-header ring) and its own peer.
//
//   announce_ready : per route, fabric atomic inc -> that destination's peer fabric_tx sem_ready
//   await_ready    : local wait sem_ready >= num_routes (every destination feeding this core has started), reset 0
//   send_block     : per packet, one scatter write of up to tiles_per_packet tiles straight into
//                    the destination's L1-interleaved output tiles (no receiver kernel). Packets are
//                    consumed in the tx reader's batches (up to tx_batch packets, one whole batch
//                    region of cb_tx_stage + one cb_tx_meta page naming the batch's packets): issue
//                    every packet, one flush, one pop per batch. Destination addresses come from
//                    addr_table, filled once before the first send (Refinement 3), ALREADY ENCODED for
//                    the receiving router's NoC (Perf 1).
//   announce_done  : per route, fabric atomic inc with flush -> peer sem_done (after the last data packet)
//   await_done     : local wait sem_done >= num_routes (every peer's data has landed here), reset 0
//
// The peers that send to this chip through this core's destinations are, by the symmetric route plan,
// exactly num_routes cores (the same (direction, link) core on each source chip); each sends one ready
// and one done per call, so every count is num_routes and the op_design.md ordering argument holds per
// peer. Resets happen right after each wait.
//
// HELPER BYPASS (measured; see changelog.md Perf 1 "Helper bypasses"): the connection lifecycle is the
// helper's DirectConn policy (dataflow_kernel_lib::ccl), but headers are armed and packets issued with the
// tt_fabric linear set_state / with_state / adapter calls instead of FabricStreamSender::open /
// FabricStream::arm_* / ScatterWriteChannel::write_scatter, for two capability gaps:
//   1. FabricStream binds ONE route at open(route) and every arm_* reuses it; open() completes the
//      connection, so one connection cannot serve two destinations (needed for the link-1 overflow heads;
//      measured 82.3 -> 79.0 us per call in isolation, x = 3).
//   2. write_scatter runs populate_unicast_scatter_write_fields on every packet, re-encoding each chunk
//      address for the EDM-local NoC: 208 of ~370 cycles per packet on the send path. There is no entry
//      point for pre-encoded addresses; here the table holds them and a send copies 2 * tiles_per_packet
//      words + chunk count + payload size (the same header bytes). Measured 82.6 (helper) / 81.0 (raw mirror
//      of the helper) / 79.8 (pre-encoded) us per call, output bit-identical.
//
// Packet headers: a fabric send writes the payload, then the header, from L1 with non-blocking NoC
// writes, so a header must not be re-programmed until the previous packet that used it has left L1.
// Packets rotate over a per-route ring of independently armed scatter headers; a ring is reused only
// after a flush (per batch, or mid-batch when the ring runs out). Re-arming one header per packet without
// a flush sent packet i's payload with packet i+1's addresses (Refinement 2 finding).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
namespace lf = tt::tt_fabric::linear::experimental;
using tt::tt_fabric::common::experimental::UnicastAtomicIncUpdateMask;
using tt::tt_fabric::common::experimental::UnicastScatterWriteUpdateMask;
using tt::tt_fabric::common::experimental::UnicastWriteUpdateMask;
using Hdr = volatile tt_l1_ptr PACKET_HEADER_TYPE*;
constexpr uint32_t MAX_ROUTES = 2;
constexpr uint32_t MAX_RING = 16;
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_tx_stage = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t mt_block = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t ring_size = get_compile_time_arg_val(6);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(7);
    constexpr uint32_t tx_batch = get_compile_time_arg_val(8);
    constexpr uint32_t cb_tx_addrs = get_compile_time_arg_val(9);
    constexpr uint32_t cb_tx_meta = get_compile_time_arg_val(10);
    constexpr auto out_args = TensorAccessorArgs<11>();

    // tiles_per_packet is clamped on the host to the scatter cap (and the fabric payload); restated
    // here only as a guard, never as a second source of the packet size.
    static_assert(tiles_per_packet >= 1 && tiles_per_packet <= NOC_SCATTER_WRITE_MAX_CHUNKS);

    constexpr uint32_t num_mt_blocks = (Mt + mt_block - 1) / mt_block;
    constexpr uint32_t tiles_per_source_head = ring_size * Mt * Nt;
    constexpr uint32_t packet_bytes = tiles_per_packet * tile_bytes;
    constexpr uint32_t batch_pages = tx_batch * tiles_per_packet;
    constexpr uint32_t headers_per_risc = NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType;
    constexpr uint32_t scatter_chunks = tiles_per_packet > 1 ? tiles_per_packet : 2;

    // Connection block first (build_ccl_fabric_rt_args on the host), then:
    //   num_routes, {dst mesh id, dst chip id, peer noc x, peer noc y} x num_routes,
    //   sem_ready, sem_done, out_addr, my_pos, num_heads, (h_local | route << 16) per head
    size_t arg_idx = 0;
    dataflow_kernel_lib::ccl::DirectConn conn(arg_idx, /*is_forward=*/true);  // starts the open handshake

    const uint32_t num_routes = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t route_args = arg_idx;
    arg_idx += 4 * num_routes;
    const uint32_t sem_ready_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_done_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_pos = get_arg_val<uint32_t>(arg_idx++);  // source index j on the destination
    const uint32_t num_heads = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t heads_base = arg_idx;
    ASSERT(num_routes >= 1 && num_routes <= MAX_ROUTES);

    ccl_routing_utils::line_unicast_route_info_t route[MAX_ROUTES]{};
    uint64_t peer_ready_noc[MAX_ROUTES], peer_done_noc[MAX_ROUTES];
    for (uint32_t d = 0; d < num_routes; ++d) {
        route[d].dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(route_args + 4 * d + 0));
        route[d].dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(route_args + 4 * d + 1));
        const uint32_t px = get_arg_val<uint32_t>(route_args + 4 * d + 2);
        const uint32_t py = get_arg_val<uint32_t>(route_args + 4 * d + 3);
        // Fabric NoC addresses are noc0-encoded.
        peer_ready_noc[d] = safe_get_noc_addr(px, py, sem_ready_addr, 0);
        peer_done_noc[d] = safe_get_noc_addr(px, py, sem_done_addr, 0);
    }

    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    volatile tt_l1_ptr uint32_t* sem_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready_addr);
    volatile tt_l1_ptr uint32_t* sem_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_done_addr);

    conn.open();
    auto* snd = conn.sender();

    // Header budget per RISC: inc + unicast tail per route, the rest split evenly into per-route scatter
    // rings (capped at tx_batch). One route = 1 + 1 + min(tx_batch, pool - 2).
    Hdr inc_hdr[MAX_ROUTES];
    Hdr uni_hdr[MAX_ROUTES];
    Hdr ring[MAX_ROUTES][MAX_RING];
    uint32_t ring_len = (headers_per_risc - 2 * num_routes) / num_routes;
    ring_len = ring_len < tx_batch ? ring_len : tx_batch;
    ring_len = ring_len < MAX_RING ? ring_len : MAX_RING;
    ASSERT(ring_len >= 1);

    // inc headers, armed once (= FabricStream::arm_inc(1): Val|Flush)
    for (uint32_t d = 0; d < num_routes; ++d) {
        inc_hdr[d] = PacketHeaderPool::allocate_header();
        lf::fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            inc_hdr[d],
            static_cast<uint8_t>(route[d].distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});
        ccl_routing_utils::fabric_set_line_unicast_route(inc_hdr[d], route[d]);
    }
    auto send_inc = [&](uint32_t d, uint64_t sem_noc_addr) {
        lf::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            snd, inc_hdr[d], tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 0});
    };

    // Packets arrive in the tx reader's order (strict or ready, see the tx reader); each batch's meta
    // page names its packets as (mt block, head, packet) codes, which index addr_table: entry
    // (mb, s, pp) sits at mb * num_heads * full_block_packets + s * packets_per_block(mb) + pp (only the
    // last mt block can be ragged, so every earlier block has full_block_packets per head).
    constexpr uint32_t full_block_packets = (mt_block * Nt + tiles_per_packet - 1) / tiles_per_packet;
    constexpr uint32_t last_block_tiles = (Mt - (num_mt_blocks - 1) * mt_block) * Nt;
    constexpr uint32_t packets_per_head =
        (num_mt_blocks - 1) * full_block_packets + (last_block_tiles + tiles_per_packet - 1) / tiles_per_packet;
    const uint32_t stream_packets = num_heads * packets_per_head;

    // Destination address table: every packet's tile NoC addresses (tiles_per_packet entries per packet),
    // in table order (mt block, head, packet), filled once right after announce_ready. Scatter packets
    // (>= 2 tiles) hold the address ENCODED for the receiving router's NoC (exactly what
    // populate_unicast_scatter_write_fields would compute per packet), zero-padded to tiles_per_packet;
    // a single-tile tail holds the plain address (its unicast with_state call encodes it itself). The
    // destination's output tensor has the same address and the same source index my_pos on every route.
    tt_l1_ptr uint64_t* addr_table = reinterpret_cast<tt_l1_ptr uint64_t*>(get_write_ptr(cb_tx_addrs));
    uint32_t filled = 0;                   // packets whose entries are written, in table order
    uint32_t f_mb = 0, f_s = 0, f_pp = 0;  // table position of entry `filled`
    auto fill_next = [&]() {
        const uint32_t block_tiles = f_mb + 1 == num_mt_blocks ? last_block_tiles : mt_block * Nt;
        const uint32_t h_local = get_arg_val<uint32_t>(heads_base + f_s) & 0xFFFF;
        const uint32_t tile0 =
            h_local * tiles_per_source_head + (my_pos * Mt + f_mb * mt_block) * Nt + f_pp * tiles_per_packet;
        const uint32_t chunks = (block_tiles - f_pp * tiles_per_packet) < tiles_per_packet
                                    ? (block_tiles - f_pp * tiles_per_packet)
                                    : tiles_per_packet;
        tt_l1_ptr uint64_t* e = addr_table + filled * tiles_per_packet;
        for (uint32_t c = 0; c < tiles_per_packet; ++c) {
            uint64_t v = 0;
            if (c < chunks) {
                v = tt::tt_fabric::linear::addrgen_detail::get_noc_address(out, tile0 + c, 0);
                if (chunks > 1) {
                    const auto comps = get_noc_address_components(v);
                    v = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
                }
            }
            e[c] = v;
        }
        ++filled;
        if (++f_pp * tiles_per_packet >= block_tiles) {
            f_pp = 0;
            if (++f_s == num_heads) {
                f_s = 0;
                ++f_mb;
            }
        }
    };

    // announce_ready (one per destination) / await_ready (one per peer)
    for (uint32_t d = 0; d < num_routes; ++d) {
        send_inc(d, peer_ready_noc[d]);
    }
    {
        MaybeDeviceZoneScope("tx_writer_addr_table");
        while (filled < stream_packets) {
            fill_next();  // the whole address table, before the first batch can arrive
        }
    }
    {
        MaybeDeviceZoneScope("tx_writer_await_ready");
        noc_semaphore_wait_min(sem_ready, num_routes);
        noc_semaphore_set(sem_ready, 0);
    }

    // per route: single-tile tail header (= FabricStream::arm_unicast_write(tile_bytes)) and the scatter
    // header ring (= arm_scatter_write(tile_bytes, tiles_per_packet): send type, route, chunk sizes, chunk
    // count, payload size; addresses zero)
    for (uint32_t d = 0; d < num_routes; ++d) {
        uni_hdr[d] = PacketHeaderPool::allocate_header();
        lf::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            uni_hdr[d],
            static_cast<uint8_t>(route[d].distance_in_hops),
            nullptr,
            static_cast<uint16_t>(align(tile_bytes, l1_alignment)));
        ccl_routing_utils::fabric_set_line_unicast_route(uni_hdr[d], route[d]);
        for (uint32_t i = 0; i < ring_len; ++i) {
            ring[d][i] = PacketHeaderPool::allocate_header();
            uint64_t zero_addrs[NOC_SCATTER_WRITE_MAX_CHUNKS] = {0, 0, 0, 0};
            uint16_t chunk_sizes[NOC_SCATTER_WRITE_MAX_CHUNKS - 1] = {tile_bytes, tile_bytes, tile_bytes};
            lf::fabric_unicast_noc_scatter_write_set_state<
                UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                UnicastScatterWriteUpdateMask::PayloadSize>(
                ring[d][i],
                static_cast<uint8_t>(route[d].distance_in_hops),
                tt::tt_fabric::NocUnicastScatterCommandHeader(
                    zero_addrs, chunk_sizes, static_cast<uint8_t>(scatter_chunks)),
                static_cast<uint16_t>(tile_bytes * scatter_chunks));
            ccl_routing_utils::fabric_set_line_unicast_route(ring[d][i], route[d]);
        }
    }

    // One scatter packet (= ScatterWriteChannel::write_scatter with pre-encoded addresses): per-packet
    // header fields, wait for an EDM slot, payload then header NoC writes into the slot + credit.
    auto send_scatter = [&](Hdr hdr, const tt_l1_ptr uint64_t* enc_addrs, uint32_t chunks, uint32_t src_l1) {
        volatile tt_l1_ptr uint32_t* hw =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&hdr->command_fields.unicast_scatter_write.noc_address[0]);
        const tt_l1_ptr uint32_t* tw = reinterpret_cast<const tt_l1_ptr uint32_t*>(enc_addrs);
        for (uint32_t w = 0; w < 2 * tiles_per_packet; ++w) {
            hw[w] = tw[w];
        }
        hdr->command_fields.unicast_scatter_write.chunk_count = static_cast<uint8_t>(chunks);
        hdr->payload_size_bytes = static_cast<uint16_t>(tile_bytes * chunks);
        snd->wait_for_empty_write_slot();
        snd->send_payload_without_header_non_blocking_from_address(src_l1, tile_bytes * chunks);
        snd->send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
    };

    // No per-batch zone (batches can be one packet: the marker budget would run out silently).
    {
        MaybeDeviceZoneScope("tx_writer_stream");  // waits on the tx reader + fabric sends + flushes
        for (uint32_t sent = 0; sent < stream_packets;) {
            cb_wait_front(cb_tx_meta, 1);
            cb_wait_front(cb_tx_stage, batch_pages);
            const uint32_t base = get_read_ptr(cb_tx_stage);
            volatile tt_l1_ptr uint32_t* meta =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_tx_meta));
            const uint32_t k = meta[0];
            // headers used since the last flush, per route (a header is reused only after a flush)
            uint32_t ring_used[MAX_ROUTES] = {0, 0};
            bool uni_used[MAX_ROUTES] = {false, false};
            auto flush_all = [&]() {
                noc_async_writes_flushed();
                for (uint32_t d = 0; d < MAX_ROUTES; ++d) {
                    ring_used[d] = 0;
                    uni_used[d] = false;
                }
            };
            for (uint32_t i = 0; i < k; ++i) {
                const uint32_t code = meta[1 + i];
                const uint32_t mb = code >> 16;
                const uint32_t s = (code >> 8) & 0xFF;
                const uint32_t pp = code & 0xFF;
                const uint32_t d = get_arg_val<uint32_t>(heads_base + s) >> 16;
                const uint32_t block_tiles = mb + 1 == num_mt_blocks ? last_block_tiles : mt_block * Nt;
                const uint32_t packets_per_block = (block_tiles + tiles_per_packet - 1) / tiles_per_packet;
                const uint32_t p = pp * tiles_per_packet;
                const uint32_t chunks = (block_tiles - p) < tiles_per_packet ? (block_tiles - p) : tiles_per_packet;
                const uint32_t l1 = base + i * packet_bytes;
                const uint32_t entry = mb * num_heads * full_block_packets + s * packets_per_block + pp;
                const tt_l1_ptr uint64_t* dst_addrs = addr_table + entry * tiles_per_packet;
                if (chunks == 1) {
                    if (uni_used[d]) {
                        flush_all();  // the unicast header is still being read by the previous tail packet
                    }
#ifndef A2A_ABL_NO_SEND
                    lf::fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                        snd, uni_hdr[d], l1, tt::tt_fabric::NocUnicastCommandHeader{dst_addrs[0]});
#endif
                    uni_used[d] = true;
                } else {
                    if (ring_used[d] == ring_len) {
                        flush_all();  // the route's ring wraps inside the batch: its headers must have left L1
                    }
#ifndef A2A_ABL_NO_SEND
                    send_scatter(ring[d][ring_used[d]], dst_addrs, chunks, l1);
#endif
                    ++ring_used[d];
                }
            }
            sent += k;
            noc_async_writes_flushed();  // every payload of the batch has left its L1 slot before the reader reuses it
            cb_pop_front(cb_tx_stage, batch_pages);
            cb_pop_front(cb_tx_meta, 1);
        }
    }

    // announce_done (Flush armed: lands after every data packet of the route), then close (= FabricStream::
    // close: drain writes + atomics, then the connection)
    {
        MaybeDeviceZoneScope("tx_writer_close");
        for (uint32_t d = 0; d < num_routes; ++d) {
            send_inc(d, peer_done_noc[d]);
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        conn.close();
    }

    // await_done
    MaybeDeviceZoneScope("tx_writer_await_done");
    noc_semaphore_wait_min(sem_done, num_routes);
    noc_semaphore_set(sem_done, 0);
}
