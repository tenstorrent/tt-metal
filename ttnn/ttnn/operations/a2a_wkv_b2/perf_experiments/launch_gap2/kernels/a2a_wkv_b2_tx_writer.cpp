// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 fabric_tx writer (BRISC) — send_block + handshakes for one stream.
// Perf 1 (idea tx_send_rate, variant "prehdr"): destination addresses pre-encoded in the address table.
//
//   announce_ready : fabric atomic inc -> peer fabric_tx sem_ready on the destination
//   await_ready    : local wait sem_ready >= 1, reset 0 (destination has started this program)
//   send_block     : per packet, one scatter write of up to tiles_per_packet tiles straight into
//                    the destination's L1-interleaved output tiles (no receiver kernel). Packets are
//                    consumed in the tx reader's batches (up to tx_batch packets, one whole batch
//                    region of cb_tx_stage + one cb_tx_meta page naming the batch's packets): issue
//                    every packet, one flush, one pop per batch. Destination addresses come from
//                    addr_table, filled once before the first send (Refinement 3), ALREADY ENCODED for
//                    the receiving router's NoC (see "Raw fabric send" below).
//   announce_done  : fabric atomic inc with flush -> peer sem_done (after the last data packet)
//   await_done     : local wait sem_done >= 1 (the peer's data has landed here), reset 0
//
// Raw fabric send (helper bypass, measured; changelog.md Perf 1 "Helper bypasses"): the connection is the helper's
// policy (dataflow_kernel_lib::ccl::DirectConn, or MuxConn<N> under A2A_TX_MUX), but headers are armed and packets
// issued with the tt_fabric linear set_state / adapter calls instead of FabricStream::arm_* /
// ScatterWriteChannel::write_scatter. write_scatter runs populate_unicast_scatter_write_fields on every
// packet, which re-encodes each chunk address for the EDM-local NoC (get_noc_address_components +
// safe_get_noc_addr) and re-stores the chunk sizes: 208 of the ~370 cycles of a packet's send (TXW_DIAG
// accumulators, tx path alone). Here the table holds the encoded addresses (computed once, off the send
// loop), and a send copies 2 * tiles_per_packet words + the chunk count + payload size into the header
// (~120 cycles) — the same header bytes write_scatter produces. The helper has no entry point for
// pre-encoded addresses. Measured (trace replay, us per call): helper 82.6, raw mirror of the helper 81.0,
// this 79.8; output bit-identical.
//
// Packet headers: a fabric send writes the payload, then the header, from L1 with non-blocking NoC
// writes, so a header must not be re-programmed until the previous packet that used it has left L1.
// Packets of a batch therefore rotate over a ring of independently armed scatter headers
// (hdr_ring, derived from the per-RISC header pool); the ring is only reused after a flush (per batch,
// or mid-batch when a batch is longer than the ring). Re-arming one header per packet without a flush
// sent packet i's payload with packet i+1's addresses (Refinement 2 finding).
//
// The peer is the destination's stream that targets this chip; it sends exactly one ready and one
// done per call, so every count is 1. Resets happen right after each wait (see op_design.md
// "Synchronization contract" for why the next call always sees 0).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace lf = tt::tt_fabric::linear::experimental;
using tt::tt_fabric::common::experimental::UnicastAtomicIncUpdateMask;
using tt::tt_fabric::common::experimental::UnicastScatterWriteUpdateMask;
using tt::tt_fabric::common::experimental::UnicastWriteUpdateMask;

void kernel_main() {
#ifdef A2A_TX_COMMON
    // launch_gap2: no per-core runtime args. This core's args sit in the kernel's common table:
    //   [grid_x, first logical core index, counters_base, offset per tx core slot..., per-core arg blocks...]
    // get_arg_val is pointed at this core's block, so the rest of the kernel (and the fabric connection
    // helper, which consumes its block through get_arg_val) is unchanged.
    {
        const uint32_t slot = get_relative_logical_y() * get_common_arg_val<uint32_t>(0) + get_relative_logical_x() -
                              get_common_arg_val<uint32_t>(1);
        const uint32_t off = get_common_arg_val<uint32_t>(3 + slot);
        rta_l1_base = crta_l1_base + off;
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT)
        rta_count = crta_count - off;
#endif
    }
#endif
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
    // Header pool: inc + unicast take one header each, the rest (up to tx_batch) form the scatter ring.
    constexpr uint32_t headers_per_risc = NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType;
    constexpr uint32_t other_headers = 2;
    static_assert(headers_per_risc > other_headers, "no packet header left for the scatter ring");
    constexpr uint32_t hdr_ring =
        tx_batch < headers_per_risc - other_headers ? tx_batch : headers_per_risc - other_headers;

    // Connection block first. Direct (one stream per (direction, link) router): built on the host by
    // build_ccl_fabric_rt_args. Mux (R6, Refinement 3): this stream is one client channel of the link's
    // fabric mux core (ttnn.experimental.fabric_mux client args, after the output accessor's CT args).
    size_t arg_idx = 0;
#ifdef A2A_TX_MUX
    constexpr uint32_t mux_ct = out_args.next_compile_time_args_offset();
    constexpr uint8_t mux_num_buffers = get_compile_time_arg_val(mux_ct);
    constexpr size_t mux_buffer_bytes = get_compile_time_arg_val(mux_ct + 1);
    constexpr size_t mux_status_address = get_compile_time_arg_val(mux_ct + 2);
    constexpr size_t mux_termination_address = get_compile_time_arg_val(mux_ct + 3);
    constexpr uint32_t mux_num_clients = get_compile_time_arg_val(mux_ct + 4);
    dataflow_kernel_lib::ccl::MuxConn<mux_num_buffers> conn(
        arg_idx, mux_buffer_bytes, mux_status_address, mux_termination_address, mux_num_clients);
#else
    dataflow_kernel_lib::ccl::DirectConn conn(arg_idx, /*is_forward=*/true);  // starts the open handshake
#endif

    const uint32_t dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t peer_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t peer_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_ready_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_done_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_pos = get_arg_val<uint32_t>(arg_idx++);  // source index j on the destination
    const uint32_t num_heads = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t heads_base = arg_idx;  // h_local (on the destination) per head

    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    ccl_routing_utils::line_unicast_route_info_t route{};
    route.dst_mesh_id = static_cast<uint16_t>(dst_mesh_id);
    route.dst_chip_id = static_cast<uint16_t>(dst_chip_id);

    // Fabric NoC addresses are noc0-encoded.
    const uint64_t peer_ready_noc = safe_get_noc_addr(peer_noc_x, peer_noc_y, sem_ready_addr, 0);
    const uint64_t peer_done_noc = safe_get_noc_addr(peer_noc_x, peer_noc_y, sem_done_addr, 0);
    volatile tt_l1_ptr uint32_t* sem_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready_addr);
    volatile tt_l1_ptr uint32_t* sem_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_done_addr);

    conn.open();
    auto* snd = conn.sender();

    // inc header, armed once (= FabricStream::arm_inc(1): Val|Flush)
    auto* inc_hdr = PacketHeaderPool::allocate_header();
    lf::fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        inc_hdr, static_cast<uint8_t>(route.distance_in_hops), tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});
    ccl_routing_utils::fabric_set_line_unicast_route(inc_hdr, route);
    auto send_inc = [&](uint64_t sem_noc_addr) {
        lf::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            snd, inc_hdr, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 0});
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
    // a single-tile tail holds the plain address (its unicast with_state call encodes it itself).
    tt_l1_ptr uint64_t* addr_table = reinterpret_cast<tt_l1_ptr uint64_t*>(get_write_ptr(cb_tx_addrs));
    uint32_t filled = 0;                   // packets whose entries are written, in table order
    uint32_t f_mb = 0, f_s = 0, f_pp = 0;  // table position of entry `filled`
    auto fill_next = [&]() {
        const uint32_t block_tiles = f_mb + 1 == num_mt_blocks ? last_block_tiles : mt_block * Nt;
        const uint32_t h_local = get_arg_val<uint32_t>(heads_base + f_s);
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

    // announce_ready / await_ready
    send_inc(peer_ready_noc);

    {
        MaybeDeviceZoneScope("tx_writer_addr_table");
        while (filled < stream_packets) {
            fill_next();  // the whole address table, before the first batch can arrive
        }
    }
    {
        MaybeDeviceZoneScope("tx_writer_await_ready");
        noc_semaphore_wait_min(sem_ready, 1);
        noc_semaphore_set(sem_ready, 0);
    }

    // single-tile tail header (= FabricStream::arm_unicast_write(tile_bytes))
    auto* uni_hdr = PacketHeaderPool::allocate_header();
    lf::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        uni_hdr,
        static_cast<uint8_t>(route.distance_in_hops),
        nullptr,
        static_cast<uint16_t>(align(tile_bytes, l1_alignment)));
    ccl_routing_utils::fabric_set_line_unicast_route(uni_hdr, route);

    // scatter header ring (= FabricStream::arm_scatter_write(tile_bytes, tiles_per_packet) x hdr_ring:
    // send type, route, chunk sizes, chunk count, payload size; addresses zero)
    constexpr uint32_t arm_chunks = tiles_per_packet > 1 ? tiles_per_packet : 2;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* scatter_ring[hdr_ring];
    for (uint32_t r = 0; r < hdr_ring; ++r) {
        scatter_ring[r] = PacketHeaderPool::allocate_header();
        uint64_t zero_addrs[NOC_SCATTER_WRITE_MAX_CHUNKS] = {0, 0, 0, 0};
        uint16_t chunk_sizes[NOC_SCATTER_WRITE_MAX_CHUNKS - 1] = {tile_bytes, tile_bytes, tile_bytes};
        lf::fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
            UnicastScatterWriteUpdateMask::PayloadSize>(
            scatter_ring[r],
            static_cast<uint8_t>(route.distance_in_hops),
            tt::tt_fabric::NocUnicastScatterCommandHeader(zero_addrs, chunk_sizes, static_cast<uint8_t>(arm_chunks)),
            static_cast<uint16_t>(tile_bytes * arm_chunks));
        ccl_routing_utils::fabric_set_line_unicast_route(scatter_ring[r], route);
    }

    // One scatter packet (= ScatterWriteChannel::write_scatter with pre-encoded addresses): per-packet
    // header fields, wait for an EDM slot, payload then header NoC writes into the slot + credit.
    auto send_scatter = [&](volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr,
                            const tt_l1_ptr uint64_t* enc_addrs,
                            uint32_t chunks,
                            uint32_t src_l1) {
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
            bool unicast_used = false;
            for (uint32_t i = 0; i < k; ++i) {
                const uint32_t code = meta[1 + i];
                const uint32_t mb = code >> 16;
                const uint32_t s = (code >> 8) & 0xFF;
                const uint32_t pp = code & 0xFF;
                const uint32_t block_tiles = mb + 1 == num_mt_blocks ? last_block_tiles : mt_block * Nt;
                const uint32_t packets_per_block = (block_tiles + tiles_per_packet - 1) / tiles_per_packet;
                const uint32_t p = pp * tiles_per_packet;
                const uint32_t chunks = (block_tiles - p) < tiles_per_packet ? (block_tiles - p) : tiles_per_packet;
                const uint32_t l1 = base + i * packet_bytes;
                const uint32_t entry = mb * num_heads * full_block_packets + s * packets_per_block + pp;
                const tt_l1_ptr uint64_t* dst_addrs = addr_table + entry * tiles_per_packet;
                if (chunks == 1) {
                    if (unicast_used) {
                        noc_async_writes_flushed();  // the unicast header is still being read by the previous tail
                    }
#ifndef A2A_ABL_NO_SEND
                    lf::fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                        snd, uni_hdr, l1, tt::tt_fabric::NocUnicastCommandHeader{dst_addrs[0]});
#endif
                    unicast_used = true;
                } else {
                    const uint32_t r = i % hdr_ring;
                    if (r == 0 && i != 0) {
                        noc_async_writes_flushed();  // ring wraps inside the batch: its headers must have left L1
                    }
#ifndef A2A_ABL_NO_SEND
                    send_scatter(scatter_ring[r], dst_addrs, chunks, l1);
#endif
                }
            }
            sent += k;
            noc_async_writes_flushed();  // every payload of the batch has left its L1 slot before the reader reuses it
            cb_pop_front(cb_tx_stage, batch_pages);
            cb_pop_front(cb_tx_meta, 1);
        }
    }

    // announce_done (flush: lands after every data packet of this stream), then close (= FabricStream::close:
    // drain writes + atomics, then the connection)
    {
        MaybeDeviceZoneScope("tx_writer_close");
        send_inc(peer_done_noc);
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        conn.close();
    }

    // await_done
    MaybeDeviceZoneScope("tx_writer_await_done");
    noc_semaphore_wait_min(sem_done, 1);
    noc_semaphore_set(sem_done, 0);
}
