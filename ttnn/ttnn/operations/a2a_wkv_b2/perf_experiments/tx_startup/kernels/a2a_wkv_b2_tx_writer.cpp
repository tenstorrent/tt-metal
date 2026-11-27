// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 fabric_tx writer (BRISC) — send_block + handshakes for one stream.
//
//   announce_ready : fabric atomic inc -> peer fabric_tx sem_ready on the destination
//   await_ready    : local wait sem_ready >= 1, reset 0 (destination has started this program)
//   send_block     : per packet, one scatter write of up to tiles_per_packet tiles straight into
//                    the destination's L1-interleaved output tiles (no receiver kernel). Packets are
//                    consumed in the tx reader's batches (up to tx_batch packets, one whole batch
//                    region of cb_tx_stage + one cb_tx_meta page naming the batch's packets): issue
//                    every packet, one flush, one pop per batch. Destination NoC addresses come from
//                    addr_table, filled once before the first send (Refinement 3).
//   announce_done  : fabric atomic inc with flush -> peer sem_done (after the last data packet)
//   await_done     : local wait sem_done >= 1 (the peer's data has landed here), reset 0
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

#include <array>
#include <cstdint>
#include <utility>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "txs_addr_fill.hpp"

namespace {
template <typename Stream, size_t... I>
FORCE_INLINE auto arm_scatter_ring(
    Stream& stream, uint32_t chunk_bytes, uint32_t max_chunks, std::index_sequence<I...>) {
    using Channel = decltype(stream.arm_scatter_write(chunk_bytes, max_chunks));
    return std::array<Channel, sizeof...(I)>{((void)I, stream.arm_scatter_write(chunk_bytes, max_chunks))...};
}
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
    // Header pool: inc + unicast take one header each, the rest (up to tx_batch) form the scatter ring.
    constexpr uint32_t headers_per_risc = NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType;
    constexpr uint32_t other_headers = 2;
    static_assert(headers_per_risc > other_headers, "no packet header left for the scatter ring");
    constexpr uint32_t hdr_ring =
        tx_batch < headers_per_risc - other_headers ? tx_batch : headers_per_risc - other_headers;

    using namespace dataflow_kernel_lib::ccl;

    // Connection block first. Direct (one stream per (direction, link) router): built on the host by
    // build_ccl_fabric_rt_args. Mux (R6, Refinement 3: several streams share one link): this stream is
    // one client channel of the link's fabric mux core (ttnn.experimental.fabric_mux client args, after
    // the output accessor's CT args); the stream code below is identical for both ConnT policies.
    size_t arg_idx = 0;
#ifdef A2A_TX_MUX
    constexpr uint32_t mux_ct = out_args.next_compile_time_args_offset();
    constexpr uint8_t mux_num_buffers = get_compile_time_arg_val(mux_ct);
    constexpr size_t mux_buffer_bytes = get_compile_time_arg_val(mux_ct + 1);
    constexpr size_t mux_status_address = get_compile_time_arg_val(mux_ct + 2);
    constexpr size_t mux_termination_address = get_compile_time_arg_val(mux_ct + 3);
    constexpr uint32_t mux_num_clients = get_compile_time_arg_val(mux_ct + 4);
    using ConnT = MuxConn<mux_num_buffers>;
    FabricStreamSender<ConnT> sender(
        ConnT(arg_idx, mux_buffer_bytes, mux_status_address, mux_termination_address, mux_num_clients), l1_alignment);
#else
    FabricStreamSender<DirectConn> sender(arg_idx, /*is_forward=*/true, l1_alignment);
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

    auto stream = [&]() {
        MaybeDeviceZoneScope("tx_writer_open");
        return sender.open(route);
    }();
    auto inc_ch = stream.arm_inc(1);  // Val|Flush armed

    // Packets arrive in the tx reader's order (strict or ready, see the tx reader); each batch's meta
    // page names its packets as (mt block, head, packet) codes, which index addr_table: entry
    // (mb, s, pp) sits at mb * num_heads * full_block_packets + s * packets_per_block(mb) + pp (only the
    // last mt block can be ragged, so every earlier block has full_block_packets per head).
    constexpr uint32_t full_block_packets = (mt_block * Nt + tiles_per_packet - 1) / tiles_per_packet;
    constexpr uint32_t last_block_tiles = (Mt - (num_mt_blocks - 1) * mt_block) * Nt;
    constexpr uint32_t packets_per_head =
        (num_mt_blocks - 1) * full_block_packets + (last_block_tiles + tiles_per_packet - 1) / tiles_per_packet;
    const uint32_t stream_packets = num_heads * packets_per_head;

    // Destination address table (Refinement 3): every packet's tile NoC addresses, in table order
    // (mt block, head, packet), filled once right after announce_ready (while the destination starts and
    // the first blocks are produced), so the send loop does no TensorAccessor work.
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
        for (uint32_t c = 0; c < chunks; ++c) {
            e[c] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(out, tile0 + c, 0);
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
    inc_ch.inc(peer_ready_noc);

#if TXS_EARLY_ARM
    // Arm the data channels (header programming only, nothing is sent). tx_startup: TXS_EARLY_ARM arms them
    // right after announce_ready, before the address table, instead of after await_ready.
    auto txs_arms = [&]() {
        MaybeDeviceZoneScope("tx_writer_arm");
        return std::make_pair(
            stream.arm_unicast_write(tile_bytes),  // single-tile tail packets
            arm_scatter_ring(
                stream, tile_bytes, tiles_per_packet > 1 ? tiles_per_packet : 2, std::make_index_sequence<hdr_ring>{}));
    }();
    auto& unicast_ch = txs_arms.first;
    [[maybe_unused]] auto& scatter_ring = txs_arms.second;
#endif

#if TXS_SPLIT || TXS_WALK
    // tx_startup: head-granular fill. The tx reader (split) fills heads [0, reader_heads) at its start, this
    // writer the rest after announce_ready, then waits for the reader's CB_TX_SYNC page.
    (void)fill_next;
    {
        MaybeDeviceZoneScope("tx_writer_addr_table");
        const uint32_t reader_heads = get_arg_val<uint32_t>(heads_base + num_heads);
        for (uint32_t s = reader_heads; s < num_heads; ++s) {
            const uint32_t h_local = get_arg_val<uint32_t>(heads_base + s);
            txs::fill_head<Mt, Nt, mt_block, tiles_per_packet, TXS_WALK != 0>(
                addr_table,
                out,
                out_addr,
                tile_bytes,
                s,
                num_heads,
                h_local * tiles_per_source_head + my_pos * Mt * Nt);
        }
        filled = stream_packets;
    }
#if TXS_SPLIT
    {
        MaybeDeviceZoneScope("tx_writer_table_sync");
        cb_wait_front(TXS_CB_SYNC, 1);
    }
#endif
#else
    {
        MaybeDeviceZoneScope("tx_writer_addr_table");
        while (filled < stream_packets) {
            fill_next();  // the whole address table, before the first batch can arrive
        }
    }
#endif
    {
        MaybeDeviceZoneScope("tx_writer_await_ready");
        noc_semaphore_wait_min(sem_ready, 1);
        noc_semaphore_set(sem_ready, 0);
    }

#if !TXS_EARLY_ARM
    // Arm the data channels (header programming only, nothing is sent). tx_startup: TXS_EARLY_ARM arms them
    // right after announce_ready, before the address table, instead of after await_ready.
    auto txs_arms = [&]() {
        MaybeDeviceZoneScope("tx_writer_arm");
        return std::make_pair(
            stream.arm_unicast_write(tile_bytes),  // single-tile tail packets
            arm_scatter_ring(
                stream, tile_bytes, tiles_per_packet > 1 ? tiles_per_packet : 2, std::make_index_sequence<hdr_ring>{}));
    }();
    auto& unicast_ch = txs_arms.first;
    [[maybe_unused]] auto& scatter_ring = txs_arms.second;
#endif

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
                const uint64_t* dst_addrs = addr_table + entry * tiles_per_packet;
                if (chunks == 1) {
                    if (unicast_used) {
                        stream.flush();  // the unicast header is still being read by the previous tail packet
                    }
#ifndef A2A_ABL_NO_SEND
                    unicast_ch.write(dst_addrs[0], l1);
#endif
                    unicast_used = true;
                } else {
                    const uint32_t r = i % hdr_ring;
                    if (r == 0 && i != 0) {
                        stream.flush();  // ring wraps inside the batch: its headers must have left L1
                    }
#ifndef A2A_ABL_NO_SEND
                    scatter_ring[r].write_scatter(dst_addrs, chunks, l1);
#endif
                }
            }
            sent += k;
            stream.flush();  // every payload of the batch has left its L1 slot before the reader reuses it
            cb_pop_front(cb_tx_stage, batch_pages);
            cb_pop_front(cb_tx_meta, 1);
        }
    }

    // announce_done (flush: lands after every data packet of this stream)
    {
        MaybeDeviceZoneScope("tx_writer_close");
        inc_ch.inc(peer_done_noc);
        stream.close();
    }

    // await_done
    MaybeDeviceZoneScope("tx_writer_await_done");
    noc_semaphore_wait_min(sem_done, 1);
    noc_semaphore_set(sem_done, 0);
}
