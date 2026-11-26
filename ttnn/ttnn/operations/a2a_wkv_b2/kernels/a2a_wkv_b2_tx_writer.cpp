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
//                    region of cb_tx_stage): issue every packet, one flush, one pop per batch.
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
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp"

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
    constexpr auto out_args = TensorAccessorArgs<10>();

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

    // Fabric connection block first (built on the host by build_ccl_fabric_rt_args).
    size_t arg_idx = 0;
    FabricStreamSender<DirectConn> sender(arg_idx, /*is_forward=*/true, l1_alignment);

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

    auto stream = sender.open(route);
    auto inc_ch = stream.arm_inc(1);  // Val|Flush armed

    // announce_ready / await_ready
    inc_ch.inc(peer_ready_noc);

    // Destination address table (Refinement 3): every packet's tile NoC addresses, computed here while
    // the first block is still being produced, so the send loop does no TensorAccessor work. Entry
    // (stream packet index, chunk); the stream's packet order is the send loop's below.
    tt_l1_ptr uint64_t* addr_table = reinterpret_cast<tt_l1_ptr uint64_t*>(get_write_ptr(cb_tx_addrs));
    {
        uint32_t e = 0;
        for (uint32_t mb = 0; mb < num_mt_blocks; ++mb) {
            const uint32_t mt0 = mb * mt_block;
            const uint32_t mt_rows = (Mt - mt0) < mt_block ? (Mt - mt0) : mt_block;
            const uint32_t block_tiles = mt_rows * Nt;
            for (uint32_t s = 0; s < num_heads; ++s) {
                const uint32_t h_local = get_arg_val<uint32_t>(heads_base + s);
                const uint32_t tile_base = h_local * tiles_per_source_head + (my_pos * Mt + mt0) * Nt;
                for (uint32_t p = 0; p < block_tiles; p += tiles_per_packet) {
                    for (uint32_t c = 0; c < tiles_per_packet; ++c, ++e) {
                        addr_table[e] = p + c < block_tiles ? tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                                                  out, tile_base + p + c, 0)
                                                            : 0;
                    }
                }
            }
        }
    }
    noc_semaphore_wait_min(sem_ready, 1);
    noc_semaphore_set(sem_ready, 0);

    auto unicast_ch = stream.arm_unicast_write(tile_bytes);  // single-tile tail packets
    [[maybe_unused]] auto scatter_ring = arm_scatter_ring(
        stream, tile_bytes, tiles_per_packet > 1 ? tiles_per_packet : 2, std::make_index_sequence<hdr_ring>{});

    uint32_t pkt = 0;  // stream packet index into addr_table
    for (uint32_t mb = 0; mb < num_mt_blocks; ++mb) {
        const uint32_t mt0 = mb * mt_block;
        const uint32_t mt_rows = (Mt - mt0) < mt_block ? (Mt - mt0) : mt_block;
        const uint32_t block_tiles = mt_rows * Nt;
        const uint32_t packets_per_block = (block_tiles + tiles_per_packet - 1) / tiles_per_packet;
        const uint32_t round_packets = num_heads * packets_per_block;
        for (uint32_t q0 = 0; q0 < round_packets; q0 += tx_batch) {
            const uint32_t k = (round_packets - q0) < tx_batch ? (round_packets - q0) : tx_batch;
            {
                DeviceZoneScopedN("TX_WAIT");
                cb_wait_front(cb_tx_stage, batch_pages);
            }
            DeviceZoneScopedN("TX_SEND");
            const uint32_t base = get_read_ptr(cb_tx_stage);
            bool unicast_used = false;
            for (uint32_t i = 0; i < k; ++i) {
                const uint32_t q = q0 + i;
                const uint32_t p = (q % packets_per_block) * tiles_per_packet;
                const uint32_t chunks = (block_tiles - p) < tiles_per_packet ? (block_tiles - p) : tiles_per_packet;
                const uint32_t l1 = base + i * packet_bytes;
                const uint64_t* dst_addrs = addr_table + (pkt++) * tiles_per_packet;
                if (chunks == 1) {
                    if (unicast_used) {
                        stream.flush();  // the unicast header is still being read by the previous tail packet
                    }
                    unicast_ch.write(dst_addrs[0], l1);
                    unicast_used = true;
                } else {
                    const uint32_t r = i % hdr_ring;
                    if (r == 0 && i != 0) {
                        stream.flush();  // ring wraps inside the batch: its headers must have left L1
                    }
                    scatter_ring[r].write_scatter(dst_addrs, chunks, l1);
                }
            }
            stream.flush();  // every payload of the batch has left its L1 slot before the reader reuses it
            cb_pop_front(cb_tx_stage, batch_pages);
        }
    }

    // announce_done (flush: lands after every data packet of this stream)
    inc_ch.inc(peer_done_noc);
    stream.close();

    // await_done
    noc_semaphore_wait_min(sem_done, 1);
    noc_semaphore_set(sem_done, 0);
}
