// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 fabric_tx writer, MULTI-ROUTE variant (perf experiment tx_rebalance_direct).
//
// Same as kernels/a2a_wkv_b2_tx_writer.cpp, except that one tx core / one DIRECT router connection
// serves up to MAX_ROUTES destinations: per destination its own route (dst chip id), its own armed
// packet headers (inc, unicast tail, a slice of the scatter ring) and its own peer handshake. Each head
// of the stream names its destination (route index); the address table is unchanged (same output
// tensor address and the same source index `my_pos` on every destination).
//
// Used to move the last x heads of the neighbour stream (link 0) onto the link-1 tx core, which
// otherwise carries only the antipode half: per worker-channel load 120/60 -> (120-5x)/(60+5x) packets,
// with no fabric mux core and no extra kernel.
//
// Handshakes: the tx core sends one ready and one done per destination (to that destination's peer
// tx core) and waits for sem_ready / sem_done >= num_routes. By the symmetric plan, the cores that send
// to this chip through this core's peers are exactly num_routes cores (the same (direction, link) core
// on each destination), so every count is num_routes; the op_design.md ordering argument holds per peer.
//
// HELPER BYPASS (dataflow_kernel_lib::ccl::FabricStreamSender / FabricStream): capability gap, not
// ergonomics. FabricStream binds ONE route at open(route) and every arm_* reuses it; open() finishes the
// connection, so a sender can yield only one stream, and FabricStream's ctor is private. Two destinations
// over one connection therefore cannot be expressed with the helper. This kernel uses the helper's public
// DirectConn policy (connection lifecycle) and issues the SAME linear_fabric set_state / with_state calls
// the helper's channels issue (ccl_helpers_dataflow.inl), one header set per route.

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

    static_assert(tiles_per_packet >= 1 && tiles_per_packet <= NOC_SCATTER_WRITE_MAX_CHUNKS);

    constexpr uint32_t num_mt_blocks = (Mt + mt_block - 1) / mt_block;
    constexpr uint32_t tiles_per_source_head = ring_size * Mt * Nt;
    constexpr uint32_t packet_bytes = tiles_per_packet * tile_bytes;
    constexpr uint32_t batch_pages = tx_batch * tiles_per_packet;
    constexpr uint32_t headers_per_risc = NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType;
    constexpr uint32_t scatter_chunks = tiles_per_packet > 1 ? tiles_per_packet : 2;

    size_t arg_idx = 0;
    dataflow_kernel_lib::ccl::DirectConn conn(arg_idx, /*is_forward=*/true);

    const uint32_t num_routes = get_arg_val<uint32_t>(arg_idx++);
    ccl_routing_utils::line_unicast_route_info_t route[MAX_ROUTES]{};
    uint64_t peer_ready_noc[MAX_ROUTES], peer_done_noc[MAX_ROUTES];
    const uint32_t route_args = arg_idx;
    arg_idx += 4 * num_routes;
    const uint32_t sem_ready_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_done_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_pos = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_heads = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t heads_base = arg_idx;  // per head: h_local | (route << 16)
    for (uint32_t d = 0; d < num_routes; ++d) {
        route[d].dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(route_args + 4 * d + 0));
        route[d].dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(route_args + 4 * d + 1));
        const uint32_t px = get_arg_val<uint32_t>(route_args + 4 * d + 2);
        const uint32_t py = get_arg_val<uint32_t>(route_args + 4 * d + 3);
        peer_ready_noc[d] = safe_get_noc_addr(px, py, sem_ready_addr, 0);
        peer_done_noc[d] = safe_get_noc_addr(px, py, sem_done_addr, 0);
    }

    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    volatile tt_l1_ptr uint32_t* sem_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready_addr);
    volatile tt_l1_ptr uint32_t* sem_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_done_addr);

    conn.open();
    auto* snd = conn.sender();

    // Header budget per RISC: inc + unicast tail per route, the rest split evenly into per-route scatter
    // rings (capped at tx_batch). One route = the baseline's 1 + 1 + min(tx_batch, pool - 2).
    Hdr inc_hdr[MAX_ROUTES];
    Hdr uni_hdr[MAX_ROUTES];
    Hdr ring[MAX_ROUTES][MAX_RING];
    uint32_t ring_len = (headers_per_risc - 2 * num_routes) / num_routes;
    ring_len = ring_len < tx_batch ? ring_len : tx_batch;
    ring_len = ring_len < MAX_RING ? ring_len : MAX_RING;
    for (uint32_t d = 0; d < num_routes; ++d) {
        inc_hdr[d] = PacketHeaderPool::allocate_header();
        lf::fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            inc_hdr[d],
            static_cast<uint8_t>(route[d].distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});
        ccl_routing_utils::fabric_set_line_unicast_route(inc_hdr[d], route[d]);
    }
    auto inc = [&](uint32_t d, uint64_t sem_noc) {
        lf::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            snd, inc_hdr[d], tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc, 0});
    };

    constexpr uint32_t full_block_packets = (mt_block * Nt + tiles_per_packet - 1) / tiles_per_packet;
    constexpr uint32_t last_block_tiles = (Mt - (num_mt_blocks - 1) * mt_block) * Nt;
    constexpr uint32_t packets_per_head =
        (num_mt_blocks - 1) * full_block_packets + (last_block_tiles + tiles_per_packet - 1) / tiles_per_packet;
    const uint32_t stream_packets = num_heads * packets_per_head;
    // Per-route packets left: with A2A_TXR_EARLY_DONE a route's done is sent as soon as its last packet
    // is issued (same channel, so it still lands after that route's data), instead of after the stream.
    uint32_t route_left[MAX_ROUTES] = {0, 0};
    for (uint32_t s = 0; s < num_heads; ++s) {
        route_left[get_arg_val<uint32_t>(heads_base + s) >> 16] += packets_per_head;
    }

    tt_l1_ptr uint64_t* addr_table = reinterpret_cast<tt_l1_ptr uint64_t*>(get_write_ptr(cb_tx_addrs));
    uint32_t filled = 0;
    uint32_t f_mb = 0, f_s = 0, f_pp = 0;
    auto fill_next = [&]() {
        const uint32_t block_tiles = f_mb + 1 == num_mt_blocks ? last_block_tiles : mt_block * Nt;
        const uint32_t h_local = get_arg_val<uint32_t>(heads_base + f_s) & 0xFFFF;
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

    // announce_ready (one per destination) / await_ready (one per peer)
    for (uint32_t d = 0; d < num_routes; ++d) {
        inc(d, peer_ready_noc[d]);
    }
    {
        MaybeDeviceZoneScope("tx_writer_addr_table");
        while (filled < stream_packets) {
            fill_next();
        }
    }
    {
        MaybeDeviceZoneScope("tx_writer_await_ready");
        noc_semaphore_wait_min(sem_ready, num_routes);
        noc_semaphore_set(sem_ready, 0);
    }

    uint16_t chunk_sizes[3] = {
        static_cast<uint16_t>(tile_bytes), static_cast<uint16_t>(tile_bytes), static_cast<uint16_t>(tile_bytes)};
    for (uint32_t d = 0; d < num_routes; ++d) {
        uni_hdr[d] = PacketHeaderPool::allocate_header();
        lf::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            uni_hdr[d],
            static_cast<uint8_t>(route[d].distance_in_hops),
            nullptr,
            static_cast<uint16_t>(align(tile_bytes, l1_alignment)));
        ccl_routing_utils::fabric_set_line_unicast_route(uni_hdr[d], route[d]);
        for (uint32_t i = 0; i < ring_len; ++i) {
            Hdr h = PacketHeaderPool::allocate_header();
            uint64_t dummy[4] = {0, 0, 0, 0};
            lf::fabric_unicast_noc_scatter_write_set_state<
                UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
                h,
                static_cast<uint8_t>(route[d].distance_in_hops),
                tt::tt_fabric::NocUnicastScatterCommandHeader(dummy, chunk_sizes, static_cast<uint8_t>(scatter_chunks)),
                static_cast<uint16_t>(tile_bytes * scatter_chunks));
            ccl_routing_utils::fabric_set_line_unicast_route(h, route[d]);
            ring[d][i] = h;
        }
    }

    {
        MaybeDeviceZoneScope("tx_writer_stream");
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
                const uint64_t* dst_addrs = addr_table + entry * tiles_per_packet;
                if (chunks == 1) {
                    if (uni_used[d]) {
                        flush_all();
                    }
#ifndef A2A_ABL_NO_SEND
                    lf::fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                        snd, uni_hdr[d], l1, tt::tt_fabric::NocUnicastCommandHeader{dst_addrs[0]});
#endif
                    uni_used[d] = true;
                } else {
                    if (ring_used[d] == ring_len) {
                        flush_all();
                    }
#ifndef A2A_ABL_NO_SEND
                    lf::fabric_unicast_noc_scatter_write_with_state<
                        UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                        UnicastScatterWriteUpdateMask::PayloadSize>(
                        snd,
                        ring[d][ring_used[d]],
                        l1,
                        tt::tt_fabric::NocUnicastScatterCommandHeader(
                            dst_addrs, chunk_sizes, static_cast<uint8_t>(chunks)),
                        static_cast<uint16_t>(tile_bytes * chunks));
#endif
                    ++ring_used[d];
                }
#ifdef A2A_TXR_EARLY_DONE
                if (--route_left[d] == 0) {
                    inc(d, peer_done_noc[d]);  // Flush armed: lands after this route's data packets
                }
#endif
            }
            sent += k;
            noc_async_writes_flushed();  // every payload of the batch has left its L1 slot
            cb_pop_front(cb_tx_stage, batch_pages);
            cb_pop_front(cb_tx_meta, 1);
        }
    }

    {
        MaybeDeviceZoneScope("tx_writer_close");
#ifndef A2A_TXR_EARLY_DONE
        for (uint32_t d = 0; d < num_routes; ++d) {
            inc(d, peer_done_noc[d]);  // Flush armed: lands after this route's data packets
        }
#endif
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        conn.close();
    }

    MaybeDeviceZoneScope("tx_writer_await_done");
    noc_semaphore_wait_min(sem_done, num_routes);
    noc_semaphore_set(sem_done, 0);
}
