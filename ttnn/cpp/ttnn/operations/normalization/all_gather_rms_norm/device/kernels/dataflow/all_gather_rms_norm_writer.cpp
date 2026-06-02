// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer dataflow kernel for the generic fused all_gather_rms_norm op.
//
// ring_size == 1: just write the normalized output tiles from cb_out back to interleaved DRAM.
// ring_size  > 1: BATCHED all-gather. The compute kernel runs a chunked two-pass scheme; for each chunk
//   of up to `gather_chunk` rows the writer:
//     1. waits for the chunk's per-device stat partials (cb_local_stats),
//     2. fabric line-multicasts every partial to the matching slot on every ring peer (and copies its own
//        into place), issuing ALL the chunk's writes back-to-back,
//     3. signals completion with a SINGLE atomic-inc per peer (so one out-ready barrier covers the whole
//        chunk instead of one barrier per row — this is the batching that removes the per-row fabric stall),
//     4. releases cb_gathered_stats so compute can finish the chunk (pass 2), then drains the chunk's
//        output rows from cb_out back to DRAM.
//
// Fabric mechanics ported from ccl/broadcast/device/kernels/broadcast_tile_writer.cpp (RoutingPlane line
// multicast + set_state/with_state), which pairs with the host-side
// append_routing_plane_connection_manager_rt_args used by the program factory.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#if defined(RING_GT_1)
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
using namespace tt::tt_fabric::linear::experimental;
#endif

void kernel_main() {
    // Compile-time args (order must match all_gather_rms_norm_program_factory.cpp: writer_ct_args).
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t ring_size = get_compile_time_arg_val(2);
    constexpr uint32_t cb_local_stats = get_compile_time_arg_val(3);
    constexpr uint32_t cb_gathered_stats = get_compile_time_arg_val(4);
    constexpr uint32_t cb_packet_header = get_compile_time_arg_val(5);
    constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(6);
    constexpr uint32_t num_links = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t ring_index = get_compile_time_arg_val(9);
    constexpr uint32_t num_targets_forward = get_compile_time_arg_val(10);
    constexpr uint32_t num_targets_backward = get_compile_time_arg_val(11);
    constexpr uint32_t start_hops_forward = get_compile_time_arg_val(12);
    constexpr uint32_t range_hops_forward = get_compile_time_arg_val(13);
    constexpr uint32_t start_hops_backward = get_compile_time_arg_val(14);
    constexpr uint32_t range_hops_backward = get_compile_time_arg_val(15);
    constexpr uint32_t gather_chunk = get_compile_time_arg_val(16);  // rows per batched fabric all-gather
    constexpr auto dst_args = TensorAccessorArgs<17>();

    (void)cb_local_stats;
    (void)cb_gathered_stats;
    (void)cb_packet_header;
    (void)num_packet_headers_storable;
    (void)num_links;
    (void)ring_index;
    (void)num_targets_forward;
    (void)num_targets_backward;
    (void)start_hops_forward;
    (void)range_hops_forward;
    (void)start_hops_backward;
    (void)range_hops_backward;
    (void)gather_chunk;

    // Runtime args.
    uint32_t ai = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(ai++);
    const uint32_t tile_offset = get_arg_val<uint32_t>(ai++);

    const auto s = TensorAccessor(dst_args, dst_addr);
    const uint32_t tile_bytes = get_tile_size(cb_out);

#if defined(RING_GT_1)
    // ----- fabric stats all-gather (ring_size > 1) -----
    const uint32_t out_ready_sem_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t barrier_sem_addr = get_arg_val<uint32_t>(ai++);  // unused in the per-row scheme
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(ai++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(ai++);
    const uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(ai++);  // = ring_size (one inc per peer/chunk)
    const uint32_t num_connections = get_arg_val<uint32_t>(ai++);
    const uint32_t stats_ready_sem_id = get_arg_val<uint32_t>(ai++);
    (void)barrier_sem_addr;
    (void)stats_ready_sem_id;
    size_t arg_for_fab = ai;

    const uint32_t stat_tile_bytes = get_tile_size(cb_local_stats);
    const uint32_t NCHt = num_tiles / Wt;

    auto unicast_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    auto sem_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_for_fab);

    uint8_t starts[] = {static_cast<uint8_t>(start_hops_forward), static_cast<uint8_t>(start_hops_backward)};
    uint8_t ranges[] = {static_cast<uint8_t>(range_hops_forward), static_cast<uint8_t>(range_hops_backward)};
    if (ranges[0] == 0) {
        starts[0] = starts[1];
        ranges[0] = ranges[1];
    }
    fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        fabric_connection, unicast_route_id, starts, ranges, nullptr, stat_tile_bytes);
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection, sem_route_id, starts, ranges, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});

    volatile tt_l1_ptr uint32_t* out_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_addr);
    const uint64_t out_ready_sem_noc_addr_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_addr, 0);
    const uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_addr);

    uint32_t out_tile_id = tile_offset;
    for (uint32_t chunk_start = 0; chunk_start < NCHt; chunk_start += gather_chunk) {
        const uint32_t rows = (NCHt - chunk_start) < gather_chunk ? (NCHt - chunk_start) : gather_chunk;

        // ----- batched fabric all-gather of the chunk's partials -----
        cb_wait_front(cb_local_stats, rows);
        cb_reserve_back(cb_gathered_stats, ring_size * rows);
        const uint32_t local_base = get_read_ptr(cb_local_stats);
        const uint32_t gathered_base = get_write_ptr(cb_gathered_stats);

        // For each row: copy our partial into our own slot and fabric-multicast it to the SAME slot on
        // every ring peer. Slot layout per row is [ring_size] consecutive tiles (peer r at index r), so the
        // compute kernel can SUM-reduce row(ring_size) directly. Issue all writes back-to-back.
        for (uint32_t r = 0; r < rows; r++) {
            const uint32_t my_partial_addr = local_base + r * stat_tile_bytes;
            const uint32_t my_slot_addr = gathered_base + (r * ring_size + ring_index) * stat_tile_bytes;
            const uint64_t my_slot_noc_addr = get_noc_addr(my_x[noc_index], my_y[noc_index], my_slot_addr);
            noc_async_write(my_partial_addr, my_slot_noc_addr, stat_tile_bytes);
            fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                fabric_connection,
                unicast_route_id,
                my_partial_addr,
                tt::tt_fabric::NocUnicastCommandHeader{my_slot_noc_addr});
        }
        noc_async_writes_flushed();

        // Single completion signal for the whole chunk: one atomic-inc to every ring peer (line multicast)
        // plus a local inc. Fabric preserves per-connection ordering, so the inc lands after all the chunk's
        // writes on each receiver. Each device therefore observes exactly ring_size incs once its peers are
        // done -> one barrier per chunk instead of one per row.
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection,
            sem_route_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_pkt, 0});
        noc_semaphore_inc(out_ready_sem_noc_addr, 1);

        cb_pop_front(cb_local_stats, rows);

        noc_semaphore_wait(out_ready_sem_ptr, out_ready_sem_wait_value);
        noc_semaphore_set(out_ready_sem_ptr, 0);

        cb_push_back(cb_gathered_stats, ring_size * rows);

        // ----- drain this chunk's output rows back to DRAM (compute pass 2 produced them) -----
        // cb_out is block-sized and filled/drained in `blk`-tile blocks, so wait/write/pop per block.
        for (uint32_t r = 0; r < rows; r++) {
            for (uint32_t wt = 0; wt < Wt; wt += blk) {
                cb_wait_front(cb_out, blk);
                uint32_t write_offset = 0;
                for (uint32_t j = 0; j < blk; j++) {
                    noc_async_write_tile(out_tile_id, s, get_read_ptr(cb_out) + write_offset);
                    out_tile_id++;
                    write_offset += tile_bytes;
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, blk);
            }
        }
    }
    close_connections(fabric_connection);
    noc_async_write_barrier();
#else
    // ----- single device (ring_size == 1): just write output -----
    uint32_t tile_id = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        cb_wait_front(cb_out, blk);
        uint32_t write_offset = 0;
        for (uint32_t j = 0; j < blk; j++) {
            noc_async_write_tile(tile_id, s, get_read_ptr(cb_out) + write_offset);
            tile_id++;
            write_offset += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, blk);
    }
#endif
}
