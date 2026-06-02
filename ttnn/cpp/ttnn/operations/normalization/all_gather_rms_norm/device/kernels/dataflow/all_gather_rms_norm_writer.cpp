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
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "mux_common.hpp"  // worker-side fabric-mux connection helpers (same dir)
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
#if defined(RING_GT_1)
    // Fabric-mux compile-time args (these slots are always present in the arg list; the single-device path
    // simply never reads them, which is why the TensorAccessor offset below is fixed at 22 either way).
    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(17);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(18);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(19);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(20);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(21);
#endif
    // Head-split output addressing (always present). The normalized (1, B, N, F) row tiles are scattered
    // into the (1, num_heads, M, head_dim) output: width-tile w of global tile-row gr -> head h = w /
    // head_dim_tiles, within-head tile e = w % head_dim_tiles, output page = h*(m_tiles*head_dim_tiles) +
    // gr*head_dim_tiles + e. With head_dim_tiles == Wt (num_heads == 1) this is the contiguous gr*Wt + w.
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(22);
    constexpr uint32_t m_tiles = get_compile_time_arg_val(23);
    constexpr auto dst_args = TensorAccessorArgs<24>();
    constexpr uint32_t per_head_stride = m_tiles * head_dim_tiles;

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
    // ----- batched fabric stats all-gather through the fabric mux (ring_size > 1) -----
    const uint32_t out_ready_sem_addr = get_arg_val<uint32_t>(ai++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(ai++);  // THIS core's own virtual coords
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(ai++);
    const uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(ai++);  // = ring_size

    // Connect to the fabric mux (forward then backward). build_and_connect blocks until the mux is ready.
    auto mux_fwd =
        parse_mux_connection_args<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes>(ai);
    auto mux_bwd =
        parse_mux_connection_args<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes>(ai);
    auto* sender_fwd = mux_fwd.build_and_connect(fabric_mux_status_address);
    auto* sender_bwd = mux_bwd.build_and_connect(fabric_mux_status_address);

    const uint32_t stat_tile_bytes = get_tile_size(cb_local_stats);
    const uint32_t NCHt = num_tiles / Wt;

    // One write header + one atomic-inc header per direction; each carries a line-multicast route so a
    // single send reaches every ring peer in that direction (the mux just forwards the packet to the EDM).
    auto* hdr_w_fwd = PacketHeaderPool::allocate_header();
    auto* hdr_w_bwd = PacketHeaderPool::allocate_header();
    auto* hdr_s_fwd = PacketHeaderPool::allocate_header();
    auto* hdr_s_bwd = PacketHeaderPool::allocate_header();
    if (sender_fwd != nullptr) {
        fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            hdr_w_fwd,
            static_cast<uint8_t>(start_hops_forward),
            static_cast<uint8_t>(range_hops_forward),
            nullptr,
            stat_tile_bytes);
        fabric_multicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            hdr_s_fwd,
            static_cast<uint8_t>(start_hops_forward),
            static_cast<uint8_t>(range_hops_forward),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});
    }
    if (sender_bwd != nullptr) {
        fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            hdr_w_bwd,
            static_cast<uint8_t>(start_hops_backward),
            static_cast<uint8_t>(range_hops_backward),
            nullptr,
            stat_tile_bytes);
        fabric_multicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            hdr_s_bwd,
            static_cast<uint8_t>(start_hops_backward),
            static_cast<uint8_t>(range_hops_backward),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});
    }

    volatile tt_l1_ptr uint32_t* out_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_addr);
    const uint64_t out_ready_sem_noc_addr_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_addr, 0);
    const uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_addr);

    uint32_t gr = tile_offset / Wt;  // global tile-row index of this worker's first row (for head-split)
    for (uint32_t chunk_start = 0; chunk_start < NCHt; chunk_start += gather_chunk) {
        const uint32_t rows = (NCHt - chunk_start) < gather_chunk ? (NCHt - chunk_start) : gather_chunk;

        // ----- batched fabric all-gather of the chunk's partials, through the mux -----
        cb_wait_front(cb_local_stats, rows);
        cb_reserve_back(cb_gathered_stats, ring_size * rows);
        const uint32_t local_base = get_read_ptr(cb_local_stats);
        const uint32_t gathered_base = get_write_ptr(cb_gathered_stats);

        // For each row: copy our partial into our own slot and line-multicast it (fwd + bwd) to the SAME
        // slot on every ring peer. Slot layout per row is [ring_size] consecutive tiles (peer r at index r),
        // so the compute kernel can SUM-reduce row(ring_size) directly. Issue all writes back-to-back.
        for (uint32_t r = 0; r < rows; r++) {
            const uint32_t my_partial_addr = local_base + r * stat_tile_bytes;
            const uint32_t my_slot_addr = gathered_base + (r * ring_size + ring_index) * stat_tile_bytes;
            const uint64_t my_slot_noc_addr = get_noc_addr(my_x[noc_index], my_y[noc_index], my_slot_addr);
            noc_async_write(my_partial_addr, my_slot_noc_addr, stat_tile_bytes);
            if (sender_fwd != nullptr) {
                fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                    sender_fwd, hdr_w_fwd, my_partial_addr, tt::tt_fabric::NocUnicastCommandHeader{my_slot_noc_addr});
            }
            if (sender_bwd != nullptr) {
                fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                    sender_bwd, hdr_w_bwd, my_partial_addr, tt::tt_fabric::NocUnicastCommandHeader{my_slot_noc_addr});
            }
        }
        noc_async_writes_flushed();

        // One completion signal per direction for the whole chunk (line multicast) + a local inc. Each
        // device observes exactly ring_size incs on its own out-ready sem once all peers are done -> one
        // barrier per chunk. Targets THIS core's out-ready sem (peers' same-coord core sends here).
        if (sender_fwd != nullptr) {
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                sender_fwd, hdr_s_fwd, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_pkt, 0});
        }
        if (sender_bwd != nullptr) {
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                sender_bwd, hdr_s_bwd, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_pkt, 0});
        }
        noc_semaphore_inc(out_ready_sem_noc_addr, 1);

        cb_pop_front(cb_local_stats, rows);

        noc_semaphore_wait(out_ready_sem_ptr, out_ready_sem_wait_value);
        noc_semaphore_set(out_ready_sem_ptr, 0);

        cb_push_back(cb_gathered_stats, ring_size * rows);

        // ----- drain this chunk's output rows back to DRAM (compute pass 2 produced them) -----
        // cb_out is block-sized and filled/drained in `blk`-tile blocks, so wait/write/pop per block. Each
        // width-tile is scattered to its head-split position in the (1, num_heads, M, head_dim) output.
        for (uint32_t r = 0; r < rows; r++) {
            for (uint32_t wt = 0; wt < Wt; wt += blk) {
                cb_wait_front(cb_out, blk);
                uint32_t write_offset = 0;
                for (uint32_t j = 0; j < blk; j++) {
                    const uint32_t w = wt + j;
                    const uint32_t h = w / head_dim_tiles;
                    const uint32_t out_id = h * per_head_stride + gr * head_dim_tiles + (w - h * head_dim_tiles);
                    noc_async_write_tile(out_id, s, get_read_ptr(cb_out) + write_offset);
                    write_offset += tile_bytes;
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, blk);
            }
            gr++;
        }
    }

    // Teardown: disconnect from the mux; the elected termination master then signals the mux to stop.
    if (mux_bwd.connection_valid) {
        close_mux(
            sender_bwd,
            mux_bwd.is_termination_master,
            mux_bwd.termination_sync_address,
            num_mux_clients,
            mux_bwd.fabric_mux_x,
            mux_bwd.fabric_mux_y,
            fabric_mux_termination_signal_address,
            mux_bwd.termination_master_noc_x,
            mux_bwd.termination_master_noc_y);
    }
    if (mux_fwd.connection_valid) {
        close_mux(
            sender_fwd,
            mux_fwd.is_termination_master,
            mux_fwd.termination_sync_address,
            num_mux_clients,
            mux_fwd.fabric_mux_x,
            mux_fwd.fabric_mux_y,
            fabric_mux_termination_signal_address,
            mux_fwd.termination_master_noc_x,
            mux_fwd.termination_master_noc_y);
    }
    noc_async_write_barrier();
#else
    // ----- single device (ring_size == 1): write output, scattering each width-tile to its head-split
    // position in the (1, num_heads, M, head_dim) output (blk divides Wt, so a block stays within one row).
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        const uint32_t gl0 = tile_offset + i;  // global linear tile index of this block's first tile
        const uint32_t gr = gl0 / Wt;          // global tile-row
        const uint32_t w0 = gl0 - gr * Wt;     // width-tile of the block start
        cb_wait_front(cb_out, blk);
        uint32_t write_offset = 0;
        for (uint32_t j = 0; j < blk; j++) {
            const uint32_t w = w0 + j;
            const uint32_t h = w / head_dim_tiles;
            const uint32_t out_id = h * per_head_stride + gr * head_dim_tiles + (w - h * head_dim_tiles);
            noc_async_write_tile(out_id, s, get_read_ptr(cb_out) + write_offset);
            write_offset += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, blk);
    }
#endif
}
