// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// vsa_ring_sdpa's K/V ring all-gather, sender writer (forked from all_gather_async's minimal_default_writer).
//
// Two inputs (K and V, each [1, H, T_local, d], TILE, interleaved DRAM) into two persistent outputs
// ([1, H, T_local * ring_size, d]); Ring topology only; multi-worker per link behind a fabric MUX (USE_WORKER_MUX) or
// one worker per link on a direct fabric connection. Every worker owns the tile ROWS [row_start, row_end) of a slice
// and walks them TOKEN-MAJOR: for each tile row, K of head 0..H-1 (DHt tiles each), then V of head 0..H-1 -- so the
// receiving VSA leaders see every head's blocks land progressively and can gate per block (RingGate in
// vsa_sdpa_stream_reader.cpp; its tile index within the slice is row * 2*H*DHt + (tensor*H + head) * DHt + col).
// After every `chunks_per_sync` packets the receiver's out_ready_sem is bumped (Flush-ordered behind the data).
// No split forwarding (fused mode only) and the local slice is NOT written into the local gathered buffer (the fused
// consumer reads its own shard from the local tensors).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "api/tensor/noc_traits.h"

using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t ring_size = get_compile_time_arg_val(0);
constexpr uint32_t my_chip_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(6);
constexpr uint32_t n_heads = get_compile_time_arg_val(7);
constexpr uint32_t dht = get_compile_time_arg_val(8);        // d / 32
constexpr uint32_t ht_local = get_compile_time_arg_val(9);   // T_local / 32
constexpr uint32_t ht_total = get_compile_time_arg_val(10);  // ring_size * ht_local
constexpr bool fuse_op = get_compile_time_arg_val(11);
#ifdef USE_WORKER_MUX
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(12);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(13);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(14);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(15);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(16);
constexpr uint32_t rt_arg_count = 17;
#else
constexpr uint32_t rt_arg_count = 12;
#endif

constexpr ccl_routing_utils::line_unicast_route_info_t forward_unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<rt_arg_count>();
constexpr ccl_routing_utils::line_unicast_route_info_t backward_unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<rt_arg_count + ccl_routing_utils::num_line_unicast_args>();
constexpr uint32_t accessor_args_base = rt_arg_count + 2 * ccl_routing_utils::num_line_unicast_args;
constexpr auto gk_args = TensorAccessorArgs<accessor_args_base>();
constexpr auto gv_args = TensorAccessorArgs<gk_args.next_compile_time_args_offset()>();

constexpr uint32_t tiles_per_row = 2 * n_heads * dht;  // the token-major sequence: K heads, then V heads

namespace detail {
bool valid_targets(const bool direction) {
    if constexpr (num_targets_backward_direction + num_targets_forward_direction == 0) {
        return false;
    } else {
        return (direction == 0 && num_targets_forward_direction) || (direction == 1 && num_targets_backward_direction);
    }
}
}  // namespace detail

// Position in the token-major sequence of a slice: tile row r (within the slice), tensor t (0 = K, 1 = V), head h,
// column tile c. Advances without divisions.
struct SeqCursor {
    uint32_t r = 0, t = 0, h = 0, c = 0;
    void advance() {
        if (++c == dht) {
            c = 0;
            if (++h == n_heads) {
                h = 0;
                if (++t == 2) {
                    t = 0;
                    ++r;
                }
            }
        }
    }
    // page id of this position in a gathered tensor, slice of ring position `chip`
    uint32_t gathered_page(uint32_t chip) const { return h * ht_total * dht + (chip * ht_local + r) * dht + c; }
};

void kernel_main() {
    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    const address_t gk_address = get_arg_val<address_t>(arg_idx++);
    const address_t gv_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);  // 0: sends to the forward neighbour, 1: backward
    const uint32_t row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t row_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
#ifdef USE_WORKER_MUX
    const bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool is_termination_master = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t termination_sync_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t termination_sync_address = get_semaphore(termination_sync_id);
    const uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    const uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
#endif
    const auto& unicast_route_info = (direction == 0) ? forward_unicast_route_info : backward_unicast_route_info;

    const auto gk = TensorAccessor(gk_args, gk_address);
    const auto gv = TensorAccessor(gv_args, gv_address);

#ifdef USE_WORKER_MUX
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* fabric_direction_connection;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_connection;
    if (mux_connection_valid) {
        mux_connection = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
            fabric_mux_x,
            fabric_mux_y,
            fabric_mux_channel_id,
            fabric_mux_num_buffers_per_channel,
            fabric_mux_channel_buffer_size_bytes,
            fabric_mux_channel_base_address,
            fabric_mux_connection_info_address,
            fabric_mux_connection_handshake_address,
            fabric_mux_flow_control_address,
            fabric_mux_buffer_index_address,
            local_flow_control_address,
            local_teardown_address,
            local_buffer_index_address);
        fabric_direction_connection = &mux_connection;
        // the mux must be ready before anyone connects
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);
    } else {
        fabric_direction_connection = nullptr;
    }
#else
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
#endif
    Noc noc_obj;
    CircularBuffer cb_output(cb_output_id);

    // fused-op signaling (the local slice's availability is signaled by the direction-1 writer)
    OpSignaler op_signaler_sender;
    uint32_t self_write_done_semaphore_addr = 0;
    if constexpr (fuse_op) {
#ifndef USE_WORKER_MUX
        arg_idx = arg_for_fab;
#endif
        self_write_done_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        op_signaler_sender = OpSignaler(arg_idx);
    }

#ifdef USE_WORKER_MUX
    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_connect(*fabric_direction_connection);
    }
#else
    fabric_connection.open();
    auto* fabric_direction_connection =
        direction ? &fabric_connection.get_backward_connection() : &fabric_connection.get_forward_connection();
#endif

    // pre-populated packet headers: scatter (2..4 tiles), unicast (1 tile), semaphore increment
    auto pkt_scatter_hdr = PacketHeaderPool::allocate_header();
    auto pkt_unicast_hdr = PacketHeaderPool::allocate_header();
    auto pkt_hdr_sem_inc = PacketHeaderPool::allocate_header();
    if (detail::valid_targets(direction)) {
        constexpr uint32_t scatter_header_chunk_count = num_tiles_to_write_per_packet < NOC_SCATTER_WRITE_MIN_CHUNKS
                                                            ? NOC_SCATTER_WRITE_MIN_CHUNKS
                                                            : num_tiles_to_write_per_packet;
        static_assert(
            scatter_header_chunk_count <= NOC_SCATTER_WRITE_MAX_CHUNKS, "tiles per packet > 4 is unsupported");
        uint64_t dummy_addrs[4] = {0, 0, 0, 0};
        uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
        fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            pkt_scatter_hdr,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            NocUnicastScatterCommandHeader(dummy_addrs, chunk_sizes, scatter_header_chunk_count),
            page_size * scatter_header_chunk_count);
        fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, page_size);
        fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            pkt_hdr_sem_inc,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, static_cast<uint32_t>(1)});
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc, unicast_route_info);
    }
    const uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
    const bool sends = detail::valid_targets(direction);

    // Send one slice (ring position `chip`) to the next hop: the reader hands the tiles over in the same
    // token-major order, packets of num_tiles_to_write_per_packet.
    const auto send_slice = [&](uint32_t chip) {
        SeqCursor cur;
        cur.r = row_start;
        const uint32_t total = (row_end - row_start) * tiles_per_row;
        uint32_t chunk_count = 0;
        for (uint32_t sent = 0; sent < total;) {
            const uint32_t n = std::min(total - sent, num_tiles_to_write_per_packet);
            cb_output.wait_front(num_tiles_to_write_per_packet);
            const size_t l1_read_addr = cb_output.get_read_ptr();
            uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
            uint64_t noc_addrs[4] = {0, 0, 0, 0};
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t page = cur.gathered_page(chip);
                noc_addrs[i] = cur.t == 0 ? tt::tt_fabric::linear::addrgen_detail::get_noc_address(gk, page, 0)
                                          : tt::tt_fabric::linear::addrgen_detail::get_noc_address(gv, page, 0);
                cur.advance();
            }
            if (sends) {
                if (n > 1) {
                    fabric_unicast_noc_scatter_write_with_state<
                        UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                        UnicastScatterWriteUpdateMask::PayloadSize>(
                        fabric_direction_connection,
                        pkt_scatter_hdr,
                        l1_read_addr,
                        NocUnicastScatterCommandHeader(noc_addrs, chunk_sizes, n),
                        page_size * n);
                } else {
                    fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                        fabric_direction_connection,
                        pkt_unicast_hdr,
                        l1_read_addr,
                        NocUnicastCommandHeader{noc_addrs[0]});
                }
            }
            sent += n;
            noc_obj.async_writes_flushed();
            cb_output.pop_front(num_tiles_to_write_per_packet);
            if (++chunk_count % chunks_per_sync == 0 && sends) {
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    fabric_direction_connection,
                    pkt_hdr_sem_inc,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
            }
            noc_obj.async_writes_flushed();
        }
        if (chunk_count % chunks_per_sync != 0 && sends) {
            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_direction_connection,
                pkt_hdr_sem_inc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
        }
    };

    // 1. the local slice
    send_slice(my_chip_id);
    if constexpr (fuse_op) {
        if (direction == 1) {
            // the local slice is available (the fused consumer reads it from the local tensors)
            op_signaler_sender.synchronize_workers_and_signal_op(my_chip_id);
            const uint64_t self_write_done_semaphore_noc_addr =
                safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, self_write_done_semaphore_addr, 0);
            noc_semaphore_inc(self_write_done_semaphore_noc_addr, 1);
        }
    }

    // 2. forward the slices the reader receives from the other side (all but the last one it receives)
    const uint32_t writes_expected =
        (direction == 1 ? num_targets_backward_direction : num_targets_forward_direction) - 1;
    for (uint32_t slice_writes = 0; slice_writes < writes_expected; ++slice_writes) {
        uint32_t chip;
        if (direction == 1) {
            chip = my_chip_id + slice_writes + 1;
            chip = chip >= ring_size ? chip - ring_size : chip;
        } else {
            chip = my_chip_id + ring_size - slice_writes - 1;
            chip = chip >= ring_size ? chip - ring_size : chip;
        }
        send_slice(chip);
    }

    noc_obj.async_write_barrier();
    noc_obj.async_atomic_barrier();
#ifdef USE_WORKER_MUX
    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(*fabric_direction_connection);
        if (is_termination_master) {
            Semaphore<> termination_sync(termination_sync_id);
            termination_sync.wait(num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
        } else {
            const uint64_t dest_addr =
                safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
            noc_semaphore_inc(dest_addr, 1);
            noc_obj.async_atomic_barrier();
        }
    }
#else
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }
#endif
    noc_obj.async_write_barrier();
}
