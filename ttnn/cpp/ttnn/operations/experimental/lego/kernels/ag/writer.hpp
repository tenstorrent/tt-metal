// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring all-gather writer. Port of all_gather_async/device/kernels/minimal_default_writer.cpp (interleaved output,
// direct fabric connection, no fused-op signaller): start-of-op barrier multicast, send the local slice (scatter
// writes of up to 4 tiles, out_ready increment per chunks_per_sync packets), the dir-1 worker also writes the local
// slice locally, then forward the slices this worker relays (split forwarding on even rings).
//
// CT (A::): ring_size my_chip_id cb_output_id num_tiles_to_write_per_packet page_size
//   num_targets_forward_direction num_targets_backward_direction topology gather_dim input_batch_head_count
//   input_tensor_Wt input_tensor_C output_tensor_Wt output_tensor_C reverse barrier_target_count
//   cta_route (16 words: fwd unicast, fwd barrier mcast, bwd unicast, bwd barrier mcast) cta_output
// Common RT: the 6 AllGatherCommonArgs words
//   use_mux, and when use_mux the MuxClient CT values (fabric/mux_client.hpp)
//   signal_slices: as the reader's hook, for the local slice the dir-1 worker writes
//   write_local: the dir-1 worker writes the local slice into the output (false when a producer already put it there)
// RT: own_x own_y use_barrier_sem opposite_x opposite_y direction global_worker_id global_worker_count
//   chunks_per_sync_or_zero, then the fabric connection block (direct) or the 17-word mux client block (use_mux),
//   then [signal_slices: slice_counters num_targets x0 y0 ...]

#pragma once

#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
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
#include "ttnn/operations/experimental/ccl/all_gather_async/device/kernels/all_gather_worker_split.hpp"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"
#include "ttnn/operations/experimental/lego/kernels/fabric/mux_client.hpp"
#include <type_traits>

namespace lego::ag {

using namespace tt::tt_fabric::linear::experimental;

template <typename A>
struct Writer {
    using address_t = uint32_t;
    using Topology = ttnn::ccl::Topology;
    static constexpr uint32_t ring_size = A::ring_size;
    static constexpr uint32_t my_chip_id = A::my_chip_id;
    static constexpr uint32_t num_tiles_to_write_per_packet = A::num_tiles_to_write_per_packet;
    static constexpr uint32_t page_size = A::page_size;
    static constexpr uint32_t num_targets_forward_direction = A::num_targets_forward_direction;
    static constexpr uint32_t num_targets_backward_direction = A::num_targets_backward_direction;
    static constexpr Topology topology = static_cast<Topology>(A::topology);
    static constexpr uint32_t gather_dim = A::gather_dim;
    static constexpr uint32_t input_batch_head_count = A::input_batch_head_count;
    static constexpr uint32_t input_tensor_Wt = A::input_tensor_Wt;
    static constexpr uint32_t input_tensor_C = A::input_tensor_C;
    static constexpr uint32_t output_tensor_Wt = A::output_tensor_Wt;
    static constexpr uint32_t output_tensor_C = A::output_tensor_C;
    static constexpr bool reverse = A::reverse;
    static constexpr uint32_t barrier_target_count = A::barrier_target_count;

    static constexpr ccl_routing_utils::line_unicast_route_info_t forward_unicast_route_info =
        ccl_routing_utils::get_line_unicast_route_info_from_args<A::cta_route>();
    static constexpr ccl_routing_utils::line_multicast_route_info_t forward_barrier_multicast_route_info =
        ccl_routing_utils::get_line_multicast_route_info_from_args<
            A::cta_route + ccl_routing_utils::num_line_unicast_args>();
    static constexpr ccl_routing_utils::line_unicast_route_info_t backward_unicast_route_info =
        ccl_routing_utils::get_line_unicast_route_info_from_args<
            A::cta_route + ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args>();
    static constexpr ccl_routing_utils::line_multicast_route_info_t backward_barrier_multicast_route_info =
        ccl_routing_utils::get_line_multicast_route_info_from_args<
            A::cta_route + 2 * ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args>();

    static bool valid_targets_forward(const bool direction) {
        if constexpr (num_targets_forward_direction) {
            return (direction == 0);
        } else {
            return false;
        }
    }
    static bool valid_targets_backward(const bool direction) {
        if constexpr (num_targets_backward_direction) {
            return (direction == 1);
        } else {
            return false;
        }
    }
    static bool valid_targets(const bool direction) {
        if constexpr (num_targets_backward_direction + num_targets_forward_direction == 0) {
            return false;
        } else {
            return (valid_targets_forward(direction) || valid_targets_backward(direction));
        }
    }

    address_t output_address;
    uint32_t input_tensor_Ht, output_tensor_Ht;
    uint8_t out_ready_sem_noc0_x, out_ready_sem_noc0_y, opposite_core_sem_noc0_x, opposite_core_sem_noc0_y;
    bool use_barrier_sem, direction;
    size_t barrier_sem, out_ready_sem;
    AllGatherWorkerSplit split;
    FabricConnectionManager fabric_connection;
    struct NoMux {};
    std::conditional_t<A::use_mux, lego::fabric::MuxClient<A>, NoMux> mux;
    uint32_t sig_arg = 0;

    void init() {
        uint32_t arg_idx = 0;
        output_address = crt<A>(ttnn::ccl::AllGatherCommonArgs::output);
        input_tensor_Ht = crt<A>(ttnn::ccl::AllGatherCommonArgs::input_Ht);
        output_tensor_Ht = gather_dim == 2 ? input_tensor_Ht * ring_size : input_tensor_Ht;
        out_ready_sem_noc0_x = rt<A>(arg_idx++);
        out_ready_sem_noc0_y = rt<A>(arg_idx++);
        use_barrier_sem = rt<A>(arg_idx++);
        barrier_sem = crt<A>(ttnn::ccl::AllGatherCommonArgs::barrier);
        opposite_core_sem_noc0_x = rt<A>(arg_idx++);
        opposite_core_sem_noc0_y = rt<A>(arg_idx++);
        direction = rt<A>(arg_idx++);
        out_ready_sem = crt<A>(ttnn::ccl::AllGatherCommonArgs::semaphore_0 + direction);
        const auto global_worker_id = rt<A>(arg_idx++);
        const auto global_worker_count = rt<A>(arg_idx++);
        const auto chunks_per_sync_or_zero = rt<A>(arg_idx++);
        split = all_gather_worker_split(
            global_worker_id,
            global_worker_count,
            input_tensor_Ht,
            input_tensor_Wt,
            output_tensor_Wt,
            num_tiles_to_write_per_packet,
            chunks_per_sync_or_zero);
        if constexpr (A::use_mux) {
            mux.read_args(arg_idx);
            sig_arg = arg_idx + 17;
        } else {
            size_t arg_for_fab = rt_index<A>(arg_idx);
            fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
            sig_arg = arg_for_fab - A::rt_base;
        }
    }

    void run() {
        if constexpr (A::use_mux) {
            auto* conn = mux.open();
            body(conn);
            mux.close();
        } else {
            fabric_connection.open();
            auto* conn =
                direction ? (fabric_connection.has_backward_connection() ? &fabric_connection.get_backward_connection()
                                                                         : nullptr)
                          : (fabric_connection.has_forward_connection() ? &fabric_connection.get_forward_connection()
                                                                        : nullptr);
            body(conn);
            // Close here, not in teardown(): a later block on this RISC may connect to the same fabric channel.
            if (fabric_connection.is_logically_connected()) {
                fabric_connection.close();
            }
        }
        Noc noc_obj;
        noc_obj.async_write_barrier();
    }

    template <typename Conn, typename Hdr>
    static void send_packet(
        Conn* conn,
        Hdr pkt_scatter_hdr,
        Hdr pkt_unicast_hdr,
        size_t l1_read_addr,
        uint64_t* noc_addrs,
        uint16_t* chunk_sizes,
        uint32_t n) {
        if (n > 1) {
            fabric_unicast_noc_scatter_write_with_state<
                UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                UnicastScatterWriteUpdateMask::PayloadSize>(
                conn,
                pkt_scatter_hdr,
                l1_read_addr,
                NocUnicastScatterCommandHeader(noc_addrs, chunk_sizes, n),
                page_size * n);
        } else {
            fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                conn, pkt_unicast_hdr, l1_read_addr, NocUnicastCommandHeader{noc_addrs[0]});
        }
    }

    template <typename Conn>
    void body(Conn* fabric_direction_connection) {
        const uint32_t input_tile_id_start = split.input_tile_id_start;
        const uint32_t input_tile_id_end = split.input_tile_id_end;
        const uint32_t start_pages_read_in_row = split.start_pages_read_in_row;
        const uint32_t start_row_offset = split.start_row_offset;
        const uint32_t chunks_per_sync = split.chunks_per_sync;

        const auto& unicast_route_info = (direction == 0) ? forward_unicast_route_info : backward_unicast_route_info;
        const auto& barrier_multicast_route_info =
            (direction == 0) ? forward_barrier_multicast_route_info : backward_barrier_multicast_route_info;
        constexpr auto output_tensor_args = TensorAccessorArgs<A::cta_output>();
        const auto output_addrgen = TensorAccessor(output_tensor_args, output_address);

        Noc noc_obj;
        CircularBuffer cb_output(A::cb_output_id);

        auto pkt_scatter_hdr = PacketHeaderPool::allocate_header();
        auto pkt_unicast_hdr = PacketHeaderPool::allocate_header();
        auto pkt_hdr_sem_inc = PacketHeaderPool::allocate_header();

        if (use_barrier_sem) {
            if (valid_targets(direction)) {
                ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_sem_inc, barrier_multicast_route_info);
                fabric_multicast_noc_unicast_atomic_inc_set_state<
                    UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                    pkt_hdr_sem_inc,
                    static_cast<uint8_t>(barrier_multicast_route_info.start_distance_in_hops),
                    static_cast<uint8_t>(barrier_multicast_route_info.range_hops),
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, static_cast<uint32_t>(1)});
                if constexpr (topology == Topology::Linear) {
                    uint64_t same = safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0);
                    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        fabric_direction_connection,
                        pkt_hdr_sem_inc,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{same, 0});
                    uint64_t opp =
                        safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, barrier_sem, 0);
                    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        fabric_direction_connection,
                        pkt_hdr_sem_inc,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{opp, 0});
                } else {
                    uint64_t addr = safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0);
                    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        fabric_direction_connection,
                        pkt_hdr_sem_inc,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{addr, 0});
                }
            }
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), barrier_target_count);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
        }

        uint32_t slice_writes = 0;
        bool split_forwarding_enabled = false;
        if constexpr (topology == Topology::Ring) {
            if (ring_size % 2 == 0 && ring_size > 2 && input_tile_id_end - input_tile_id_start >= 2) {
                split_forwarding_enabled = true;
            }
        }

        uint32_t pages_read_in_row = start_pages_read_in_row;
        uint32_t row_offset = start_row_offset;
        uint32_t tiles_read = input_tile_id_start;
        uint32_t tiles_to_read = input_tile_id_end;
        uint32_t position = reverse ? (ring_size - 1) - my_chip_id : my_chip_id;
        uint32_t tile_id_start;
        if constexpr (gather_dim == 3) {
            tile_id_start = position * input_tensor_Wt;
        } else if constexpr (gather_dim == 2) {
            tile_id_start = position * input_tensor_Ht * input_tensor_Wt;
        } else if constexpr (gather_dim == 1) {
            tile_id_start = position * input_tensor_C * input_tensor_Ht * input_tensor_Wt;
        } else {
            tile_id_start = position * input_batch_head_count * input_tensor_Ht * input_tensor_Wt;
        }

        if (valid_targets(direction)) {
            constexpr uint32_t scatter_header_chunk_count = num_tiles_to_write_per_packet < NOC_SCATTER_WRITE_MIN_CHUNKS
                                                                ? NOC_SCATTER_WRITE_MIN_CHUNKS
                                                                : num_tiles_to_write_per_packet;
            static_assert(scatter_header_chunk_count <= NOC_SCATTER_WRITE_MAX_CHUNKS, "tiles per packet > 4");
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

        uint64_t out_ready_sem_noc_addr_in_pkt =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
        auto send_sem_inc = [&]() {
            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_direction_connection,
                pkt_hdr_sem_inc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
        };
        uint32_t num_channels_processed_in_current_batch = 0;
        uint32_t chunk_count = 0;

        // Local slice: send it (and, on the dir-1 worker, also write it locally).
        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
            chunk_count = 0;
            while (tiles_read < tiles_to_read) {
                uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                uint32_t n = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);
                cb_output.wait_front(num_tiles_to_write_per_packet);
                size_t l1_read_addr = cb_output.get_read_ptr();
                uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
                uint64_t noc_addrs[4] = {0, 0, 0, 0};
                uint64_t local_noc_addrs[4] = {0, 0, 0, 0};
                for (uint32_t i = 0; i < n; i++) {
                    uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;
                    pages_read_in_row++;
                    if (pages_read_in_row >= input_tensor_Wt) {
                        row_offset += output_tensor_Wt;
                        pages_read_in_row = 0;
                    }
                    noc_addrs[i] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_id, 0);
                    local_noc_addrs[i] = output_addrgen.get_noc_addr(tile_id);
                }
                if (direction == 1) {
                    if constexpr (num_targets_backward_direction) {
                        send_packet(
                            fabric_direction_connection,
                            pkt_scatter_hdr,
                            pkt_unicast_hdr,
                            l1_read_addr,
                            noc_addrs,
                            chunk_sizes,
                            n);
                    }
                    if constexpr (A::write_local) {
                        for (uint32_t i = 0; i < n; i++) {
                            noc_async_write(l1_read_addr + i * page_size, local_noc_addrs[i], page_size);
                        }
                        noc_obj.async_write_barrier();
                    }
                } else {
                    if constexpr (num_targets_forward_direction) {
                        send_packet(
                            fabric_direction_connection,
                            pkt_scatter_hdr,
                            pkt_unicast_hdr,
                            l1_read_addr,
                            noc_addrs,
                            chunk_sizes,
                            n);
                    }
                }
                tiles_read += n;
                noc_obj.async_writes_flushed();
                cb_output.pop_front(num_tiles_to_write_per_packet);
                chunk_count++;
                if (chunk_count % chunks_per_sync == 0) {
                    if (valid_targets(direction)) {
                        send_sem_inc();
                    }
                }
                noc_obj.async_writes_flushed();
            }
            if (chunk_count % chunks_per_sync != 0) {
                if (valid_targets(direction)) {
                    send_sem_inc();
                }
            }
            num_channels_processed_in_current_batch++;
            if (gather_dim == 1 && num_channels_processed_in_current_batch == input_tensor_C) {
                tile_id_start += output_tensor_Wt * output_tensor_Ht * (output_tensor_C - input_tensor_C + 1);
            } else {
                tile_id_start += output_tensor_Wt * output_tensor_Ht;
            }
            if (num_channels_processed_in_current_batch == input_tensor_C) {
                num_channels_processed_in_current_batch = 0;
            }
            tiles_read = input_tile_id_start;
            tiles_to_read = input_tile_id_end;
            pages_read_in_row = start_pages_read_in_row;
            row_offset = start_row_offset;
        }

        if constexpr (A::signal_slices && A::write_local) {
            if (direction == 1) {  // the dir-1 worker wrote its part of the local slice
                noc_obj.async_write_barrier();
                const uint32_t counters = rt<A>(sig_arg);
                const uint32_t n = rt<A>(sig_arg + 1) * A::signal_copies;
                for (uint32_t t = 0; t < n; t++) {
                    noc_semaphore_inc(
                        get_noc_addr(rt<A>(sig_arg + 2 + 2 * t), rt<A>(sig_arg + 3 + 2 * t), counters + 16 * position),
                        1);
                }
            }
        }

        // Forward the slices this worker relays.
        uint32_t writes_expected = 0;
        if constexpr (topology == Topology::Linear) {
            if (valid_targets_backward(direction)) {
                writes_expected = num_targets_forward_direction;
            } else if (valid_targets_forward(direction)) {
                writes_expected = num_targets_backward_direction;
            }
        } else if constexpr (topology == Topology::Ring) {
            if (direction == 1) {
                writes_expected = num_targets_backward_direction - 1;
                if (split_forwarding_enabled) {
                    writes_expected++;
                }
            } else {
                writes_expected = num_targets_forward_direction - 1;
            }
        }

        while (slice_writes < writes_expected) {
            bool is_last_slice = (slice_writes == writes_expected - 1);
            int slice_chip_id;
            uint32_t actual_slice_chip_id;
            if (direction == 1) {
                slice_chip_id = my_chip_id + slice_writes + 1;
                actual_slice_chip_id = (slice_chip_id >= (int)ring_size) ? slice_chip_id - ring_size : slice_chip_id;
            } else {
                slice_chip_id = my_chip_id - slice_writes - 1;
                actual_slice_chip_id = (slice_chip_id < 0) ? ring_size + slice_chip_id : slice_chip_id;
            }
            if constexpr (reverse) {
                actual_slice_chip_id = (ring_size - 1) - actual_slice_chip_id;
            }
            uint32_t f_tiles_read = input_tile_id_start;
            uint32_t f_tiles_to_read = input_tile_id_end;
            uint32_t f_row_offset = start_row_offset;
            uint32_t f_pages_read_in_row = start_pages_read_in_row;
            const uint32_t slice_Wt = input_tensor_Wt;
            const uint32_t stride_Wt = output_tensor_Wt;
            auto apply_split = [&]() {
                if (split_forwarding_enabled && is_last_slice) {
                    uint32_t first_half_tiles = (input_tile_id_end - input_tile_id_start) / 2;
                    if (direction == 0) {
                        f_tiles_to_read = input_tile_id_start + first_half_tiles;
                    } else {
                        f_tiles_read = input_tile_id_start + first_half_tiles;
                        uint32_t tiles_to_skip = first_half_tiles;
                        while (tiles_to_skip > 0) {
                            if (tiles_to_skip < slice_Wt - f_pages_read_in_row) {
                                f_pages_read_in_row += tiles_to_skip;
                                tiles_to_skip = 0;
                            } else {
                                tiles_to_skip -= (slice_Wt - f_pages_read_in_row);
                                f_row_offset += stride_Wt;
                                f_pages_read_in_row = 0;
                            }
                        }
                    }
                }
            };
            apply_split();
            uint32_t f_tile_id_start;
            if constexpr (gather_dim == 3) {
                f_tile_id_start = actual_slice_chip_id * input_tensor_Wt;
            } else if constexpr (gather_dim == 2) {
                f_tile_id_start = actual_slice_chip_id * input_tensor_Ht * input_tensor_Wt;
            } else if constexpr (gather_dim == 1) {
                f_tile_id_start = actual_slice_chip_id * input_tensor_C * input_tensor_Ht * input_tensor_Wt;
            } else {
                f_tile_id_start = actual_slice_chip_id * input_batch_head_count * input_tensor_Ht * input_tensor_Wt;
            }
            num_channels_processed_in_current_batch = 0;
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                chunk_count = 0;
                while (f_tiles_read < f_tiles_to_read) {
                    uint32_t n = std::min(f_tiles_to_read - f_tiles_read, num_tiles_to_write_per_packet);
                    cb_output.wait_front(num_tiles_to_write_per_packet);
                    size_t l1_read_addr = cb_output.get_read_ptr();
                    uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
                    uint64_t noc_addrs[4] = {0, 0, 0, 0};
                    for (uint32_t i = 0; i < n; i++) {
                        uint32_t tile_id = f_tile_id_start + f_row_offset + f_pages_read_in_row;
                        f_pages_read_in_row++;
                        if (f_pages_read_in_row >= slice_Wt) {
                            f_row_offset += stride_Wt;
                            f_pages_read_in_row = 0;
                        }
                        noc_addrs[i] =
                            tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_id, 0);
                    }
                    send_packet(
                        fabric_direction_connection,
                        pkt_scatter_hdr,
                        pkt_unicast_hdr,
                        l1_read_addr,
                        noc_addrs,
                        chunk_sizes,
                        n);
                    f_tiles_read += n;
                    noc_obj.async_writes_flushed();
                    cb_output.pop_front(num_tiles_to_write_per_packet);
                    chunk_count++;
                    if (chunk_count % chunks_per_sync == 0) {
                        send_sem_inc();
                    }
                    noc_obj.async_writes_flushed();
                }
                if (chunk_count % chunks_per_sync != 0) {
                    send_sem_inc();
                }
                num_channels_processed_in_current_batch++;
                if (gather_dim == 1 && num_channels_processed_in_current_batch == input_tensor_C) {
                    f_tile_id_start += output_tensor_Wt * output_tensor_Ht * (output_tensor_C - input_tensor_C + 1);
                } else {
                    f_tile_id_start += output_tensor_Wt * output_tensor_Ht;
                }
                if (num_channels_processed_in_current_batch == input_tensor_C) {
                    num_channels_processed_in_current_batch = 0;
                }
                f_tiles_read = input_tile_id_start;
                f_tiles_to_read = input_tile_id_end;
                f_row_offset = start_row_offset;
                f_pages_read_in_row = start_pages_read_in_row;
                apply_split();
            }
            slice_writes++;
        }
        noc_obj.async_write_barrier();
        noc_obj.async_atomic_barrier();
    }

    void teardown() {}
};

}  // namespace lego::ag
