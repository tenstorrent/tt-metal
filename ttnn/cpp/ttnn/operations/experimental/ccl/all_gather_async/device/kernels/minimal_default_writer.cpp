// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"

using address_t = uint32_t;
using ttnn::ccl::Topology;
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
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(7));
constexpr uint32_t gather_dim = get_compile_time_arg_val(8);
constexpr uint32_t input_batch_head_count = get_compile_time_arg_val(9);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(10);
constexpr uint32_t input_tensor_Ht = get_compile_time_arg_val(11);
constexpr uint32_t input_tensor_C = get_compile_time_arg_val(12);
constexpr uint32_t output_tensor_Wt = get_compile_time_arg_val(13);
constexpr uint32_t output_tensor_Ht = get_compile_time_arg_val(14);
constexpr uint32_t output_tensor_C = get_compile_time_arg_val(15);
constexpr bool fuse_op = get_compile_time_arg_val(16);
constexpr uint32_t reverse = get_compile_time_arg_val(17) == 1;
#ifdef USE_WORKER_MUX
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(18);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(19);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(20);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(21);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(22);
constexpr uint32_t rt_arg_count = 23;
#else
constexpr uint32_t rt_arg_count = 18;
#endif

constexpr ccl_routing_utils::line_unicast_route_info_t forward_unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<rt_arg_count>();
constexpr ccl_routing_utils::line_multicast_route_info_t forward_barrier_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<
        rt_arg_count + ccl_routing_utils::num_line_unicast_args>();

constexpr ccl_routing_utils::line_unicast_route_info_t backward_unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<
        rt_arg_count + ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args>();
constexpr ccl_routing_utils::line_multicast_route_info_t backward_barrier_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<
        rt_arg_count + 2 * ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args>();

inline constexpr uint32_t sharded_args_start_idx =
    rt_arg_count + 2 * (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args);

namespace detail {

bool valid_targets_forward(const bool direction) {
    if constexpr (num_targets_forward_direction) {
        return (direction == 0);
    } else {
        return false;
    }
}

bool valid_targets_backward(const bool direction) {
    if constexpr (num_targets_backward_direction) {
        return (direction == 1);
    } else {
        return false;
    }
}

bool valid_targets(const bool direction) {
    if constexpr (num_targets_backward_direction + num_targets_forward_direction == 0) {
        return false;
    } else {
        return (valid_targets_forward(direction) || valid_targets_backward(direction));
    }
}
}  // namespace detail
void kernel_main() {
    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t opposite_core_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t opposite_core_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

    const bool direction = get_arg_val<uint32_t>(arg_idx++);  // 0 is forward, 1 is backward
    const auto input_tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    const auto input_tile_id_end = get_arg_val<uint32_t>(arg_idx++);
    const auto start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const auto start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    const auto chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
#ifdef USE_WORKER_MUX
    bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool is_termination_master = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);

    uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
#endif
    const auto& unicast_route_info = (direction == 0) ? forward_unicast_route_info : backward_unicast_route_info;
    const auto& barrier_multicast_route_info =
        (direction == 0) ? forward_barrier_multicast_route_info : backward_barrier_multicast_route_info;

#ifdef OUTPUT_IS_SHARDED
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(sharded_args_start_idx),   // Memory layout
        get_compile_time_arg_val(sharded_args_start_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(sharded_args_start_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(sharded_args_start_idx + 3),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(sharded_args_start_idx + 4),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(sharded_args_start_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(sharded_args_start_idx + 6)>;  // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<tensor_shard_info> output_addrgen = {
        .bank_base_address = output_address, .shard_array = mapping_table};

    arg_idx += rt_increment;
#else
    constexpr auto output_tensor_args = TensorAccessorArgs<sharded_args_start_idx>();
    const auto output_addrgen = TensorAccessor(output_tensor_args, output_address, page_size);
#endif

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
    } else {
        fabric_direction_connection = nullptr;
    }

    if (mux_connection_valid) {
        // need to wait for fabric mux to be ready to accept connections
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);
    }
#else
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
#endif
    /* Args for overlapped all gather */
    OpSignaler op_signaler_sender;
    uint32_t self_write_done_semaphore_addr;
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
    // pre-populate packet headers
    auto pkt_scatter_hdr = PacketHeaderPool::allocate_header();
    auto pkt_unicast_hdr = PacketHeaderPool::allocate_header();
    auto pkt_hdr_sem_inc = PacketHeaderPool::allocate_header();

    if (use_barrier_sem) {
        if (detail::valid_targets(direction)) {
            // only initialize if we're actually going to send something over fabric

            ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_sem_inc, barrier_multicast_route_info);
            fabric_multicast_noc_unicast_atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                pkt_hdr_sem_inc,
                static_cast<uint8_t>(barrier_multicast_route_info.start_distance_in_hops),
                static_cast<uint8_t>(barrier_multicast_route_info.range_hops),
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    0,  // ignore
                    static_cast<uint32_t>(1)});

            if constexpr (topology == Topology::Linear) {
                // multicast to both the forward and backward worker on all devices that you write to.
                // this only executes if the worker actually sends something over fabric (i.e. the writers
                // on the end of the line pointing outward don't issue sem incs)

                // device going in the same direction
                uint64_t same_direction_barrier_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0);
                fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    fabric_direction_connection,
                    pkt_hdr_sem_inc,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{same_direction_barrier_sem_noc_addr_in_pkt, 0});

                // device going in the opposite direction
                uint64_t opposite_direction_barrier_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, barrier_sem, 0);
                fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    fabric_direction_connection,
                    pkt_hdr_sem_inc,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{opposite_direction_barrier_sem_noc_addr_in_pkt, 0});

            } else if constexpr (topology == Topology::Ring) {
                // multicast to entire ring of workers going in the same direction
                uint64_t barrier_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0);
                fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    fabric_direction_connection,
                    pkt_hdr_sem_inc,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
            } else {
                ASSERT(false);
            }
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), ring_size - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    uint32_t slice_writes = 0;
    bool split_forwarding_enabled = false;
    if constexpr (topology == Topology::Ring) {
        if (ring_size % 2 == 0 && ring_size > 2 &&
            input_tile_id_end - input_tile_id_start >= 2) {  // if ring size is even, we need to write the first half of
                                                             // the tiles, otherwise we write the entire packet
            split_forwarding_enabled = true;
        }
    }

    // Write out the local slice to both DRAM and forward and backward
    uint32_t pages_read_in_row = start_pages_read_in_row;
    uint32_t row_offset = start_row_offset;
    uint32_t tiles_read = input_tile_id_start;
    uint32_t tiles_to_read = input_tile_id_end;

    uint32_t position = my_chip_id;
    if constexpr (reverse) {
        position = (ring_size - 1) - my_chip_id;
    }

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

    // only initialize if we're actually going to send something over fabric
    if (detail::valid_targets(direction)) {
        static_assert(num_tiles_to_write_per_packet <= 4, "tiles per packet > 4 is unsupported");
        uint64_t dummy_addrs[4] = {0, 0, 0, 0};
        uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
        fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            pkt_scatter_hdr,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            NocUnicastScatterCommandHeader(dummy_addrs, chunk_sizes, num_tiles_to_write_per_packet),
            page_size * num_tiles_to_write_per_packet);

        fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, page_size);

        fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            pkt_hdr_sem_inc,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                0,  // ignore
                static_cast<uint32_t>(1)});
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc, unicast_route_info);
    }

    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
    uint32_t num_channels_processed_in_current_batch = 0;
    uint32_t chunk_count = 0;

        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
            chunk_count = 0;
            while (tiles_read < tiles_to_read) {
                uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                uint32_t tiles_to_put_in_current_packet =
                    std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);

            cb_wait_front(cb_output_id, num_tiles_to_write_per_packet);
            size_t l1_read_addr = get_read_ptr(cb_output_id);

            uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
            uint64_t noc_addrs[4] = {0, 0, 0, 0};
            uint64_t local_noc_addrs[4] = {0, 0, 0, 0};
            for (uint32_t i = 0; i < tiles_to_put_in_current_packet; i++) {
                uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;
                pages_read_in_row++;
                if (pages_read_in_row >= input_tensor_Wt) {
                    row_offset += output_tensor_Wt;
                    pages_read_in_row = 0;
                }

                noc_addrs[i] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_id, 0);
                local_noc_addrs[i] = get_noc_addr(tile_id, output_addrgen);
            }

            if (direction == 1) {
                if constexpr (num_targets_backward_direction) {
                    if (tiles_to_put_in_current_packet > 1) {
                        fabric_unicast_noc_scatter_write_with_state<
                            UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                            UnicastScatterWriteUpdateMask::PayloadSize>(
                            fabric_direction_connection,
                            pkt_scatter_hdr,
                            l1_read_addr,
                            NocUnicastScatterCommandHeader(noc_addrs, chunk_sizes, tiles_to_put_in_current_packet),
                            page_size * tiles_to_put_in_current_packet);
                    } else {
                        fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            fabric_direction_connection,
                            pkt_unicast_hdr,
                            l1_read_addr,
                            NocUnicastCommandHeader{noc_addrs[0]});
                    }
                }

                for (uint32_t i = 0; i < tiles_to_put_in_current_packet; i++) {
                    noc_async_write(l1_read_addr + i * page_size, local_noc_addrs[i], page_size);
                }
                noc_async_write_barrier();
            } else {
                if constexpr (num_targets_forward_direction) {
                    if (tiles_to_put_in_current_packet > 1) {
                        fabric_unicast_noc_scatter_write_with_state<
                            UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                            UnicastScatterWriteUpdateMask::PayloadSize>(
                            fabric_direction_connection,
                            pkt_scatter_hdr,
                            l1_read_addr,
                            NocUnicastScatterCommandHeader(noc_addrs, chunk_sizes, tiles_to_put_in_current_packet),
                            page_size * tiles_to_put_in_current_packet);
                    } else {
                        fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            fabric_direction_connection,
                            pkt_unicast_hdr,
                            l1_read_addr,
                            NocUnicastCommandHeader{noc_addrs[0]});
                    }
                }
            }
            tiles_read += tiles_to_put_in_current_packet;

                noc_async_writes_flushed();

                cb_pop_front(cb_output_id, num_tiles_to_write_per_packet);

            chunk_count++;
            if (chunk_count % chunks_per_sync == 0) {
                // 2. unicast output ready semaphore
                if (detail::valid_targets(direction)) {
                    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        fabric_direction_connection,
                        pkt_hdr_sem_inc,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
                }
            }
            noc_async_writes_flushed();
        }

        if (chunk_count % chunks_per_sync != 0) {
            // Write the unicast packet
            if (detail::valid_targets(direction)) {
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    fabric_direction_connection,
                    pkt_hdr_sem_inc,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
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

    // increment locally
    if constexpr (fuse_op) {
        if (direction == 1) {
            // Synchronize and signal that the local tensor slice is available
            op_signaler_sender.synchronize_workers_and_signal_op(my_chip_id);
            uint64_t self_write_done_semaphore_noc_addr =
                safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, self_write_done_semaphore_addr, 0);
            noc_semaphore_inc(self_write_done_semaphore_noc_addr, 1);
        }
    }

    uint32_t writes_expected = 0;
    if constexpr (topology == Topology::Linear) {
        if (detail::valid_targets_backward(direction)) {
            writes_expected = num_targets_forward_direction;
        } else if (detail::valid_targets_forward(direction)) {
            writes_expected = num_targets_backward_direction;
        }
    } else if constexpr (topology == Topology::Ring) {
        if (direction == 1) {
            writes_expected = num_targets_backward_direction - 1;
            if (split_forwarding_enabled) {
                writes_expected++;  // Backward worker will also forward 1 slice (but only half of it)
            }
        } else {
            writes_expected = num_targets_forward_direction - 1;
            // For 4-device ring, forward worker will only send half of last slice
        }
    }

    while (slice_writes < writes_expected) {
        // Direction == backward
        // Did I get something from my left to send to my right?
        // In the linear case, I expect num_targets_backward_direction slices from the left, and check if I have a
        // neighbor to the right
        // In the ring case, I expect to write to the right num_forward_target times
        // Direction == forward
        // Did I get something from my right to send to my left?
        // In the linear case, I expect num_targets_forward_direction slices from the right, and check if I have a
        // neighbor to the left
        // In the ring case, I expect to write to the left num_backward_target times

        // Check if this is the last slice for split-forwarding
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
        uint32_t tiles_read = input_tile_id_start;
        uint32_t tiles_to_read = input_tile_id_end;
        uint32_t tile_id_start;
        uint32_t row_offset = start_row_offset;
        uint32_t pages_read_in_row = start_pages_read_in_row;
        uint32_t slice_Wt = input_tensor_Wt;
        uint32_t stride_Wt = output_tensor_Wt;

        if (split_forwarding_enabled && is_last_slice) {
            uint32_t total_tiles = input_tile_id_end - input_tile_id_start;
            uint32_t first_half_tiles = total_tiles / 2;

            if (direction == 0) {
                // Forward worker: only forward first half
                tiles_to_read = input_tile_id_start + first_half_tiles;
            } else {
                // Backward worker: skip first half, forward second half
                tiles_read = input_tile_id_start + first_half_tiles;

                // Adjust starting position for tiles
                uint32_t tiles_to_skip = first_half_tiles;
                while (tiles_to_skip > 0) {
                    if (tiles_to_skip < slice_Wt - pages_read_in_row) {
                        pages_read_in_row += tiles_to_skip;
                        tiles_to_skip = 0;
                    } else {
                        tiles_to_skip -= (slice_Wt - pages_read_in_row);
                        row_offset += stride_Wt;
                        pages_read_in_row = 0;
                    }
                }
            }
        }
        if constexpr (gather_dim == 3) {
            tile_id_start = actual_slice_chip_id * input_tensor_Wt;
        } else if constexpr (gather_dim == 2) {
            tile_id_start = actual_slice_chip_id * input_tensor_Ht * input_tensor_Wt;
        } else if constexpr (gather_dim == 1) {
            tile_id_start = actual_slice_chip_id * input_tensor_C * input_tensor_Ht * input_tensor_Wt;
        } else {
            tile_id_start = actual_slice_chip_id * input_batch_head_count * input_tensor_Ht * input_tensor_Wt;
        }

        num_channels_processed_in_current_batch = 0;
        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
            chunk_count = 0;

            while (tiles_read < tiles_to_read) {
                uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                uint32_t tiles_to_put_in_current_packet =
                    std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);

                cb_wait_front(cb_output_id, num_tiles_to_write_per_packet);
                size_t l1_read_addr = get_read_ptr(cb_output_id);

                uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
                uint64_t noc_addrs[4] = {0, 0, 0, 0};
                for (uint32_t i = 0; i < tiles_to_put_in_current_packet; i++) {
                    uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;
                    pages_read_in_row++;
                    if (pages_read_in_row >= slice_Wt) {
                        row_offset += stride_Wt;
                        pages_read_in_row = 0;
                    }
                    noc_addrs[i] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_id, 0);
                }
                if (tiles_to_put_in_current_packet > 1) {
                    fabric_unicast_noc_scatter_write_with_state<
                        UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                        UnicastScatterWriteUpdateMask::PayloadSize>(
                        fabric_direction_connection,
                        pkt_scatter_hdr,
                        l1_read_addr,
                        NocUnicastScatterCommandHeader(noc_addrs, chunk_sizes, tiles_to_put_in_current_packet),
                        page_size * tiles_to_put_in_current_packet);
                } else {
                    fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                        fabric_direction_connection,
                        pkt_unicast_hdr,
                        l1_read_addr,
                        NocUnicastCommandHeader{noc_addrs[0]});
                }

                tiles_read += tiles_to_put_in_current_packet;

                noc_async_writes_flushed();

                cb_pop_front(cb_output_id, num_tiles_to_write_per_packet);

                chunk_count++;
                if (chunk_count % chunks_per_sync == 0) {
                    // 2. unicast output ready semaphore
                    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        fabric_direction_connection,
                        pkt_hdr_sem_inc,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
                }
                noc_async_writes_flushed();
            }

            if (chunk_count % chunks_per_sync != 0) {
                // 2. unicast output ready semaphore
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    fabric_direction_connection,
                    pkt_hdr_sem_inc,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
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

            // Reset for next batch, but respect split slice boundaries
            tiles_read = input_tile_id_start;
            tiles_to_read = input_tile_id_end;
            row_offset = start_row_offset;
            pages_read_in_row = start_pages_read_in_row;
            if (split_forwarding_enabled && is_last_slice) {
                uint32_t total_tiles = input_tile_id_end - input_tile_id_start;
                uint32_t first_half_tiles = total_tiles / 2;
                if (direction == 0) {
                    tiles_to_read = input_tile_id_start + first_half_tiles;
                } else {
                    tiles_read = input_tile_id_start + first_half_tiles;
                    // Re-adjust position for second half
                    uint32_t tiles_to_skip = first_half_tiles;
                    while (tiles_to_skip > 0) {
                        if (tiles_to_skip < slice_Wt - pages_read_in_row) {
                            pages_read_in_row += tiles_to_skip;
                            tiles_to_skip = 0;
                        } else {
                            tiles_to_skip -= (slice_Wt - pages_read_in_row);
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }
                    }
                }
            }
        }
        slice_writes++;
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
#ifdef USE_WORKER_MUX
    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(*fabric_direction_connection);

        if (is_termination_master) {
            auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
            noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
        } else {
            uint64_t dest_addr =
                safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
            noc_semaphore_inc(dest_addr, 1);
            noc_async_atomic_barrier();
        }
    }
#else
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }
#endif
    noc_async_write_barrier();
}
