// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
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

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(1);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(2);
constexpr uint32_t output_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(5);
constexpr bool fuse_op = get_compile_time_arg_val(6);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(7));
constexpr bool direction = get_compile_time_arg_val(8);  // 1 is forward, 0 is backward
constexpr uint32_t chunks_per_sync = get_compile_time_arg_val(9);
constexpr uint32_t ag_worker_cores = get_compile_time_arg_val(10);
constexpr bool is_termination_master = get_compile_time_arg_val(11);
constexpr uint8_t fabric_mux_x = get_compile_time_arg_val(12);
constexpr uint8_t fabric_mux_y = get_compile_time_arg_val(13);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(14);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(15);
constexpr size_t fabric_mux_channel_base_address = get_compile_time_arg_val(16);
constexpr size_t fabric_mux_connection_info_address = get_compile_time_arg_val(17);
constexpr size_t fabric_mux_connection_handshake_address = get_compile_time_arg_val(18);
constexpr size_t fabric_mux_flow_control_address = get_compile_time_arg_val(19);
constexpr size_t fabric_mux_buffer_index_address = get_compile_time_arg_val(20);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(21);
constexpr uint8_t fabric_mux_channel_id = get_compile_time_arg_val(22);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(23);

constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<24>();
constexpr ccl_routing_utils::line_multicast_route_info_t barrier_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<24 + ccl_routing_utils::num_line_unicast_args>();

inline constexpr uint32_t sharded_args_start_idx =
    24 + ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args;

uint32_t get_next_tile_to_write(
    uint32_t local_tile_index,
    uint32_t input_start_tile_index,
    uint32_t ag_parallel_factor,
    uint32_t input_tensor_Wt,
    uint32_t output_tensor_Wt,
    uint32_t device_index) {
    uint32_t input_tile_index = input_start_tile_index + local_tile_index * ag_parallel_factor;
    uint32_t input_row = input_tile_index / input_tensor_Wt;
    uint32_t input_col = input_tile_index % input_tensor_Wt;
    return input_row * output_tensor_Wt + device_index * input_tensor_Wt +
           input_col;  // TODO should pass device_index*input_tensor_Wt to prevent recalculating them
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_C = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_C = get_arg_val<uint32_t>(arg_idx++);
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_batch_head_count = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_end = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);

    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t opposite_core_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t opposite_core_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

    bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_mux_clients = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto output_tensor_args = TensorAccessorArgs<sharded_args_start_idx>();
    const auto output_addrgen = TensorAccessor(output_tensor_args, output_address, output_page_size);

    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle;
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
        mux_connection_handle = &mux_connection;
    } else {
        mux_connection_handle = nullptr;
    }

    if (mux_connection_valid) {
        // need to wait for fabric mux to be ready to accept connections
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);
    }

    // pre-populate packet headers
    auto pkt_scatter_hdr = PacketHeaderPool::allocate_header();
    auto pkt_unicast_hdr = PacketHeaderPool::allocate_header();
    auto pkt_hdr_sem_inc = PacketHeaderPool::allocate_header();

    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_connect(*mux_connection_handle);
    }

    uint32_t slice_writes = 0;

    auto page_size = tt::tt_fabric::linear::addrgen_detail::get_page_size(output_addrgen);
    fabric_unicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        pkt_scatter_hdr,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        NocUnicastScatterCommandHeader{
            {0, 0},  // ignore
            static_cast<uint16_t>(page_size)},
        page_size * 2);

    fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, output_page_size);

    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Wrap | UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_sem_inc,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                         // ignore
            static_cast<uint16_t>(1),  // increment 1
            32});
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc, unicast_route_info);
    // 2. unicast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);

    uint32_t global_tile_id_start = input_tile_id_start;
    uint32_t current_tile_index = 0;
    uint32_t output_tiles_per_bh = output_tensor_Wt * output_tensor_Ht;
    uint32_t tile_end_id = output_tensor_Wt * (output_tensor_Ht - 1) + input_tensor_Wt * (my_chip_id + 1);
    bool done = false;

    // Write out the local slice to both DRAM and forward and backward
    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
        uint32_t tiles_to_put_in_current_packet = 2;
        while (!done) {
            cb_wait_front(cb_output_id, num_tiles_to_write_per_packet);
            size_t l1_read_addr = get_read_ptr(cb_output_id);

            uint32_t tile_one_id = get_next_tile_to_write(
                current_tile_index,
                global_tile_id_start,
                ag_worker_cores,
                input_tensor_Wt,
                output_tensor_Wt,
                my_chip_id);

            if (tile_one_id >= tile_end_id) {
                done = true;
                break;
            }
            current_tile_index++;
            uint32_t tile_two_id = get_next_tile_to_write(
                current_tile_index,
                global_tile_id_start,
                ag_worker_cores,
                input_tensor_Wt,
                output_tensor_Wt,
                my_chip_id);
            if (tile_two_id >= tile_end_id) {
                done = true;
                tiles_to_put_in_current_packet = 1;
            }
            current_tile_index++;
            auto noc_address0 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_one_id, 0);
            auto noc_address1 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_two_id, 0);

            // Will have more cases once scatter-write supports more than 2 distinct addresses
            switch (tiles_to_put_in_current_packet) {
                case 2: {
                    if (direction == 1) {
                        fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                            mux_connection_handle,
                            pkt_scatter_hdr,
                            l1_read_addr,
                            NocUnicastScatterCommandHeader{{noc_address0, noc_address1}, 0});
                        uint64_t local_noc0_dest_noc_addr_tile_one = get_noc_addr(tile_one_id, output_addrgen);
                        uint64_t local_noc0_dest_noc_addr_tile_two = get_noc_addr(tile_two_id, output_addrgen);

                        noc_async_write(l1_read_addr, local_noc0_dest_noc_addr_tile_one, output_page_size);
                        noc_async_write(
                            l1_read_addr + output_page_size, local_noc0_dest_noc_addr_tile_two, output_page_size);
                        noc_async_write_barrier();
                    } else {
                        fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                            mux_connection_handle,
                            pkt_scatter_hdr,
                            l1_read_addr,
                            NocUnicastScatterCommandHeader{{noc_address0, noc_address1}, 0});
                    }
                    break;
                }
                case 1:
                default: {
                    if (direction == 1) {
                        fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            mux_connection_handle,
                            pkt_unicast_hdr,
                            l1_read_addr,
                            NocUnicastCommandHeader{noc_address0});
                        uint64_t local_noc0_dest_noc_addr = get_noc_addr(tile_one_id, output_addrgen);
                        noc_async_write(l1_read_addr, local_noc0_dest_noc_addr, output_page_size);
                        noc_async_write_barrier();
                    } else {
                        fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            mux_connection_handle,
                            pkt_unicast_hdr,
                            l1_read_addr,
                            NocUnicastCommandHeader{noc_address0});
                    }
                    break;
                }
            }
            noc_async_writes_flushed();

            cb_pop_front(cb_output_id, num_tiles_to_write_per_packet);

            noc_async_writes_flushed();
        }

        // Write the semaphore packet
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            mux_connection_handle,
            pkt_hdr_sem_inc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                out_ready_sem_noc_addr_in_pkt, 0, 0  // ignore
            });

        global_tile_id_start += output_tiles_per_bh;
        current_tile_index = 0;
        done = false;
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);

        if constexpr (is_termination_master) {
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
    noc_async_write_barrier();
}
