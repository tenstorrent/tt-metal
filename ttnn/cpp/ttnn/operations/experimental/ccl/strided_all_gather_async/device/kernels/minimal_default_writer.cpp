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
#include "cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/strided_all_gather_common.hpp"
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
constexpr uint32_t max_tiles_per_packet = get_compile_time_arg_val(2);
constexpr uint32_t output_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(5);
constexpr bool fuse_op = get_compile_time_arg_val(6);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(7));
constexpr bool direction = get_compile_time_arg_val(8);  // 1 is forward, 0 is backward
constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(9);
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

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_batch_head_count = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

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
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_sem_inc,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint16_t>(1)});  // increment 1

    ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc, unicast_route_info);
    // 2. unicast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);

    uint32_t writes_expected = 0;
    if (topology == Topology::Linear) {
        if (direction == 1 && num_targets_backward_direction) {
            writes_expected = num_targets_forward_direction;
        } else if (direction == 0 && num_targets_forward_direction) {
            writes_expected = num_targets_backward_direction;
        }
    } else if (topology == Topology::Ring) {
        if (direction == 1) {
            writes_expected = num_targets_backward_direction - 1;
        } else {
            writes_expected = num_targets_forward_direction - 1;
        }
    }

    uint32_t global_tile_id_start = input_tile_id_start;
    uint32_t global_tile_index = 0;
    uint32_t output_tiles_per_bh = output_tensor_Wt * output_tensor_Ht;
    uint32_t tile_end_id = output_tensor_Wt * (output_tensor_Ht - 1) + input_tensor_Wt * (my_chip_id + 1);
    uint32_t chunks_per_core = div_up(input_tiles_per_core, tiles_per_chunk);

    // Write out the local slice to both DRAM and forward and backward
    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
        global_tile_index = 0;
        for (uint32_t chunk_idx = 0; chunk_idx < chunks_per_core; chunk_idx++) {
            write_chunk(
                global_tile_index,
                global_tile_id_start,
                cb_output_id,
                input_tiles_per_core,
                tiles_per_chunk,
                max_tiles_per_packet,
                ag_worker_cores,
                output_addrgen,
                output_page_size,
                input_tensor_Wt,
                output_tensor_Wt,
                my_chip_id,
                mux_connection,
                pkt_scatter_hdr,
                pkt_unicast_hdr,
                pkt_hdr_sem_inc,
                out_ready_sem_noc_addr_in_pkt,
                direction,
                true);

            // Forward chunks
            uint32_t slice_writes = 0;
            uint32_t next_tile_to_write = 0;
            while (slice_writes < writes_expected) {
                uint32_t actual_sender_chip_id = get_sender_id(direction, my_chip_id, slice_writes, ring_size);

                next_tile_to_write = write_chunk(
                    global_tile_index,
                    global_tile_id_start,
                    cb_output_id,
                    input_tiles_per_core,
                    tiles_per_chunk,
                    max_tiles_per_packet,
                    ag_worker_cores,
                    output_addrgen,
                    output_page_size,
                    input_tensor_Wt,
                    output_tensor_Wt,
                    actual_sender_chip_id,
                    mux_connection,
                    pkt_scatter_hdr,
                    pkt_unicast_hdr,
                    pkt_hdr_sem_inc,
                    out_ready_sem_noc_addr_in_pkt,
                    direction,
                    false);
                slice_writes++;
            }
            global_tile_index = next_tile_to_write;
        }
        global_tile_id_start += output_tiles_per_bh;
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
