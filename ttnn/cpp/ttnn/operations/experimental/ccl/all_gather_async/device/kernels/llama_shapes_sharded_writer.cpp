// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/global_semaphore.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/buffer_types.hpp"
#include "api/tensor/local_tensor_accessor.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_arg(args::my_chip_id);
constexpr uint32_t num_packet_headers_storable = get_arg(args::num_packet_headers_storable);
constexpr uint32_t packet_size_in_pages = get_arg(args::packet_size_in_pages);
constexpr uint32_t tensor0_page_size = get_arg(args::tensor0_page_size);
constexpr uint32_t num_targets_forward_direction = get_arg(args::num_targets_forward_direction);
constexpr uint32_t num_targets_backward_direction = get_arg(args::num_targets_backward_direction);
constexpr uint32_t ring_size = get_arg(args::ring_size);
// For 1D fabric, dst_mesh_id carries start_distance_in_hops and dst_chip_id carries range_hops.
constexpr ccl_routing_utils::line_multicast_route_info_t forward_multicast_route_info = {
    .dst_mesh_id = get_arg(args::forward_mcast_dst_mesh_id),
    .dst_chip_id = get_arg(args::forward_mcast_dst_chip_id),
    .e_num_hops = get_arg(args::forward_mcast_e_num_hops),
    .w_num_hops = get_arg(args::forward_mcast_w_num_hops),
    .n_num_hops = get_arg(args::forward_mcast_n_num_hops),
    .s_num_hops = get_arg(args::forward_mcast_s_num_hops),
};
constexpr ccl_routing_utils::line_multicast_route_info_t backward_multicast_route_info = {
    .dst_mesh_id = get_arg(args::backward_mcast_dst_mesh_id),
    .dst_chip_id = get_arg(args::backward_mcast_dst_chip_id),
    .e_num_hops = get_arg(args::backward_mcast_e_num_hops),
    .w_num_hops = get_arg(args::backward_mcast_w_num_hops),
    .n_num_hops = get_arg(args::backward_mcast_n_num_hops),
    .s_num_hops = get_arg(args::backward_mcast_s_num_hops),
};

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    // Load the input tensor spec
    const LocalTensorAccessor<uint8_t> output(tensor::output);
    address_t tensor_address0 = static_cast<address_t>(local_mem::detail::AddressAccess::address(output.local_mem()));
    experimental::GlobalSemaphore out_ready_semaphore(global_sem::out_ready_semaphore);
    uint32_t num_tiles_per_core = get_arg(args::num_tiles_per_core);
    uint32_t num_tiles_to_read = get_arg(args::num_tiles_to_read);
    uint32_t first_core_tile_start_offset = get_arg(args::first_core_tile_start_offset);
    uint32_t num_cores = get_arg(args::num_cores);
    bool wait_output_semaphore = get_arg(args::wait_output_semaphore);
    bool reset_global_semaphore = get_arg(args::reset_global_semaphore);
    const uint8_t out_ready_sem_noc0_x = get_arg(args::out_ready_sem_noc0_x);
    const uint8_t out_ready_sem_noc0_y = get_arg(args::out_ready_sem_noc0_y);
    uint32_t out_ready_sem_wait_value = get_arg(args::out_ready_sem_wait_value);
    const uint8_t barrier_sem_noc0_x = get_arg(args::barrier_sem_noc0_x);
    const uint8_t barrier_sem_noc0_y = get_arg(args::barrier_sem_noc0_y);
    // The output cores' NoC x coordinates are varargs [0, num_cores), their y coordinates [num_cores, 2 * num_cores).

    // Build and start opening the forward and backward fabric connections; the open finishes below.
    const bool has_forward_connection = tt::tt_fabric::has_fabric_connection(fabric::forward);
    tt::tt_fabric::WorkerToFabricEdmSender forward_connection;
    if (has_forward_connection) {
        forward_connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::forward);
        forward_connection.open_start();
    }
    const bool has_backward_connection = tt::tt_fabric::has_fabric_connection(fabric::backward);
    tt::tt_fabric::WorkerToFabricEdmSender backward_connection;
    if (has_backward_connection) {
        backward_connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::backward);
        backward_connection.open_start();
    }

    Noc noc_obj;
    DataflowBuffer dfb_packet_header(dfb::packet_header);
    DataflowBuffer dfb0(dfb::data);

    // packet header dfb
    dfb_packet_header.reserve_back(1);
    auto packet_header_buffer_addr_forward = dfb_packet_header.get_write_ptr();
    dfb_packet_header.push_back(1);
    dfb_packet_header.reserve_back(1);
    auto packet_header_buffer_addr_backward = dfb_packet_header.get_write_ptr();
    dfb_packet_header.push_back(1);
    dfb_packet_header.reserve_back(1);
    auto packet_header_buffer_seminc = dfb_packet_header.get_write_ptr();
    dfb_packet_header.push_back(1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_forward, forward_multicast_route_info);
    ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_backward, backward_multicast_route_info);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_hdr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);

    if (has_forward_connection) {
        forward_connection.open_finish();
    }
    if (has_backward_connection) {
        backward_connection.open_finish();
    }

#ifdef USE_BARRIER_SEM
    {
        experimental::GlobalSemaphore barrier_semaphore(global_sem::barrier_semaphore);
        uint64_t barrier_sem_noc_addr_in_pkt =
            safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_semaphore.address(), 0);
        pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            barrier_sem_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1

        // Write the mcast packet (forward)
        if (has_forward_connection) {
            forward_connection.wait_for_empty_write_slot();
            ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr, forward_multicast_route_info);
            forward_connection.send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }
        // Write the mcast packet (backward)
        if (has_backward_connection) {
            ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr, backward_multicast_route_info);
            backward_connection.wait_for_empty_write_slot();
            backward_connection.send_payload_non_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }

        barrier_semaphore.wait_min(ring_size - 1);
        barrier_semaphore.set(0);
    }
#endif

    // 1. mcast via fabric to remote tensor addresses
    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core = std::min(num_tiles_per_core - shard_tile_id, packet_size_in_pages);
        num_tiles_to_read_this_core = std::min(num_tiles_to_read - tiles_read, num_tiles_to_read_this_core);
        dfb0.wait_front(num_tiles_to_read_this_core);
        size_t l1_read_addr = dfb0.get_read_ptr();

        const uint32_t core_noc_x = get_vararg(core_id);
        const uint32_t core_noc_y = get_vararg(num_cores + core_id);
        const uint32_t dest_addr = tensor_address0 + shard_tile_id * tensor0_page_size;
        uint64_t noc0_dest_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dest_addr, 0 /*noc_id*/);
        const uint32_t payload_size_bytes = num_tiles_to_read_this_core * tensor0_page_size;

        // Write the payload to this device's output, then send it to the forward and backward devices.
        noc_obj.async_write(
            CoreLocalMem<uint8_t>(l1_read_addr),
            UnicastEndpoint{},
            payload_size_bytes,
            {},
            {.noc_x = core_noc_x, .noc_y = core_noc_y, .addr = dest_addr});
        if (has_forward_connection) {
            pkt_hdr_forward->to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
            perform_payload_send(forward_connection, l1_read_addr, payload_size_bytes, pkt_hdr_forward);
        }
        if (has_backward_connection) {
            pkt_hdr_backward->to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
            perform_payload_send(backward_connection, l1_read_addr, payload_size_bytes, pkt_hdr_backward);
        }
        // This issues a flush barrier
        noc_obj.async_writes_flushed();

        dfb0.pop_front(num_tiles_to_read_this_core);
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id += num_tiles_to_read_this_core;
        if (shard_tile_id >= num_tiles_per_core) {
            shard_tile_id = 0;
            core_id++;
        }
    }

    // 2. mcast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_semaphore.address(), 0);
    pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1
    // Write the mcast packet (forward)
    if (has_forward_connection) {
        forward_connection.wait_for_empty_write_slot();
        ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr, forward_multicast_route_info);
        forward_connection.send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // Write the mcast packet (backward)
    if (has_backward_connection) {
        ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr, backward_multicast_route_info);
        backward_connection.wait_for_empty_write_slot();
        backward_connection.send_payload_non_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    if (has_forward_connection) {
        forward_connection.close_start();
    }
    if (has_backward_connection) {
        backward_connection.close_start();
    }
    if (has_forward_connection) {
        forward_connection.close_finish();
    }
    if (has_backward_connection) {
        backward_connection.close_finish();
    }
    // increment locally
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_semaphore.address());
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);

    // 3. wait for mcast output ready semaphore
    if (wait_output_semaphore) {
        out_ready_semaphore.wait_min(out_ready_sem_wait_value);
    }

    // 4. global semaphore reset
    if (reset_global_semaphore) {
        out_ready_semaphore.set(0);
    }

    noc_obj.async_full_barrier();
}
