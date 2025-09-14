// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <cstdint>
#include <utility>
#include "tools/profiler/kernel_profiler.hpp"

using address_t = uint32_t;
// UNUSED
FORCE_INLINE void advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    uint32_t num_targets_backward_direction,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    // const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const size_t payload_l1_address = l1_read_addr;
    pkt_hdr_forward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
    pkt_hdr_backward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
    // noc_async_write(payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr),
    // payload_size_bytes);
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        RECORD_FABRIC_HEADER(pkt_hdr_forward);
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
    }

    if (num_targets_backward_direction > 0 && fabric_connection.has_backward_connection()) {
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        RECORD_FABRIC_HEADER(pkt_hdr_backward);
        fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
    }
    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(4);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(7);
constexpr bool dynamic_alternate = get_compile_time_arg_val(8);

constexpr bool is_termination_master = get_compile_time_arg_val(9);
constexpr uint8_t fabric_mux_x = get_compile_time_arg_val(10);
constexpr uint8_t fabric_mux_y = get_compile_time_arg_val(11);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(12);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(13);
constexpr size_t fabric_mux_channel_base_address = get_compile_time_arg_val(14);
constexpr size_t fabric_mux_connection_info_address = get_compile_time_arg_val(15);
constexpr size_t fabric_mux_connection_handshake_address = get_compile_time_arg_val(16);
constexpr size_t fabric_mux_flow_control_address = get_compile_time_arg_val(17);
constexpr size_t fabric_mux_buffer_index_address = get_compile_time_arg_val(18);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(19);
constexpr uint8_t fabric_mux_channel_id = get_compile_time_arg_val(20);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(21);

constexpr ccl_routing_utils::line_multicast_route_info_t multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<22>();

constexpr uint32_t num_max_targets = std::max(num_targets_forward_direction, num_targets_backward_direction);
constexpr uint32_t num_sync_targets_forward = dynamic_alternate ? num_max_targets : num_targets_forward_direction;
constexpr uint32_t num_sync_targets_backward = dynamic_alternate ? num_max_targets : num_targets_backward_direction;

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    size_t arg_for_fab = arg_idx;
    constexpr bool connect_to_fabric_when_creating = true;

    bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_mux_clients = get_arg_val<uint32_t>(arg_idx++);

    // Set up fabric mux connection
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

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers
    // NEW IMPLEMENTATION
    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
    pkt_hdr->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});

    // OLD IMPLEMENTATION
    // pkt_hdr_forward->to_chip_multicast(
    //     tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
    // pkt_hdr_backward->to_chip_multicast(
    //     tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});

    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_connect(*mux_connection_handle);
    }

    // 1. mcast via fabric to remote tensor addresses
    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;

    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core = std::min(num_tiles_per_core - shard_tile_id, packet_size_in_pages);
        num_tiles_to_read_this_core = std::min(num_tiles_to_read - tiles_read, num_tiles_to_read_this_core);
        cb_wait_front(cb0_id, num_tiles_to_read_this_core);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        uint64_t noc0_dest_noc_addr =
            get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0, 0 /*noc_id*/);
        noc0_dest_noc_addr += shard_tile_id * tensor0_page_size;
        uint32_t payload_size_bytes = num_tiles_to_read_this_core * tensor0_page_size;
        write_for_fabric_write(noc0_dest_noc_addr, pkt_hdr, *mux_connection_handle, l1_read_addr, payload_size_bytes);
        l1_read_addr += payload_size_bytes;
        cb_pop_front(cb0_id, num_tiles_to_read_this_core);
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id += num_tiles_to_read_this_core;
        if (shard_tile_id >= num_tiles_per_core) {
            shard_tile_id = 0;
            core_id++;
        }
    }

    // NEW implementation (with fabric mux)
    // 2. mcast output ready semaphore
    auto* pkt_hdr_sem_inc = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);
    pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, static_cast<uint16_t>(1), 32});
    pkt_hdr_sem_inc->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_sync_targets_forward)});
    tt::tt_fabric::fabric_atomic_inc(*mux_connection_handle, pkt_hdr_sem_inc);

    noc_async_writes_flushed();

    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);
        if constexpr (is_termination_master) {
            auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
            noc_semaphore_wait(termination_sync_ptr, 1);
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

//// WORKER WRITE OLD
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// #include "dataflow_api.h"
// #include <tt-metalium/buffer_types.hpp>
// #include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
// #include "tt_metal/fabric/hw/inc/noc_addr.h"
// #include "cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
// #include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
// #include <cstdint>
// #include <utility>

// using address_t = uint32_t;
// FORCE_INLINE void advance_local_read_address_for_fabric_write(
//     uint64_t noc0_dest_noc_addr,
//     volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
//     volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
//     uint32_t num_targets_backward_direction,
//     FabricConnectionManager& fabric_connection,
//     size_t& l1_read_addr,
//     uint32_t payload_size_bytes) {
//     // const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
//     const size_t payload_l1_address = l1_read_addr;
//     pkt_hdr_forward->to_noc_unicast_write(
//         tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
//     pkt_hdr_backward->to_noc_unicast_write(
//         tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

//     // noc_async_write(payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr),
//     // payload_size_bytes);
//     if (fabric_connection.has_forward_connection()) {
//         fabric_connection.get_forward_connection().wait_for_empty_write_slot();
//         RECORD_FABRIC_HEADER(pkt_hdr_forward);
//         fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
//             l1_read_addr, payload_size_bytes);
//         fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
//             (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
//     }

//     if (num_targets_backward_direction > 0 && fabric_connection.has_backward_connection()) {
//         fabric_connection.get_backward_connection().wait_for_empty_write_slot();
//         RECORD_FABRIC_HEADER(pkt_hdr_backward);
//         fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
//             l1_read_addr, payload_size_bytes);
//         fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
//             (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
//     }

//     noc_async_writes_flushed();

//     l1_read_addr += payload_size_bytes;
// }

// ///////////////////////////////////////////////////
// // COMPILE TIME ARGS
// ///////////////////////////////////////////////////

// constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
// constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
// constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
// constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
// constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(4);
// constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(5);
// constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(6);
// constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(7);
// constexpr bool dynamic_alternate = get_compile_time_arg_val(8);
// constexpr uint32_t num_max_targets = std::max(num_targets_forward_direction, num_targets_backward_direction);
// constexpr uint32_t num_sync_targets_forward = dynamic_alternate ? num_max_targets : num_targets_forward_direction;
// constexpr uint32_t num_sync_targets_backward = dynamic_alternate ? num_max_targets : num_targets_backward_direction;

// /*
//  * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
//  * dispatch implementations depending on those invocation parameters.
//  */
// void kernel_main() {
//     ///////////////////////////////////////////////////
//     // ARGS
//     ///////////////////////////////////////////////////

//     size_t arg_idx = 0;
//     // Load the input tensor spec
//     address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
//     const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
//     uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
//     uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
//     uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
//     uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
//     const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
//     const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

//     tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
//     arg_idx += num_cores;
//     tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
//     arg_idx += num_cores;
//     size_t arg_for_fab = arg_idx;
//     constexpr bool connect_to_fabric_when_creating = true;
//     auto fabric_connection =
//         FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
//             arg_idx);

//     // packet header cb
//     cb_reserve_back(reserved_packet_header_cb_id, 1);
//     auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
//     cb_push_back(reserved_packet_header_cb_id, 1);
//     cb_reserve_back(reserved_packet_header_cb_id, 1);
//     auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
//     cb_push_back(reserved_packet_header_cb_id, 1);
//     cb_reserve_back(reserved_packet_header_cb_id, 1);
//     auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
//     cb_push_back(reserved_packet_header_cb_id, 1);

//     // pre-populate packet headers
//     volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
//         reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
//     volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
//         reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
//     pkt_hdr_forward->to_chip_multicast(
//         tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
//     pkt_hdr_backward->to_chip_multicast(
//         tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});

//     fabric_connection.open_finish();

//     // 1. mcast via fabric to remote tensor addresses
//     uint32_t tiles_read = 0;
//     uint32_t shard_tile_id = first_core_tile_start_offset;
//     uint32_t core_id = 0;
//     while (tiles_read < num_tiles_to_read) {
//         uint32_t num_tiles_to_read_this_core = std::min(num_tiles_per_core - shard_tile_id, packet_size_in_pages);
//         num_tiles_to_read_this_core = std::min(num_tiles_to_read - tiles_read, num_tiles_to_read_this_core);
//         cb_wait_front(cb0_id, num_tiles_to_read_this_core);
//         size_t l1_read_addr = get_read_ptr(cb0_id);

//         uint64_t noc0_dest_noc_addr =
//             get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0, 0 /*noc_id*/);
//         noc0_dest_noc_addr += shard_tile_id * tensor0_page_size;

//         // This issues a flush barrier
//         advance_local_read_address_for_fabric_write(
//             noc0_dest_noc_addr,
//             pkt_hdr_forward,
//             pkt_hdr_backward,
//             num_targets_backward_direction,
//             fabric_connection,
//             l1_read_addr,
//             num_tiles_to_read_this_core * tensor0_page_size);
//         if constexpr (dynamic_alternate) {
//             std::swap(
//                 pkt_hdr_forward->routing_fields.value,
//                 pkt_hdr_backward->routing_fields.value);  // alternate the packet header distance for better
//                 balancing
//         }

//         cb_pop_front(cb0_id, num_tiles_to_read_this_core);
//         tiles_read += num_tiles_to_read_this_core;
//         shard_tile_id += num_tiles_to_read_this_core;
//         if (shard_tile_id >= num_tiles_per_core) {
//             shard_tile_id = 0;
//             core_id++;
//         }
//     }

//     // 2. mcast output ready semaphore

//     auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
//     uint64_t out_ready_sem_noc_addr_in_pkt =
//         safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);
//     pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
//         out_ready_sem_noc_addr_in_pkt,
//         static_cast<uint16_t>(1),  // increment 1
//         32});
//     // Write the mcast packet (forward)
//     if (fabric_connection.has_forward_connection()) {
//         fabric_connection.get_forward_connection().wait_for_empty_write_slot();
//         pkt_hdr->to_chip_multicast(
//             tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_sync_targets_forward)});
//         fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
//             packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
//     }
//     // Write the mcast packet (backward)
//     if (num_targets_backward_direction > 0 && fabric_connection.has_backward_connection()) {
//         pkt_hdr->to_chip_multicast(
//             tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_sync_targets_backward)});
//         fabric_connection.get_backward_connection().wait_for_empty_write_slot();
//         fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
//             packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
//     }

//     fabric_connection.close();
//     noc_async_write_barrier();
// }
