// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;

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
constexpr uint32_t num_max_targets = std::max(num_targets_forward_direction, num_targets_backward_direction);
constexpr uint32_t num_sync_targets_forward = dynamic_alternate ? num_max_targets : num_targets_forward_direction;
constexpr uint32_t num_sync_targets_backward = dynamic_alternate ? num_max_targets : num_targets_backward_direction;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    uint32_t reduction_input_cb_id = get_arg_val<address_t>(arg_idx++);
    address_t reduction_input_addr = get_write_ptr(reduction_input_cb_id);

    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_mcast_cores = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t reduction_semaphore_send_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t num_mcast_ranges = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t link = get_arg_val<uint32_t>(arg_idx++);

    // Set up for mcasting to reduction workers
    volatile tt_l1_ptr uint32_t* reduction_semaphore_send_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduction_semaphore_send_addr);
    noc_semaphore_set(reduction_semaphore_send_addr_ptr, VALID);

    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;

    tt_l1_ptr uint32_t* mcast_dest_noc_start_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_mcast_ranges;
    tt_l1_ptr uint32_t* mcast_dest_noc_start_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_mcast_ranges;
    tt_l1_ptr uint32_t* mcast_dest_noc_end_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_mcast_ranges;
    tt_l1_ptr uint32_t* mcast_dest_noc_end_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_mcast_ranges;

    size_t arg_for_fab = arg_idx;
    // Open the fabric connection at the same time we build but only start the "open" process and don't
    // require it to finish before exiting because open_finish requires a barrier which can impact performance
    // We do open_finish later - deferred as late as possible
    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    pkt_hdr_forward->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
    pkt_hdr_backward->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open_finish();
    }

    // 1. mcast via fabric to remote tensor addresses
    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    uint32_t writer_chip_offset = my_chip_id * num_tiles_per_core * tensor0_page_size;

    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core = std::min(num_tiles_per_core - shard_tile_id, packet_size_in_pages);
        num_tiles_to_read_this_core = std::min(num_tiles_to_read - tiles_read, num_tiles_to_read_this_core);
        cb_wait_front(cb0_id, num_tiles_to_read_this_core);
        size_t l1_read_addr = get_read_ptr(cb0_id);

        uint64_t noc0_dest_noc_addr =
            safe_get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], reduction_input_addr + writer_chip_offset);

        uint64_t sema_noc_addr = safe_get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], out_ready_sem_bank_addr);

        // Within-shard offset
        noc0_dest_noc_addr += shard_tile_id * tensor0_page_size;

        // This issues a flush barrier
        if (shard_tile_id + num_tiles_to_read_this_core >= num_tiles_per_core ||
            tiles_read + num_tiles_to_read_this_core >= num_tiles_to_read) {
            fused_write_atomic_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                num_tiles_to_read_this_core * tensor0_page_size,
                sema_noc_addr,
                static_cast<uint16_t>(1),
                static_cast<uint16_t>(32),
                false);
            noc_async_writes_flushed();
        } else {
            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                num_tiles_to_read_this_core * tensor0_page_size);
        }

        if constexpr (dynamic_alternate) {
            std::swap(
                pkt_hdr_forward->routing_fields.value,
                pkt_hdr_backward->routing_fields.value);  // alternate the packet header distance for better balancing
        }

        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id += num_tiles_to_read_this_core;
        if (shard_tile_id >= num_tiles_per_core) {
            shard_tile_id = 0;
            core_id++;
        }
        cb_pop_front(cb0_id, num_tiles_to_read_this_core);
    }

    // 2. local semaphore increment
    for (uint32_t i = 0; i < core_id; i++) {
        noc_semaphore_inc(safe_get_noc_addr(core_noc_x[i], core_noc_y[i], out_ready_sem_bank_addr), 1);
    }
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_start();
    }
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_finish();
    }

    noc_async_write_barrier();
}
