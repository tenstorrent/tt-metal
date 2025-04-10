// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/buffer_constants.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include <cstdint>
#include <utility>
void kernel_main() {
    constexpr bool is_all_to_all_worker = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_in_2 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in_4 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_to_allgather_writer = get_compile_time_arg_val(3);
    // Todo add these CBs
    constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(5);
    constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(7);
    constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(8);
    constexpr uint32_t num_links = get_compile_time_arg_val(9);
    size_t arg_idx = 0;
    const uint32_t scalar_w = get_arg_val<uint32_t>(arg_idx++);
    wh_generate_reduce_scaler<true>(cb_in_2, scalar_w);

    if constexpr (is_all_to_all_worker) {
        const uint32_t scalar_c = get_arg_val<uint32_t>(arg_idx++);
        wh_generate_reduce_scaler<true>(cb_in_4, scalar_c);
    } else {
        arg_idx++;
    }

    const uint32_t iteration_number = get_arg_val<uint32_t>(arg_idx++);
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    ttnn::ccl::address_t tensor_address0 = get_arg_val<ttnn::ccl::address_t>(arg_idx++);

    // Start the all gather part
    if (iteration_number < num_links) {
        // Do this only on one of the cores

        // To do add these to Program Factory on i=0 case
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
        auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
        cb_push_back(reserved_packet_header_cb_id, 1);
        // pre-populate packet headers
        volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
        volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
        pkt_hdr_forward->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
        pkt_hdr_backward->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});
        fabric_connection.open_finish();
        // 1. mcast via fabric to remote tensor addresses
        uint32_t tiles_read = 0;
        uint32_t shard_tile_id = first_core_tile_start_offset;
        uint32_t core_id = 0;
        while (tiles_read < num_tiles_to_read) {
            uint32_t num_tiles_to_read_this_core = std::min(num_tiles_per_core - shard_tile_id, packet_size_in_pages);
            num_tiles_to_read_this_core = std::min(num_tiles_to_read - tiles_read, num_tiles_to_read_this_core);
            cb_wait_front(cb_to_allgather_writer, num_tiles_to_read_this_core);
            size_t l1_read_addr = get_read_ptr(cb_to_allgather_writer);

            uint64_t noc0_dest_noc_addr =
                get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0, 0 /*noc_id*/);
            noc0_dest_noc_addr += shard_tile_id * tensor0_page_size;
            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                num_tiles_to_read_this_core * tensor0_page_size);
            noc_async_writes_flushed();
            cb_pop_front(cb_to_allgather_writer, num_tiles_to_read_this_core);
            tiles_read += num_tiles_to_read_this_core;
            shard_tile_id += num_tiles_to_read_this_core;
            if (shard_tile_id >= num_tiles_per_core) {
                shard_tile_id = 0;
                core_id++;
            }
        }
        // 2. mcast output ready semaphore
        auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
        uint64_t out_ready_sem_noc_addr_in_pkt =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);
        pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            out_ready_sem_noc_addr_in_pkt,
            static_cast<uint16_t>(1),  // increment 1
            32});
        // Write the mcast packet (forward)
        if (fabric_connection.has_forward_connection()) {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }
        // Write the mcast packet (backward)
        if (fabric_connection.has_backward_connection()) {
            pkt_hdr->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }
        fabric_connection.close_start();
        // increment locally
        uint64_t out_ready_sem_noc_addr =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
        if constexpr (num_links == 1) {
            // We deduct the local increment from the target semaphore value as we don't need internal synchronization
            noc_semaphore_wait(
                reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addr), out_ready_sem_wait_value - 1);
        } else {
            // if multiple links then we need to also ensure the local ones have completed by having them also
            // increment the semaphore and including them in the total
            noc_semaphore_inc(out_ready_sem_noc_addr, 1);
            noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addr), out_ready_sem_wait_value);
        }

        // 4. global semaphore reset
        if (iteration_number == 0) {
            *reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addr) = 0;
        }
        fabric_connection.close_finish();  // Includes a noc async write barrier
    }
}
