// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include <array>

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
constexpr uint32_t num_sem_ranges = get_compile_time_arg_val(8);
constexpr uint32_t out_ready_sem_wait_value = get_compile_time_arg_val(9);

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
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    constexpr uint8_t out_ready_sem_noc0_x = 19;       // get_arg_val<uint32_t>(arg_idx++);
    constexpr uint8_t out_ready_sem_noc0_y = 18;       // get_arg_val<uint32_t>(arg_idx++);

    const uint32_t concat_semaphore_send_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t concat_semaphore_send_addr2 = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;

    tt_l1_ptr uint32_t* mcast_dest_noc_start_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_sem_ranges;
    tt_l1_ptr uint32_t* mcast_dest_noc_start_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_sem_ranges;
    tt_l1_ptr uint32_t* mcast_dest_noc_end_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_sem_ranges;
    tt_l1_ptr uint32_t* mcast_dest_noc_end_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_sem_ranges;

    size_t arg_for_fab = arg_idx;
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
    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core = std::min(num_tiles_per_core - shard_tile_id, packet_size_in_pages);
        num_tiles_to_read_this_core = std::min(num_tiles_to_read - tiles_read, num_tiles_to_read_this_core);
        cb_wait_front(cb0_id, num_tiles_to_read_this_core);
        size_t l1_read_addr = get_read_ptr(cb0_id);

        uint64_t noc0_dest_noc_addr = get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0, 0);
        noc0_dest_noc_addr += shard_tile_id * tensor0_page_size;

        write_and_advance_local_read_address_for_fabric_write(
            noc0_dest_noc_addr,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection,
            l1_read_addr,
            num_tiles_to_read_this_core * tensor0_page_size);

        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id += num_tiles_to_read_this_core;
        if (shard_tile_id >= num_tiles_per_core) {
            shard_tile_id = 0;
            core_id++;
        }

        noc_async_writes_flushed();
        cb_pop_front(cb0_id, num_tiles_to_read_this_core);
    }

    // 2. mcast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);
    // Write the mcast packet (forward)
    if (fabric_connection.has_forward_connection()) {
        auto* pkt_hdr_fwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
        pkt_hdr_fwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            out_ready_sem_noc_addr_in_pkt,
            static_cast<uint16_t>(1),  // increment 1
            32});

        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        pkt_hdr_fwd->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            reinterpret_cast<uint32_t>(pkt_hdr_fwd), sizeof(PACKET_HEADER_TYPE));
    }
    // Write the mcast packet (backward)
    if (fabric_connection.has_backward_connection()) {
        auto* pkt_hdr_bwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
        pkt_hdr_bwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            out_ready_sem_noc_addr_in_pkt,
            static_cast<uint16_t>(1),  // increment 1
            32});

        pkt_hdr_bwd->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
            reinterpret_cast<uint32_t>(pkt_hdr_bwd), sizeof(PACKET_HEADER_TYPE));
    }
    // increment locally
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);

    // 3. wait for mcast output ready semaphore
    if (wait_output_semaphore) {
        while (*reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addr) != out_ready_sem_wait_value);
    }

    // Set up for mcasting to concat workers
    if (wait_output_semaphore) {
        volatile tt_l1_ptr uint32_t* concat_semaphore_send_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(concat_semaphore_send_addr);
        volatile tt_l1_ptr uint32_t* concat_semaphore_send_addr_ptr2 =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(concat_semaphore_send_addr2);

        noc_semaphore_set(concat_semaphore_send_addr_ptr, 1);
        noc_semaphore_set(concat_semaphore_send_addr_ptr2, 1);
        if (concat_semaphore_send_addr_ptr[0] != 1) {
            noc_semaphore_set(concat_semaphore_send_addr_ptr, 1);
        }
        if (concat_semaphore_send_addr_ptr2[0] != 1) {
            noc_semaphore_set(concat_semaphore_send_addr_ptr2, 1);
        }
    }

    if (wait_output_semaphore) {
        for (uint32_t i = 0; i < 3; i++) {
            uint32_t mcast_dest_num = 2;
            if (i == 1) {
                mcast_dest_num = 6;
            } else if (i == 2) {
                mcast_dest_num = 8;
            }
            const uint64_t concat_sem_rcv_addr = get_noc_multicast_addr(
                mcast_dest_noc_start_x[i],
                mcast_dest_noc_start_y[i],
                mcast_dest_noc_end_x[i],
                mcast_dest_noc_end_y[i],
                concat_semaphore_send_addr);
            noc_semaphore_set_multicast(
                concat_semaphore_send_addr,
                concat_sem_rcv_addr,
                mcast_dest_num,
                false,  // linked = false
                true);  // multicast_path_reserve = true

            const uint64_t concat_sem_rcv_addr2 = get_noc_multicast_addr(
                mcast_dest_noc_start_x[i],
                mcast_dest_noc_start_y[i],
                mcast_dest_noc_end_x[i],
                mcast_dest_noc_end_y[i],
                concat_semaphore_send_addr2);

            noc_semaphore_set_multicast(
                concat_semaphore_send_addr2,
                concat_sem_rcv_addr2,
                mcast_dest_num,
                false,  // linked = false
                true);  // multicast_path_reserve = true
        }
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_start();
    }

    if (reset_global_semaphore) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr) = 0;
    }

    // noc_async_read_barrier();
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_finish();
    }
    noc_async_write_barrier();
}
