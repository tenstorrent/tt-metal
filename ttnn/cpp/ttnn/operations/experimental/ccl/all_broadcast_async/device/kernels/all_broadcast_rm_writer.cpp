// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(1);
constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(2));
constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t row_size = get_compile_time_arg_val(5);
constexpr uint32_t max_packet_size = get_compile_time_arg_val(6);
constexpr uint32_t num_packets_per_row = get_compile_time_arg_val(7);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(8);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(9);
constexpr bool dynamic_alternate = get_compile_time_arg_val(10);
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
    const address_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t row_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t row_id_end = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;

#ifdef SHARDED
    typedef ShardedInfo<
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(16),
        get_compile_time_arg_val(17)>
        tensor_shard_info;

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<tensor_shard_info> tensor0_addrgen = {
        .bank_base_address = tensor_address0, .shard_array = mapping_table};
    size_t fab_idx = arg_for_fab + rt_increment;
    auto fabric_connection = FabricConnectionManager::build_from_args(fab_idx);
#else
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    const auto tensor0_addrgen = get_interleaved_addr_gen<is_dram, row_size>(tensor_address0);
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);

#endif

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
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    pkt_hdr_forward->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
    pkt_hdr_backward->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    // 1. mcast via fabric to remote tensor addresses
    uint32_t row_id = row_id_start;
    while (row_id < row_id_end) {
        size_t l1_read_addr = get_read_ptr(cb0_id);
        cb_wait_front(cb0_id, 1);

        uint32_t offset = 0;

        for (uint32_t j = 0; j < num_packets_per_row; j++) {
            uint64_t noc0_dest_noc_addr = get_noc_addr(row_id, tensor0_addrgen, offset, 0);

            uint32_t packet_size = std::min(max_packet_size, page_size);
            packet_size = std::min(packet_size, page_size - max_packet_size * j);

            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr, pkt_hdr_forward, pkt_hdr_backward, fabric_connection, l1_read_addr, packet_size);
            if constexpr (dynamic_alternate) {
                std::swap(
                    pkt_hdr_forward->routing_fields.value,
                    pkt_hdr_backward->routing_fields
                        .value);  // alternate the packet header distance for better balancing
            }
            offset += packet_size;  // advance the noc address for the next packet
        }
        row_id++;
        cb_pop_front(cb0_id, 1);
    }

    // 2. mcast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);
    auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
    pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt,
        static_cast<uint16_t>(1),  // increment 1
        32});
    // Write the mcast packet (forward)
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        pkt_hdr->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_sync_targets_forward)});
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // Write the mcast packet (backward)
    if (fabric_connection.has_backward_connection()) {
        pkt_hdr->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_sync_targets_backward)});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // increment locally
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);

    // 3. wait for mcast output ready semaphore
    if (wait_output_semaphore) {
        volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);
        noc_semaphore_wait(sem_ptr, out_ready_sem_wait_value);
    }

    // 4. global semaphore reset
    if (reset_global_semaphore) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), 0);
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
}
