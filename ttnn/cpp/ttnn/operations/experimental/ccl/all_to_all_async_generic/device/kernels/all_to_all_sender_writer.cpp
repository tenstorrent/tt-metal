// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ckernel.h"
#include <cstdint>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t current_device_id = get_compile_time_arg_val(1);
constexpr uint32_t num_devices = get_compile_time_arg_val(2);
constexpr uint32_t concat_dim_size = get_compile_time_arg_val(3);
constexpr uint32_t inner_dims_size = get_compile_time_arg_val(4);
constexpr uint32_t has_half_tile = get_compile_time_arg_val(5);
constexpr uint32_t output_page_size = get_compile_time_arg_val(6);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(7);
constexpr uint32_t semaphore_expected_value = get_compile_time_arg_val(8);
constexpr uint32_t concat_num_tiles = get_compile_time_arg_val(9);
constexpr uint32_t full_block_offset = get_compile_time_arg_val(10);

inline tt::tt_fabric::WorkerToFabricEdmSender& select_connection(
    FabricConnectionManager& fabric_connection, int device_offset) {
    return (device_offset > 0) ? fabric_connection.get_forward_connection()
                               : fabric_connection.get_backward_connection();
}

void write_data(
    uint64_t dest_addrs[4],
    uint16_t payload_sizes[4],
    uint32_t parts_count,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    FabricConnectionManager& fabric_connection,
    size_t l1_read_addr,
    uint64_t output_semaphore_noc_addr_in_pkt,
    int device_offset,
    bool last) {
    bool local = device_offset == 0;
    if (local) {
        if (last) {
            noc_semaphore_inc(output_semaphore_noc_addr_in_pkt, 1);
        }
        for (uint32_t part = 0; part < parts_count; ++part) {
            noc_async_write(l1_read_addr, dest_addrs[part], payload_sizes[part]);
            l1_read_addr += payload_sizes[part];
        }
        noc_async_write_barrier();
    } else {
        if (last) {
            // TODO: reduce number of packages when atomic fused with scatter will be introduced
            if (parts_count > 1) {
                if (parts_count > 2) {
                    uint32_t scatter_payload = 0;
                    for (uint32_t part = 0; part < parts_count - 1; ++part) {
                        scatter_payload += payload_sizes[part];
                    }
                    pkt_hdr->to_noc_unicast_scatter_write(
                        NocUnicastScatterCommandHeader(dest_addrs, payload_sizes, parts_count - 1), scatter_payload);
                    perform_payload_send(
                        select_connection(fabric_connection, device_offset), l1_read_addr, scatter_payload, pkt_hdr);
                    l1_read_addr += scatter_payload;
                } else {
                    pkt_hdr->to_noc_unicast_write(NocUnicastCommandHeader({dest_addrs[0]}), payload_sizes[0]);
                    perform_payload_send(
                        select_connection(fabric_connection, device_offset), l1_read_addr, payload_sizes[0], pkt_hdr);
                    l1_read_addr += payload_sizes[0];
                }
                noc_async_writes_flushed();
            }

            pkt_hdr->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader(
                    {dest_addrs[parts_count - 1], output_semaphore_noc_addr_in_pkt, 1, false}),
                payload_sizes[parts_count - 1]);
            perform_payload_send(
                select_connection(fabric_connection, device_offset),
                l1_read_addr,
                payload_sizes[parts_count - 1],
                pkt_hdr);
        } else {
            uint32_t scatter_payload = 0;
            for (uint32_t part = 0; part < parts_count; ++part) {
                scatter_payload += payload_sizes[part];
            }
            pkt_hdr->to_noc_unicast_scatter_write(
                NocUnicastScatterCommandHeader(dest_addrs, payload_sizes, parts_count), scatter_payload);
            perform_payload_send(
                select_connection(fabric_connection, device_offset), l1_read_addr, scatter_payload, pkt_hdr);
        }
    }
    noc_async_writes_flushed();
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    uint32_t global_init_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t global_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);

    uint32_t core_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t link_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t mcast_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t sender_core_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t sender_core_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_num_devices = get_arg_val<uint32_t>(arg_idx++);
    constexpr auto output_args = TensorAccessorArgs<11>();
    auto output_addrgen = TensorAccessor(output_args, output_address, output_page_size);
    size_t device_offsets_idx = arg_idx;
    arg_idx += local_num_devices * 3;

    // Build fabric connection
    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);
    uint64_t init_semaphore_noc_addr_in_pkt =
        safe_get_noc_addr(sender_core_x, sender_core_y, global_init_semaphore_addr);
    uint64_t output_semaphore_noc_addr_in_pkt = safe_get_noc_addr(sender_core_x, sender_core_y, global_semaphore_addr);

    volatile tt_l1_ptr uint32_t* global_init_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_init_semaphore_addr);
    volatile tt_l1_ptr uint32_t* global_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_sema_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_sema_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_sema_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_sema_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_sema_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_sema_backward);

    fabric_connection.open_finish();

    constexpr bool has_pre_half_tile = has_half_tile && current_device_id % 2 == 1;
    constexpr bool has_post_half_tile = has_half_tile && current_device_id % 2 == 0;

    if (link_id == 0) {
#define LINEAR 1
#define RING 2
#if TOPOLOGY == LINEAR
        if (fabric_connection.has_forward_connection()) {
            pkt_hdr_sema_forward->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                init_semaphore_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr_sema_forward->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                1, static_cast<uint8_t>(num_devices - current_device_id - 1)});
            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_addr_sema_forward, sizeof(PACKET_HEADER_TYPE));
        }

        if (fabric_connection.has_backward_connection()) {
            pkt_hdr_sema_backward->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                init_semaphore_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1
            pkt_hdr_sema_backward->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(current_device_id)});
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
                packet_header_buffer_addr_sema_backward, sizeof(PACKET_HEADER_TYPE));
        }
#elif TOPOLOGY == RING
        if (fabric_connection.has_forward_connection()) {
            pkt_hdr_forward->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                init_semaphore_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr_forward->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_devices / 2)});
            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_addr_forward, sizeof(PACKET_HEADER_TYPE));
        }
        if (fabric_connection.has_backward_connection()) {
            pkt_hdr_backward->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                init_semaphore_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            pkt_hdr_backward->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                1, static_cast<uint8_t>(num_devices - num_devices / 2 - 1)});
            fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_addr_backward, sizeof(PACKET_HEADER_TYPE));
        }
#else
#error "Unsupported Topology Type"
#endif
    }
    if (core_id == 0 && link_id == 0) {
        const uint64_t local_set_semaphore_noc_addr = get_noc_multicast_addr(
            mcast_dest_noc_start_x,
            mcast_dest_noc_start_y,
            mcast_dest_noc_end_x,
            mcast_dest_noc_end_y,
            global_init_semaphore_addr);
        uint32_t local_init_semaphore_addr_ptr = reinterpret_cast<uint32_t>(global_init_semaphore_addr_ptr);

        if (mcast_size > 1) {
            noc_semaphore_wait(global_init_semaphore_addr_ptr, num_devices - 1);
            noc_semaphore_set_multicast_loopback_src(
                local_init_semaphore_addr_ptr, local_set_semaphore_noc_addr, mcast_size, false);
            noc_async_write_barrier();
        }
    }

    noc_semaphore_wait(global_init_semaphore_addr_ptr, num_devices - 1);
    noc_semaphore_set(global_init_semaphore_addr_ptr, 0);

    for (uint32_t did = 0; did < local_num_devices; ++did) {
        int32_t device_offset = get_arg_val<int32_t>(device_offsets_idx++);
        int32_t distance = abs(device_offset);
        uint32_t device_id = (current_device_id + device_offset + num_devices) % num_devices;

        volatile PACKET_HEADER_TYPE* pkt_hdr;
        if (device_offset > 0) {
            pkt_hdr = pkt_hdr_forward;
            pkt_hdr->to_chip_unicast(distance);
        }
        if (device_offset < 0) {
            pkt_hdr = pkt_hdr_backward;
            pkt_hdr->to_chip_unicast(distance);
        }

        auto calculate_params = [&](int b) {
            const uint32_t o = b / (concat_num_tiles * inner_dims_size);
            const uint32_t c = (b / inner_dims_size) % concat_num_tiles;
            const uint32_t i = b % inner_dims_size;
            const uint32_t dest_tile_id =
                o * inner_dims_size * concat_dim_size + (c + full_block_offset) * inner_dims_size + i;
            uint16_t payload_size = ((has_pre_half_tile && c == 0) || (has_post_half_tile && c == concat_num_tiles - 1))
                                        ? output_page_size / 2
                                        : output_page_size;
            uint32_t offset = (has_pre_half_tile && c == 0) ? (current_device_id % 2) * output_page_size / 2 : 0;
            uint64_t dst_addr =
                (current_device_id == device_id)
                    ? output_addrgen.get_noc_addr(dest_tile_id, offset)
                    : tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, dest_tile_id, offset);
            return std::tuple{dst_addr, payload_size};
        };

        uint32_t block_idx = get_arg_val<uint32_t>(device_offsets_idx++);
        uint32_t block_end_id = get_arg_val<uint32_t>(device_offsets_idx++);

        cb_wait_front(cb0_id, 1);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        uint32_t current_package_payload = 0;
        uint32_t current_tile = 0;
        uint64_t dst_addrs[4] = {0};
        uint16_t payload_sizes[4] = {0};
        while (block_idx < block_end_id) {
            auto [dst_addr, payload_size] = calculate_params(block_idx);
            // If package is full, flush and start new package
            if (current_tile == 4 || current_package_payload + payload_size > 2 * output_page_size) {
                write_data(
                    dst_addrs,
                    payload_sizes,
                    current_tile,
                    pkt_hdr,
                    fabric_connection,
                    l1_read_addr,
                    output_semaphore_noc_addr_in_pkt,
                    device_offset,
                    false);
                cb_pop_front(cb0_id, 1);
                cb_wait_front(cb0_id, 1);
                l1_read_addr = get_read_ptr(cb0_id);
                current_package_payload = 0;
                current_tile = 0;
            }
            current_package_payload += payload_size;
            payload_sizes[current_tile] = payload_size;
            dst_addrs[current_tile] = dst_addr;
            current_tile++;
            block_idx++;
        }

        // Flush remaining tiles in the last package
        if (current_tile > 0) {
            write_data(
                dst_addrs,
                payload_sizes,
                current_tile,
                pkt_hdr,
                fabric_connection,
                l1_read_addr,
                output_semaphore_noc_addr_in_pkt,
                device_offset,
                true);
        }
        cb_pop_front(cb0_id, 1);
    }

    noc_async_write_barrier();
    fabric_connection.close();
    if (core_id == 0 && link_id == 0) {
        noc_semaphore_wait(global_semaphore_addr_ptr, semaphore_expected_value);
        noc_semaphore_set(global_semaphore_addr_ptr, 0);
    }
}
