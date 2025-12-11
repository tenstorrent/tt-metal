// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
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
constexpr uint32_t outer_dims_size = get_compile_time_arg_val(3);
constexpr uint32_t concat_dim_size = get_compile_time_arg_val(4);
constexpr uint32_t inner_dims_size = get_compile_time_arg_val(5);
constexpr uint32_t number_pages_per_packet = get_compile_time_arg_val(6);
constexpr uint32_t has_half_tile = get_compile_time_arg_val(7);
constexpr uint32_t output_page_size = get_compile_time_arg_val(8);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(9);

inline tt::tt_fabric::WorkerToFabricEdmSender& select_connection(
    FabricConnectionManager& fabric_connection, int device_offset) {
    return (device_offset > 0) ? fabric_connection.get_forward_connection()
                               : fabric_connection.get_backward_connection();
}

template <typename AddrGenType>
void write_data(
    bool last,
    uint32_t dest_id,
    AddrGenType addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    FabricConnectionManager& fabric_connection,
    size_t l1_read_addr,
    uint32_t payload_size_bytes,
    uint32_t offset,
    uint64_t output_semaphore_noc_addr_in_pkt,
    int device_offset) {
    bool local = device_offset == 0;
    if (last) {
        if (local) {
            noc_semaphore_inc(output_semaphore_noc_addr_in_pkt, 1);
            noc_async_write(l1_read_addr, addrgen.get_noc_addr(dest_id, offset), payload_size_bytes);
            noc_async_write_barrier();
        } else {
            perform_atomic_fabric_write(
                pkt_hdr,
                dest_id,
                addrgen,
                select_connection(fabric_connection, device_offset),
                l1_read_addr,
                payload_size_bytes,
                output_semaphore_noc_addr_in_pkt,
                1,
                false,
                offset);
        }
    } else {
        if (local) {
            noc_async_write(l1_read_addr, addrgen.get_noc_addr(dest_id, offset), payload_size_bytes);
        } else {
            tt::tt_fabric::linear::to_noc_unicast_write(payload_size_bytes, pkt_hdr, dest_id, addrgen, offset);
            perform_payload_send(
                select_connection(fabric_connection, device_offset), l1_read_addr, payload_size_bytes, pkt_hdr);
        }
    }
    noc_async_writes_flushed();
}

template <typename AddrGenType>
void write_data(
    bool last,
    uint32_t dest_id0,
    uint32_t dest_id1,
    AddrGenType addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    FabricConnectionManager& fabric_connection,
    size_t l1_read_addr,
    uint32_t payload_size_bytes0,
    uint32_t payload_size_bytes1,
    uint32_t offset0,
    uint32_t offset1,
    uint64_t output_semaphore_noc_addr_in_pkt,
    int device_offset) {
    bool local = device_offset == 0;
    if (last) {
        if (local) {
            noc_semaphore_inc(output_semaphore_noc_addr_in_pkt, 1);
            noc_async_write(l1_read_addr, addrgen.get_noc_addr(dest_id0, offset0), payload_size_bytes0);
            noc_async_write(
                l1_read_addr + output_page_size, addrgen.get_noc_addr(dest_id1, offset1), payload_size_bytes1);
            noc_async_write_barrier();
        } else {
            tt::tt_fabric::linear::to_noc_unicast_write(payload_size_bytes0, pkt_hdr, dest_id0, addrgen, offset0);
            perform_payload_send(
                select_connection(fabric_connection, device_offset), l1_read_addr, payload_size_bytes0, pkt_hdr);
            size_t l1_read_addr_plus_payload = l1_read_addr + output_page_size;
            noc_async_writes_flushed();
            perform_atomic_fabric_write(
                pkt_hdr,
                dest_id1,
                addrgen,
                select_connection(fabric_connection, device_offset),
                l1_read_addr_plus_payload,
                payload_size_bytes1,
                output_semaphore_noc_addr_in_pkt,
                1,
                false,
                offset1);
        }
    } else {
        if (local) {
            noc_async_write(l1_read_addr, addrgen.get_noc_addr(dest_id0, offset0), payload_size_bytes0);
            noc_async_write(
                l1_read_addr + output_page_size, addrgen.get_noc_addr(dest_id1, offset1), payload_size_bytes1);
        } else {
            constexpr uint32_t half_output_page_size = output_page_size / 2;
            if (payload_size_bytes0 == half_output_page_size) {
                tt::data_movement::common::tt_memmove<true, false, false, half_output_page_size>(
                    l1_read_addr + half_output_page_size, l1_read_addr, half_output_page_size);
                l1_read_addr = l1_read_addr + half_output_page_size;
            }
            auto payload_size = payload_size_bytes0 + payload_size_bytes1;

            auto noc_address0 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(addrgen, dest_id0, offset0);
            auto noc_address1 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(addrgen, dest_id1, offset1);

            pkt_hdr->to_noc_unicast_scatter_write(
                NocUnicastScatterCommandHeader(
                    {noc_address0, noc_address1}, {static_cast<uint16_t>(payload_size_bytes0)}),
                payload_size);
            perform_payload_send(
                select_connection(fabric_connection, device_offset), l1_read_addr, payload_size, pkt_hdr);
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
    uint32_t sender_core_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t sender_core_y = get_arg_val<uint32_t>(arg_idx++);
    constexpr auto output_args = TensorAccessorArgs<10>();
    auto output_addrgen = TensorAccessor(output_args, output_address, output_page_size);

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

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    fabric_connection.open_finish();

    constexpr bool has_pre_half_tile = has_half_tile && current_device_id % 2 == 1;
    constexpr bool has_post_half_tile = has_half_tile && current_device_id % 2 == 0;

    constexpr uint32_t concat_num_half_tiles = concat_dim_size * 2 / num_devices;
    constexpr uint32_t concat_num_tiles = (concat_num_half_tiles + 1) / 2;
    constexpr uint32_t full_block_offset = (concat_num_half_tiles * current_device_id) / 2;

#define LINEAR 1
#define RING 2
#if TOPOLOGY == LINEAR
    if (fabric_connection.has_forward_connection()) {
        pkt_hdr_forward->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            init_semaphore_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        pkt_hdr_forward->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_devices - current_device_id - 1)});
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            packet_header_buffer_addr_forward, sizeof(PACKET_HEADER_TYPE));
    }

    if (fabric_connection.has_backward_connection()) {
        pkt_hdr_backward->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            init_semaphore_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1
        pkt_hdr_backward->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(current_device_id)});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
            packet_header_buffer_addr_backward, sizeof(PACKET_HEADER_TYPE));
    }

    noc_semaphore_wait(global_init_semaphore_addr_ptr, num_devices - 1);
    noc_semaphore_set(global_init_semaphore_addr_ptr, 0);

    for (uint32_t device_id = 0; device_id < num_devices; ++device_id) {
        volatile PACKET_HEADER_TYPE* pkt_hdr;
        int device_offset = device_id - current_device_id;
        int distance = abs(device_offset);

#elif TOPOLOGY == RING
    if (fabric_connection.has_forward_connection()) {
        pkt_hdr_forward->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            init_semaphore_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        pkt_hdr_forward->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_devices - 1)});
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            packet_header_buffer_addr_forward, sizeof(PACKET_HEADER_TYPE));
    }

    noc_semaphore_wait(global_init_semaphore_addr_ptr, num_devices - 1);
    noc_semaphore_set(global_init_semaphore_addr_ptr, 0);

    for (int d = num_devices - 1; d >= 0; --d) {
        volatile PACKET_HEADER_TYPE* pkt_hdr;
        int distance = (d + 1) / 2;
        int device_offset = (d % 2 == 0) ? distance : -distance;
        uint32_t device_id = (current_device_id + device_offset + num_devices) % num_devices;

#else
#error "Unsupported Topology Type"
#endif
        if (device_offset > 0) {
            pkt_hdr = pkt_hdr_forward;
            pkt_hdr->to_chip_unicast(distance);
        }
        if (device_offset < 0) {
            pkt_hdr = pkt_hdr_backward;
            pkt_hdr->to_chip_unicast(distance);
        }

        uint64_t block_size = outer_dims_size * concat_num_tiles * inner_dims_size;
        auto calculate_params = [&](int b) {
            const uint32_t o = b / (concat_num_tiles * inner_dims_size);
            const uint32_t c = (b / inner_dims_size) % concat_num_tiles;
            const uint32_t i = b % inner_dims_size;
            const uint32_t dest_tile_id =
                o * inner_dims_size * concat_dim_size + (c + full_block_offset) * inner_dims_size + i;
            bool last = (c == concat_num_tiles - 1) && o == outer_dims_size - 1 && i == inner_dims_size - 1;
            uint32_t payload_size = ((has_pre_half_tile && c == 0) || (has_post_half_tile && c == concat_num_tiles - 1))
                                        ? output_page_size / 2
                                        : output_page_size;
            uint32_t offset = (has_pre_half_tile && c == 0) ? (current_device_id % 2) * output_page_size / 2 : 0;
            return std::tuple{dest_tile_id, last, payload_size, offset};
        };

        for (uint64_t b = 0; b < block_size; b += number_pages_per_packet) {
            uint32_t tiles_this_iteration = std::min(number_pages_per_packet, uint32_t(block_size - b));

            cb_wait_front(cb0_id, tiles_this_iteration);
            size_t l1_read_addr = get_read_ptr(cb0_id);

            if (tiles_this_iteration == 2) {
                auto [dest_tile_id0, last0, payload_size0, offset0] = calculate_params(b);
                auto [dest_tile_id1, last1, payload_size1, offset1] = calculate_params(b + 1);

                write_data(
                    last1,
                    dest_tile_id0,
                    dest_tile_id1,
                    output_addrgen,
                    pkt_hdr,
                    fabric_connection,
                    l1_read_addr,
                    payload_size0,
                    payload_size1,
                    offset0,
                    offset1,
                    output_semaphore_noc_addr_in_pkt,
                    device_offset);
            } else {
                auto [dest_tile_id, last, payload_size, offset] = calculate_params(b);

                write_data(
                    last,
                    dest_tile_id,
                    output_addrgen,
                    pkt_hdr,
                    fabric_connection,
                    l1_read_addr,
                    payload_size,
                    offset,
                    output_semaphore_noc_addr_in_pkt,
                    device_offset);
            }

            cb_pop_front(cb0_id, tiles_this_iteration);
        }
    }

    noc_async_write_barrier();
    fabric_connection.close();

    noc_semaphore_wait(global_semaphore_addr_ptr, num_devices);
    noc_semaphore_set(global_semaphore_addr_ptr, 0);
}
