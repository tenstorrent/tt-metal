// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "cpp/ttnn/operations/ccl/common/types/fabric_directions.hpp"
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
constexpr auto topology = static_cast<tt::tt_fabric::Topology>(get_compile_time_arg_val(11));
constexpr auto replicate_axis =
    static_cast<ttnn::operations::ccl::common::ReplicateGroup>(get_compile_time_arg_val(12));
constexpr uint16_t source_chip_id = get_compile_time_arg_val(13);
constexpr uint16_t source_mesh_id = get_compile_time_arg_val(14);
constexpr bool is_fabric_2d = get_compile_time_arg_val(15);
constexpr uint32_t fabric_direction_mask = get_compile_time_arg_val(16);
constexpr auto fabric_directions =
    ttnn::operations::ccl::common::fabric_direction_mask_to_directions(fabric_direction_mask);
constexpr auto num_fabric_directions = ttnn::operations::ccl::common::num_fabric_directions;
using Fabric2DConnections = std::array<tt::tt_fabric::WorkerToFabricEdmSender, num_fabric_directions>;
using FabricConnections = std::conditional_t<is_fabric_2d, Fabric2DConnections, FabricConnectionManager>;

inline tt::tt_fabric::WorkerToFabricEdmSender& select_connection(
    Fabric2DConnections& fabric_connections,
    [[maybe_unused]] int device_offset,
    uint16_t dest_mesh_id,
    uint16_t dest_chip_id) {
    return fabric_connections[get_next_hop_router_direction(dest_mesh_id, dest_chip_id)];
}

inline tt::tt_fabric::WorkerToFabricEdmSender& select_connection(
    FabricConnectionManager& fabric_connections,
    int device_offset,
    [[maybe_unused]] uint16_t dest_mesh_id,
    [[maybe_unused]] uint16_t dest_chip_id) {
    return (device_offset > 0) ? fabric_connections.get_forward_connection()
                               : fabric_connections.get_backward_connection();
}

inline void finish_open_connections(Fabric2DConnections& fabric_connections) {
    ttnn::operations::ccl::common::open_direction_connections_barrier(fabric_directions, fabric_connections);
}

inline void finish_open_connections(FabricConnectionManager& fabric_connections) { fabric_connections.open_finish(); }

inline void close_connections(Fabric2DConnections& fabric_connections) {
    ttnn::operations::ccl::common::close_direction_connections(fabric_directions, fabric_connections);
}

inline void close_connections(FabricConnectionManager& fabric_connections) { fabric_connections.close(); }

template <typename PacketHeader>
inline void set_route(
    volatile PacketHeader* packet_header,
    const ccl_routing_utils::line_unicast_route_info_t& route_info,
    [[maybe_unused]] int32_t distance,
    Fabric2DConnections&) {
    ccl_routing_utils::fabric_set_line_unicast_route(packet_header, route_info);
}

template <typename PacketHeader>
inline void set_route(
    volatile PacketHeader* packet_header,
    [[maybe_unused]] const ccl_routing_utils::line_unicast_route_info_t& route_info,
    int32_t distance,
    FabricConnectionManager&) {
    packet_header->to_chip_unicast(distance);
}
template <tt::tt_fabric::Topology Topology, typename PacketHeader>
void send_initialization(
    Fabric2DConnections& fabric_connections,
    uint32_t core_id,
    uint32_t link_id,
    [[maybe_unused]] uint32_t local_num_devices,
    [[maybe_unused]] size_t device_offsets_idx,
    uint64_t init_semaphore_noc_addr_in_pkt,
    [[maybe_unused]] volatile PacketHeader* pkt_hdr_forward,
    [[maybe_unused]] volatile PacketHeader* pkt_hdr_backward,
    volatile PacketHeader* pkt_hdr_sema_forward,
    volatile PacketHeader* pkt_hdr_sema_backward,
    [[maybe_unused]] uint32_t packet_header_buffer_addr_forward,
    [[maybe_unused]] uint32_t packet_header_buffer_addr_backward,
    uint32_t packet_header_buffer_addr_sema_forward,
    uint32_t packet_header_buffer_addr_sema_backward) {
    using ttnn::operations::ccl::common::ReplicateGroup;
    static_assert(
        Topology == tt::tt_fabric::Topology::Linear || Topology == tt::tt_fabric::Topology::Ring,
        "FABRIC_2D all-to-all supports only Linear or Ring topology");
    static_assert(
        replicate_axis == ReplicateGroup::COLS || replicate_axis == ReplicateGroup::ROWS,
        "FABRIC_2D all-to-all requires a concrete cluster axis");

    if (core_id != 0 || link_id != 0) {
        return;
    }
    constexpr uint32_t positive_range = num_devices - current_device_id - 1;
    constexpr uint32_t negative_range = current_device_id;
    constexpr uint32_t positive_direction =
        replicate_axis == ReplicateGroup::COLS ? eth_chan_directions::SOUTH : eth_chan_directions::EAST;
    constexpr uint32_t negative_direction =
        replicate_axis == ReplicateGroup::COLS ? eth_chan_directions::NORTH : eth_chan_directions::WEST;

    if constexpr (positive_range > 0) {
        constexpr uint16_t east_range = replicate_axis == ReplicateGroup::ROWS ? positive_range : 0;
        constexpr uint16_t south_range = replicate_axis == ReplicateGroup::COLS ? positive_range : 0;
        tt::tt_fabric::fabric_set_mcast_route(
            pkt_hdr_sema_forward, source_chip_id, source_mesh_id, east_range, 0, 0, south_range);
        pkt_hdr_sema_forward->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_semaphore_noc_addr_in_pkt, 1});
        auto& connection = fabric_connections[positive_direction];
        connection.wait_for_empty_write_slot();
        connection.send_payload_flush_blocking_from_address(
            packet_header_buffer_addr_sema_forward, sizeof(PacketHeader));
    }
    if constexpr (negative_range > 0) {
        constexpr uint16_t west_range = replicate_axis == ReplicateGroup::ROWS ? negative_range : 0;
        constexpr uint16_t north_range = replicate_axis == ReplicateGroup::COLS ? negative_range : 0;
        tt::tt_fabric::fabric_set_mcast_route(
            pkt_hdr_sema_backward, source_chip_id, source_mesh_id, 0, west_range, north_range, 0);
        pkt_hdr_sema_backward->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_semaphore_noc_addr_in_pkt, 1});
        auto& connection = fabric_connections[negative_direction];
        connection.wait_for_empty_write_slot();
        connection.send_payload_flush_blocking_from_address(
            packet_header_buffer_addr_sema_backward, sizeof(PacketHeader));
    }
}

template <tt::tt_fabric::Topology Topology, typename PacketHeader>
void send_initialization(
    FabricConnectionManager& fabric_connections,
    [[maybe_unused]] uint32_t core_id,
    uint32_t link_id,
    [[maybe_unused]] uint32_t local_num_devices,
    [[maybe_unused]] size_t device_offsets_idx,
    uint64_t init_semaphore_noc_addr_in_pkt,
    volatile PacketHeader* pkt_hdr_forward,
    volatile PacketHeader* pkt_hdr_backward,
    volatile PacketHeader* pkt_hdr_sema_forward,
    volatile PacketHeader* pkt_hdr_sema_backward,
    uint32_t packet_header_buffer_addr_forward,
    uint32_t packet_header_buffer_addr_backward,
    uint32_t packet_header_buffer_addr_sema_forward,
    uint32_t packet_header_buffer_addr_sema_backward) {
    static_assert(
        Topology == tt::tt_fabric::Topology::Linear || Topology == tt::tt_fabric::Topology::Ring,
        "all_to_all_async_generic supports only Linear or Ring topology");
    if (link_id != 0) {
        return;
    }

    if constexpr (Topology == tt::tt_fabric::Topology::Linear) {
        if (fabric_connections.has_forward_connection()) {
            pkt_hdr_sema_forward->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_semaphore_noc_addr_in_pkt, 1});
            fabric_connections.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr_sema_forward->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                1, static_cast<uint8_t>(num_devices - current_device_id - 1)});
            fabric_connections.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_addr_sema_forward, sizeof(PacketHeader));
        }
        if (fabric_connections.has_backward_connection()) {
            pkt_hdr_sema_backward->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_semaphore_noc_addr_in_pkt, 1});
            pkt_hdr_sema_backward->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(current_device_id)});
            fabric_connections.get_backward_connection().wait_for_empty_write_slot();
            fabric_connections.get_backward_connection().send_payload_non_blocking_from_address(
                packet_header_buffer_addr_sema_backward, sizeof(PacketHeader));
        }
    } else {
        if (fabric_connections.has_forward_connection()) {
            pkt_hdr_forward->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_semaphore_noc_addr_in_pkt, 1});
            fabric_connections.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr_forward->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_devices / 2)});
            fabric_connections.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_addr_forward, sizeof(PacketHeader));
        }
        if (fabric_connections.has_backward_connection()) {
            pkt_hdr_backward->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_semaphore_noc_addr_in_pkt, 1});
            fabric_connections.get_backward_connection().wait_for_empty_write_slot();
            pkt_hdr_backward->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                1, static_cast<uint8_t>(num_devices - num_devices / 2 - 1)});
            fabric_connections.get_backward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_addr_backward, sizeof(PacketHeader));
        }
    }
}

void write_data(
    const Noc& noc_obj,
    uint64_t dest_addrs[4],
    uint16_t payload_sizes[4],
    uint32_t parts_count,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    FabricConnections& fabric_connections,
    size_t l1_read_addr,
    uint64_t output_semaphore_noc_addr_in_pkt,
    int device_offset,
    uint16_t dest_mesh_id,
    uint16_t dest_chip_id,
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
        noc_obj.async_write_barrier();
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
                        select_connection(fabric_connections, device_offset, dest_mesh_id, dest_chip_id),
                        l1_read_addr,
                        scatter_payload,
                        pkt_hdr);
                    l1_read_addr += scatter_payload;
                } else {
                    pkt_hdr->to_noc_unicast_write(NocUnicastCommandHeader({dest_addrs[0]}), payload_sizes[0]);
                    perform_payload_send(
                        select_connection(fabric_connections, device_offset, dest_mesh_id, dest_chip_id),
                        l1_read_addr,
                        payload_sizes[0],
                        pkt_hdr);
                    l1_read_addr += payload_sizes[0];
                }
                noc_obj.async_writes_flushed();
            }

            pkt_hdr->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader(
                    {dest_addrs[parts_count - 1], output_semaphore_noc_addr_in_pkt, 1, false}),
                payload_sizes[parts_count - 1]);
            perform_payload_send(
                select_connection(fabric_connections, device_offset, dest_mesh_id, dest_chip_id),
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
                select_connection(fabric_connections, device_offset, dest_mesh_id, dest_chip_id),
                l1_read_addr,
                scatter_payload,
                pkt_hdr);
        }
    }
    noc_obj.async_writes_flushed();
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
    constexpr auto output_args = TensorAccessorArgs<17>();
    auto output_addrgen = TensorAccessor(output_args, output_address);
    size_t device_offsets_idx = arg_idx;
    arg_idx += local_num_devices * 5;

    auto fabric_connections = [&]() {
        if constexpr (is_fabric_2d) {
            Fabric2DConnections connections;
            ttnn::operations::ccl::common::open_direction_connections_async(fabric_directions, connections, arg_idx);
            return connections;
        } else {
            return FabricConnectionManager::build_from_args<
                FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx);
        }
    }();
    uint64_t init_semaphore_noc_addr_in_pkt =
        safe_get_noc_addr(sender_core_x, sender_core_y, global_init_semaphore_addr);
    uint64_t output_semaphore_noc_addr_in_pkt = safe_get_noc_addr(sender_core_x, sender_core_y, global_semaphore_addr);

    volatile tt_l1_ptr uint32_t* global_init_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_init_semaphore_addr);
    volatile tt_l1_ptr uint32_t* global_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);

    Noc noc_obj;
    CircularBuffer cb_packet_header(reserved_packet_header_cb_id);
    CircularBuffer cb0(cb0_id);

    // packet header cb
    cb_packet_header.reserve_back(1);
    auto packet_header_buffer_addr_forward = cb_packet_header.get_write_ptr();
    cb_packet_header.push_back(1);
    cb_packet_header.reserve_back(1);
    auto packet_header_buffer_addr_backward = cb_packet_header.get_write_ptr();
    cb_packet_header.push_back(1);
    cb_packet_header.reserve_back(1);
    auto packet_header_buffer_addr_sema_forward = cb_packet_header.get_write_ptr();
    cb_packet_header.push_back(1);
    cb_packet_header.reserve_back(1);
    auto packet_header_buffer_addr_sema_backward = cb_packet_header.get_write_ptr();
    cb_packet_header.push_back(1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_sema_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_sema_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_sema_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_sema_backward);

    finish_open_connections(fabric_connections);

    constexpr bool has_pre_half_tile = has_half_tile && current_device_id % 2 == 1;
    constexpr bool has_post_half_tile = has_half_tile && current_device_id % 2 == 0;

    send_initialization<topology>(
        fabric_connections,
        core_id,
        link_id,
        local_num_devices,
        device_offsets_idx,
        init_semaphore_noc_addr_in_pkt,
        pkt_hdr_forward,
        pkt_hdr_backward,
        pkt_hdr_sema_forward,
        pkt_hdr_sema_backward,
        packet_header_buffer_addr_forward,
        packet_header_buffer_addr_backward,
        packet_header_buffer_addr_sema_forward,
        packet_header_buffer_addr_sema_backward);

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
            noc_obj.async_write_barrier();
        }
    }

    noc_semaphore_wait(global_init_semaphore_addr_ptr, num_devices - 1);
    noc_semaphore_set(global_init_semaphore_addr_ptr, 0);

    for (uint32_t did = 0; did < local_num_devices; ++did) {
        int32_t device_offset = get_arg_val<int32_t>(device_offsets_idx++);
        [[maybe_unused]] int32_t distance = abs(device_offset);
        uint32_t block_idx = get_arg_val<uint32_t>(device_offsets_idx++);
        uint32_t block_end_id = get_arg_val<uint32_t>(device_offsets_idx++);
        const ccl_routing_utils::line_unicast_route_info_t route_info = {
            .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(device_offsets_idx++)),
            .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(device_offsets_idx++))};
        uint32_t device_id = (current_device_id + device_offset + num_devices) % num_devices;

        volatile PACKET_HEADER_TYPE* pkt_hdr;
        if (device_offset > 0) {
            pkt_hdr = pkt_hdr_forward;
        } else if (device_offset < 0) {
            pkt_hdr = pkt_hdr_backward;
        }
        if (device_offset != 0) {
            set_route(pkt_hdr, route_info, distance, fabric_connections);
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

        cb0.wait_front(1);
        size_t l1_read_addr = cb0.get_read_ptr();
        uint32_t current_package_payload = 0;
        uint32_t current_tile = 0;
        uint64_t dst_addrs[4] = {0};
        uint16_t payload_sizes[4] = {0};
        while (block_idx < block_end_id) {
            auto [dst_addr, payload_size] = calculate_params(block_idx);
            // If package is full, flush and start new package
            if (current_tile == 4 || current_package_payload + payload_size > 2 * output_page_size) {
                write_data(
                    noc_obj,
                    dst_addrs,
                    payload_sizes,
                    current_tile,
                    pkt_hdr,
                    fabric_connections,
                    l1_read_addr,
                    output_semaphore_noc_addr_in_pkt,
                    device_offset,
                    route_info.dst_mesh_id,
                    route_info.dst_chip_id,
                    false);
                cb0.pop_front(1);
                cb0.wait_front(1);
                l1_read_addr = cb0.get_read_ptr();
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
                noc_obj,
                dst_addrs,
                payload_sizes,
                current_tile,
                pkt_hdr,
                fabric_connections,
                l1_read_addr,
                output_semaphore_noc_addr_in_pkt,
                device_offset,
                route_info.dst_mesh_id,
                route_info.dst_chip_id,
                true);
        }
        cb0.pop_front(1);
    }

    noc_obj.async_write_barrier();
    close_connections(fabric_connections);
    if (core_id == 0 && link_id == 0) {
        noc_semaphore_wait(global_semaphore_addr_ptr, semaphore_expected_value);
        noc_semaphore_set(global_semaphore_addr_ptr, 0);
    }
}
