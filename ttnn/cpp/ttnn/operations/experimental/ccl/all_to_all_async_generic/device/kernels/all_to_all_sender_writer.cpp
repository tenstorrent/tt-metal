// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "cpp/ttnn/operations/ccl/common/types/fabric_directions.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
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
constexpr uint32_t max_pages_per_packet = get_compile_time_arg_val(17);
constexpr uint32_t custom_fabric2d_route_words = get_compile_time_arg_val(18);
constexpr bool use_multicast_initialization = get_compile_time_arg_val(19);
constexpr auto output_tensor_args = TensorAccessorArgs<20>();
constexpr uint32_t mux_ct_base = output_tensor_args.next_compile_time_args_offset();
// This flag is specialized per stream by the host. Endpoint streams without a physical egress compile against the
// routing-plane-manager ABI even when other streams in the same program use a worker mux.
constexpr bool stream_uses_mux = get_compile_time_arg_val(mux_ct_base) != 0;
constexpr uint32_t mux_num_clients = get_compile_time_arg_val(mux_ct_base + 1);
constexpr uint32_t compile_safe_mux_num_clients = mux_num_clients == 0 ? 1 : mux_num_clients;
constexpr bool has_fabric_connections = fabric_direction_mask != 0;
constexpr bool has_custom_fabric2d_routes = custom_fabric2d_route_words != 0;
static_assert(!has_custom_fabric2d_routes || is_fabric_2d, "Custom routes require Fabric2D");
// Base record: offset, block range/stride, completion, mesh and chip. Fabric2D adds the destination's harvested drain
// coordinate. Only programs that actually split an antipode add route metadata and the words used by that route.
constexpr uint32_t target_drain_args = is_fabric_2d ? 2 : 0;
constexpr uint32_t target_custom_route_args = has_custom_fabric2d_routes ? 2 + custom_fabric2d_route_words : 0;
constexpr uint32_t target_runtime_args = 7 + target_drain_args + target_custom_route_args;
constexpr uint32_t target_drain_args_idx = 7;
constexpr uint32_t target_custom_route_args_idx = target_drain_args_idx + target_drain_args;
constexpr uint32_t default_initial_direction = static_cast<uint32_t>(tt::tt_fabric::eth_chan_directions::COUNT);
constexpr auto fabric_directions =
    ttnn::operations::ccl::common::fabric_direction_mask_to_directions(fabric_direction_mask);
using Fabric2DConnections = tt::tt_fabric::RoutingPlaneConnectionManager;
using FabricMuxSender = tt::tt_fabric::FabricMuxV2Sender</*EAGER_STAGING=*/true>;
struct FabricMuxConnection {
    FabricMuxSender sender;
};
using DirectFabricConnections = std::conditional_t<is_fabric_2d, Fabric2DConnections, FabricConnectionManager>;
using FabricConnections = std::conditional_t<stream_uses_mux, FabricMuxConnection, DirectFabricConnections>;
constexpr bool fabric2d_multicast_initialization_is_safe = is_fabric_2d && use_multicast_initialization;

FORCE_INLINE uint32_t get_custom_route_num_commands(size_t target_arg_idx) {
    if constexpr (has_custom_fabric2d_routes) {
        return get_arg_val<uint32_t>(target_arg_idx + target_custom_route_args_idx);
    }
    return 0;
}

FORCE_INLINE uint32_t get_custom_route_initial_direction(size_t target_arg_idx) {
    if constexpr (has_custom_fabric2d_routes) {
        return get_arg_val<uint32_t>(target_arg_idx + target_custom_route_args_idx + 1);
    }
    return default_initial_direction;
}

FORCE_INLINE size_t get_custom_route_packed_args_idx(size_t target_arg_idx) {
    return target_arg_idx + target_custom_route_args_idx + 2;
}

[[noreturn]] FORCE_INLINE void fail_stop_invalid_fabric_route() {
    // ASSERT gives watcher builds a precise failure signal. The explicit trap and spin remain the production fallback:
    // an invalid route must not inject on an arbitrary egress and corrupt the collective's completion count.
    ASSERT(false);
    asm volatile("ebreak");
    while (true) {
    }
}

template <typename PacketHeader>
FORCE_INLINE void set_target_unicast_route(
    volatile PacketHeader* packet_header,
    const ccl_routing_utils::line_unicast_route_info_t& route_info,
    uint32_t custom_route_num_commands,
    size_t custom_route_args_idx) {
    if constexpr (std::is_base_of_v<tt::tt_fabric::HybridMeshPacketHeader, PacketHeader>) {
        if (custom_route_num_commands == 0) {
            tt::tt_fabric::fabric_set_unicast_route(packet_header, route_info.dst_chip_id, route_info.dst_mesh_id);
        }
        if (custom_route_num_commands != 0) {
            if (custom_route_num_commands > tt::tt_fabric::FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE) {
                fail_stop_invalid_fabric_route();
            }
            packet_header->dst_start_node_id =
                (static_cast<uint32_t>(route_info.dst_mesh_id) << 16) | route_info.dst_chip_id;
            packet_header->mcast_params_64 = 0;
            packet_header->is_mcast_active = 0;
            packet_header->routing_fields.value = 0;
            for (uint32_t command = 0; command < tt::tt_fabric::FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE; ++command) {
                constexpr uint32_t commands_per_word = 8;
                uint8_t route_command = 0;
                if (command < custom_route_num_commands) {
                    const uint32_t packed_word =
                        get_arg_val<uint32_t>(custom_route_args_idx + command / commands_per_word);
                    route_command = static_cast<uint8_t>((packed_word >> ((command % commands_per_word) * 4)) & 0xf);
                }
                packet_header->route_buffer[command] = route_command;
            }
            return;
        }
    } else {
        if (custom_route_num_commands != 0) {
            fail_stop_invalid_fabric_route();
        }
        ccl_routing_utils::fabric_set_line_unicast_route(packet_header, route_info);
    }
}

inline FabricMuxSender& select_connection(
    FabricMuxConnection& fabric_connection,
    [[maybe_unused]] int device_offset,
    [[maybe_unused]] uint16_t dest_mesh_id,
    [[maybe_unused]] uint16_t dest_chip_id,
    [[maybe_unused]] uint32_t initial_direction) {
    return fabric_connection.sender;
}

inline tt::tt_fabric::WorkerToFabricEdmSender& select_connection_by_direction(
    Fabric2DConnections& fabric_connections, uint32_t direction) {
    for (uint32_t slot = 0; slot < fabric_connections.active_count(); ++slot) {
        if (fabric_connections.get_tag(slot) == direction) {
            return fabric_connections.get(slot).sender;
        }
    }
    fail_stop_invalid_fabric_route();
}

inline tt::tt_fabric::WorkerToFabricEdmSender& select_connection(
    Fabric2DConnections& fabric_connections,
    [[maybe_unused]] int device_offset,
    uint16_t dest_mesh_id,
    uint16_t dest_chip_id,
    uint32_t initial_direction) {
    const uint32_t direction = initial_direction == default_initial_direction
                                   ? static_cast<uint32_t>(get_next_hop_router_direction(dest_mesh_id, dest_chip_id))
                                   : initial_direction;
    return select_connection_by_direction(fabric_connections, direction);
}

inline tt::tt_fabric::WorkerToFabricEdmSender& select_connection(
    FabricConnectionManager& fabric_connections,
    int device_offset,
    [[maybe_unused]] uint16_t dest_mesh_id,
    [[maybe_unused]] uint16_t dest_chip_id,
    [[maybe_unused]] uint32_t initial_direction) {
    return (device_offset > 0) ? fabric_connections.get_forward_connection()
                               : fabric_connections.get_backward_connection();
}

inline void finish_all_to_all_connections(Fabric2DConnections& fabric_connections) { fabric_connections.open_finish(); }

inline void finish_all_to_all_connections(FabricConnectionManager& fabric_connections) {
    fabric_connections.open_finish();
}

inline void finish_all_to_all_connections(FabricMuxConnection& fabric_connection) {
    fabric_connection.sender.flush</*Blocking=*/true>();
}

inline void close_all_to_all_connections(Fabric2DConnections& fabric_connections) { fabric_connections.close(); }

inline void close_all_to_all_connections(FabricConnectionManager& fabric_connections) { fabric_connections.close(); }

inline void close_all_to_all_connections(FabricMuxConnection& fabric_connection) { fabric_connection.sender.close(); }

inline void send_mux_packet_blocking(
    FabricMuxConnection& fabric_connection, uint32_t packet_header_address, size_t packet_header_size) {
    fabric_connection.sender.wait_for_empty_write_slot();
    fabric_connection.sender.send_payload_flush_non_blocking_from_address(packet_header_address, packet_header_size);
    noc_async_writes_flushed();
}

template <tt::tt_fabric::Topology Topology, typename PacketHeader>
void send_initialization(
    FabricMuxConnection& fabric_connection,
    uint32_t core_id,
    uint32_t link_id,
    uint32_t local_num_devices,
    size_t device_offsets_idx,
    uint64_t init_semaphore_noc_addr_in_pkt,
    uint32_t global_init_semaphore_addr,
    volatile PacketHeader* pkt_hdr_sema_forward,
    volatile PacketHeader* pkt_hdr_sema_backward,
    uint32_t packet_header_buffer_addr_sema_forward,
    uint32_t packet_header_buffer_addr_sema_backward) {
    using ttnn::operations::ccl::common::ReplicateGroup;
    static_assert(
        Topology == tt::tt_fabric::Topology::Linear || Topology == tt::tt_fabric::Topology::Ring,
        "Muxed all-to-all supports only Linear or Ring topology");
    static_assert(
        !is_fabric_2d || replicate_axis == ReplicateGroup::COLS || replicate_axis == ReplicateGroup::ROWS,
        "Fabric2D muxed all-to-all requires a concrete cluster axis");

    if (link_id != 0) {
        return;
    }
    if constexpr (
        (Topology == tt::tt_fabric::Topology::Ring || is_fabric_2d) && !fabric2d_multicast_initialization_is_safe) {
        // Ring destinations and folded Fabric2D Linear axes are split across direction-specific mux workers. Have link
        // 0 send one unicast initialization increment for each unique destination owned by this worker. Bank schedules
        // are striped across mux clients; the client owning destination bank 0 is the single initialization owner.
        // Per-target unicast avoids assuming that a logical Linear axis follows one physical mesh row or column.
        for (uint32_t did = 0; did < local_num_devices; ++did) {
            const size_t target_arg_idx = device_offsets_idx + did * target_runtime_args;
            const int32_t device_offset = get_arg_val<int32_t>(target_arg_idx);
            const uint32_t block_start = get_arg_val<uint32_t>(target_arg_idx + 1);
            if (device_offset == 0 || block_start != 0) {
                continue;
            }

            const ccl_routing_utils::line_unicast_route_info_t route_info = {
                .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(target_arg_idx + 5)),
                .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(target_arg_idx + 6))};
            const uint64_t target_init_semaphore_noc_addr =
                is_fabric_2d ? safe_get_noc_addr(
                                   get_arg_val<uint32_t>(target_arg_idx + target_drain_args_idx),
                                   get_arg_val<uint32_t>(target_arg_idx + target_drain_args_idx + 1),
                                   global_init_semaphore_addr)
                             : init_semaphore_noc_addr_in_pkt;
            auto* packet_header = device_offset > 0 ? pkt_hdr_sema_forward : pkt_hdr_sema_backward;
            const uint32_t packet_header_address =
                device_offset > 0 ? packet_header_buffer_addr_sema_forward : packet_header_buffer_addr_sema_backward;
            set_target_unicast_route(
                packet_header,
                route_info,
                get_custom_route_num_commands(target_arg_idx),
                get_custom_route_packed_args_idx(target_arg_idx));
            packet_header->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{target_init_semaphore_noc_addr, 1});
            send_mux_packet_blocking(fabric_connection, packet_header_address, sizeof(PacketHeader));
        }
    } else {
        // A straight, uniformly harvested Fabric2D axis and Fabric1D Linear use one multicast per direction, so only
        // one Mux V2 client owns initialization.
        if (core_id % compile_safe_mux_num_clients != 0) {
            return;
        }
        constexpr uint32_t positive_range = num_devices - current_device_id - 1;
        constexpr uint32_t negative_range = current_device_id;
        constexpr uint32_t positive_direction = !is_fabric_2d || replicate_axis == ReplicateGroup::ROWS
                                                    ? eth_chan_directions::EAST
                                                    : eth_chan_directions::SOUTH;
        constexpr uint32_t negative_direction = !is_fabric_2d || replicate_axis == ReplicateGroup::ROWS
                                                    ? eth_chan_directions::WEST
                                                    : eth_chan_directions::NORTH;
        constexpr bool owns_positive_targets = fabric_directions[positive_direction];
        constexpr bool owns_negative_targets = fabric_directions[negative_direction];

        if constexpr (positive_range > 0) {
            if (owns_positive_targets) {
                if constexpr (is_fabric_2d) {
                    constexpr uint16_t east_range = replicate_axis == ReplicateGroup::ROWS ? positive_range : 0;
                    constexpr uint16_t south_range = replicate_axis == ReplicateGroup::COLS ? positive_range : 0;
                    tt::tt_fabric::fabric_set_mcast_route(
                        pkt_hdr_sema_forward, source_chip_id, source_mesh_id, east_range, 0, 0, south_range);
                } else {
                    pkt_hdr_sema_forward->to_chip_multicast(
                        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(positive_range)});
                }
                pkt_hdr_sema_forward->to_noc_unicast_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_semaphore_noc_addr_in_pkt, 1});
                send_mux_packet_blocking(
                    fabric_connection, packet_header_buffer_addr_sema_forward, sizeof(PacketHeader));
            }
        }
        if constexpr (negative_range > 0) {
            if (owns_negative_targets) {
                if constexpr (is_fabric_2d) {
                    constexpr uint16_t west_range = replicate_axis == ReplicateGroup::ROWS ? negative_range : 0;
                    constexpr uint16_t north_range = replicate_axis == ReplicateGroup::COLS ? negative_range : 0;
                    tt::tt_fabric::fabric_set_mcast_route(
                        pkt_hdr_sema_backward, source_chip_id, source_mesh_id, 0, west_range, north_range, 0);
                } else {
                    pkt_hdr_sema_backward->to_chip_multicast(
                        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(negative_range)});
                }
                pkt_hdr_sema_backward->to_noc_unicast_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_semaphore_noc_addr_in_pkt, 1});
                send_mux_packet_blocking(
                    fabric_connection, packet_header_buffer_addr_sema_backward, sizeof(PacketHeader));
            }
        }
    }
}

template <tt::tt_fabric::Topology Topology, typename PacketHeader>
void send_initialization(
    Fabric2DConnections& fabric_connections,
    [[maybe_unused]] uint32_t core_id,
    uint32_t link_id,
    uint32_t local_num_devices,
    size_t device_offsets_idx,
    uint64_t init_semaphore_noc_addr_in_pkt,
    uint32_t global_init_semaphore_addr,
    volatile PacketHeader* pkt_hdr_sema_forward,
    volatile PacketHeader* pkt_hdr_sema_backward,
    uint32_t packet_header_buffer_addr_sema_forward,
    uint32_t packet_header_buffer_addr_sema_backward) {
    using ttnn::operations::ccl::common::ReplicateGroup;
    static_assert(
        Topology == tt::tt_fabric::Topology::Linear || Topology == tt::tt_fabric::Topology::Ring,
        "FABRIC_2D all-to-all supports only Linear or Ring topology");
    static_assert(
        replicate_axis == ReplicateGroup::COLS || replicate_axis == ReplicateGroup::ROWS,
        "FABRIC_2D all-to-all requires a concrete cluster axis");

    if (link_id != 0) {
        return;
    }

    if constexpr (fabric2d_multicast_initialization_is_safe) {
        constexpr uint32_t positive_range = num_devices - current_device_id - 1;
        constexpr uint32_t negative_range = current_device_id;
        constexpr uint32_t positive_direction =
            replicate_axis == ReplicateGroup::COLS ? eth_chan_directions::SOUTH : eth_chan_directions::EAST;
        constexpr uint32_t negative_direction =
            replicate_axis == ReplicateGroup::COLS ? eth_chan_directions::NORTH : eth_chan_directions::WEST;

        if constexpr (positive_range > 0 && fabric_directions[positive_direction]) {
            constexpr uint16_t east_range = replicate_axis == ReplicateGroup::ROWS ? positive_range : 0;
            constexpr uint16_t south_range = replicate_axis == ReplicateGroup::COLS ? positive_range : 0;
            tt::tt_fabric::fabric_set_mcast_route(
                pkt_hdr_sema_forward, source_chip_id, source_mesh_id, east_range, 0, 0, south_range);
            pkt_hdr_sema_forward->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_semaphore_noc_addr_in_pkt, 1});
            auto& connection = select_connection_by_direction(fabric_connections, positive_direction);
            connection.wait_for_empty_write_slot();
            connection.send_payload_flush_blocking_from_address(
                packet_header_buffer_addr_sema_forward, sizeof(PacketHeader));
        }
        if constexpr (negative_range > 0 && fabric_directions[negative_direction]) {
            constexpr uint16_t west_range = replicate_axis == ReplicateGroup::ROWS ? negative_range : 0;
            constexpr uint16_t north_range = replicate_axis == ReplicateGroup::COLS ? negative_range : 0;
            tt::tt_fabric::fabric_set_mcast_route(
                pkt_hdr_sema_backward, source_chip_id, source_mesh_id, 0, west_range, north_range, 0);
            pkt_hdr_sema_backward->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_semaphore_noc_addr_in_pkt, 1});
            auto& connection = select_connection_by_direction(fabric_connections, negative_direction);
            connection.wait_for_empty_write_slot();
            connection.send_payload_flush_blocking_from_address(
                packet_header_buffer_addr_sema_backward, sizeof(PacketHeader));
        }
        return;
    }

    // A logical ring can turn through several physical directions. Send one initialization increment to every unique
    // destination owned by this stream, using the connection selected from the destination's actual first hop. Bank
    // schedules are striped across streams, so the stream owning destination bank 0 is the single initialization owner.
    for (uint32_t did = 0; did < local_num_devices; ++did) {
        const size_t target_arg_idx = device_offsets_idx + did * target_runtime_args;
        const int32_t device_offset = get_arg_val<int32_t>(target_arg_idx);
        const uint32_t block_start = get_arg_val<uint32_t>(target_arg_idx + 1);
        if (device_offset == 0 || block_start != 0) {
            continue;
        }

        const ccl_routing_utils::line_unicast_route_info_t route_info = {
            .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(target_arg_idx + 5)),
            .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(target_arg_idx + 6))};
        const uint64_t target_init_semaphore_noc_addr = safe_get_noc_addr(
            get_arg_val<uint32_t>(target_arg_idx + target_drain_args_idx),
            get_arg_val<uint32_t>(target_arg_idx + target_drain_args_idx + 1),
            global_init_semaphore_addr);
        auto* packet_header = device_offset > 0 ? pkt_hdr_sema_forward : pkt_hdr_sema_backward;
        const uint32_t packet_header_address =
            device_offset > 0 ? packet_header_buffer_addr_sema_forward : packet_header_buffer_addr_sema_backward;
        set_target_unicast_route(
            packet_header,
            route_info,
            get_custom_route_num_commands(target_arg_idx),
            get_custom_route_packed_args_idx(target_arg_idx));
        packet_header->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{target_init_semaphore_noc_addr, 1});
        auto& connection = select_connection(
            fabric_connections,
            device_offset,
            route_info.dst_mesh_id,
            route_info.dst_chip_id,
            get_custom_route_initial_direction(target_arg_idx));
        connection.wait_for_empty_write_slot();
        connection.send_payload_flush_blocking_from_address(packet_header_address, sizeof(PacketHeader));
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
    [[maybe_unused]] uint32_t global_init_semaphore_addr,
    volatile PacketHeader* pkt_hdr_sema_forward,
    volatile PacketHeader* pkt_hdr_sema_backward,
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
            fabric_connections.get_backward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_addr_sema_backward, sizeof(PacketHeader));
        }
    } else {
        if (fabric_connections.has_forward_connection()) {
            pkt_hdr_sema_forward->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_semaphore_noc_addr_in_pkt, 1});
            fabric_connections.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr_sema_forward->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_devices / 2)});
            fabric_connections.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_addr_sema_forward, sizeof(PacketHeader));
        }
        if (fabric_connections.has_backward_connection()) {
            pkt_hdr_sema_backward->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_semaphore_noc_addr_in_pkt, 1});
            fabric_connections.get_backward_connection().wait_for_empty_write_slot();
            pkt_hdr_sema_backward->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                1, static_cast<uint8_t>(num_devices - num_devices / 2 - 1)});
            fabric_connections.get_backward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_addr_sema_backward, sizeof(PacketHeader));
        }
    }
}

template <typename FabricSender>
void write_data(
    const Noc& noc_obj,
    uint64_t dest_addrs[4],
    uint16_t payload_sizes[4],
    uint32_t parts_count,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    FabricSender* target_connection,
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
    } else {
        ASSERT(target_connection != nullptr);
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
                    perform_payload_send(*target_connection, l1_read_addr, scatter_payload, pkt_hdr);
                    l1_read_addr += scatter_payload;
                } else {
                    pkt_hdr->to_noc_unicast_write(NocUnicastCommandHeader({dest_addrs[0]}), payload_sizes[0]);
                    perform_payload_send(*target_connection, l1_read_addr, payload_sizes[0], pkt_hdr);
                    l1_read_addr += payload_sizes[0];
                }
                noc_obj.async_writes_flushed();
            }

            pkt_hdr->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader(
                    {dest_addrs[parts_count - 1], output_semaphore_noc_addr_in_pkt, 1, false}),
                payload_sizes[parts_count - 1]);
            perform_payload_send(*target_connection, l1_read_addr, payload_sizes[parts_count - 1], pkt_hdr);
        } else {
            uint32_t scatter_payload = 0;
            for (uint32_t part = 0; part < parts_count; ++part) {
                scatter_payload += payload_sizes[part];
            }
            if (parts_count == 1) {
                pkt_hdr->to_noc_unicast_write(NocUnicastCommandHeader({dest_addrs[0]}), payload_sizes[0]);
            } else {
                pkt_hdr->to_noc_unicast_scatter_write(
                    NocUnicastScatterCommandHeader(dest_addrs, payload_sizes, parts_count), scatter_payload);
            }
            perform_payload_send(*target_connection, l1_read_addr, scatter_payload, pkt_hdr);
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
    auto output_addrgen = TensorAccessor(output_tensor_args, output_address);
    size_t device_offsets_idx = arg_idx;
    arg_idx += local_num_devices * target_runtime_args;

    auto fabric_connections = [&]() {
        if constexpr (stream_uses_mux) {
            auto sender = FabricMuxSender::build_from_args(arg_idx);
            // Match the direct-fabric connection lifecycle: begin the mux handshake before local setup, then wait for
            // readiness at finish_all_to_all_connections() immediately before issuing initialization traffic.
            sender.open();
            return FabricMuxConnection{.sender = sender};
        } else if constexpr (is_fabric_2d) {
            const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
            return Fabric2DConnections::build_from_args<Fabric2DConnections::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
                arg_idx, num_connections);
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

    {
        if constexpr ((is_fabric_2d && !stream_uses_mux) || has_fabric_connections) {
            finish_all_to_all_connections(fabric_connections);
        }
    }

    constexpr bool has_pre_half_tile = has_half_tile && current_device_id % 2 == 1;
    constexpr bool has_post_half_tile = has_half_tile && current_device_id % 2 == 0;

    {
        send_initialization<topology>(
            fabric_connections,
            core_id,
            link_id,
            local_num_devices,
            device_offsets_idx,
            init_semaphore_noc_addr_in_pkt,
            global_init_semaphore_addr,
            pkt_hdr_sema_forward,
            pkt_hdr_sema_backward,
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
    }

    {
        for (uint32_t did = 0; did < local_num_devices; ++did) {
            const int32_t device_offset = get_arg_val<int32_t>(device_offsets_idx++);
            uint32_t block_idx = get_arg_val<uint32_t>(device_offsets_idx++);
            const uint32_t block_end_id = get_arg_val<uint32_t>(device_offsets_idx++);
            const uint32_t block_stride = get_arg_val<uint32_t>(device_offsets_idx++);
            const bool signal_completion = get_arg_val<uint32_t>(device_offsets_idx++) != 0;
            const ccl_routing_utils::line_unicast_route_info_t route_info = {
                .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(device_offsets_idx++)),
                .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(device_offsets_idx++))};
            uint32_t target_drain_sync_core_x = 0;
            uint32_t target_drain_sync_core_y = 0;
            if constexpr (is_fabric_2d) {
                target_drain_sync_core_x = get_arg_val<uint32_t>(device_offsets_idx++);
                target_drain_sync_core_y = get_arg_val<uint32_t>(device_offsets_idx++);
            }
            uint32_t custom_route_num_commands = 0;
            uint32_t custom_route_initial_direction = default_initial_direction;
            if constexpr (has_custom_fabric2d_routes) {
                custom_route_num_commands = get_arg_val<uint32_t>(device_offsets_idx++);
                custom_route_initial_direction = get_arg_val<uint32_t>(device_offsets_idx++);
            }
            const size_t custom_route_args_idx = device_offsets_idx;
            device_offsets_idx += custom_fabric2d_route_words;
            const uint64_t target_output_semaphore_noc_addr =
                is_fabric_2d
                    ? safe_get_noc_addr(target_drain_sync_core_x, target_drain_sync_core_y, global_semaphore_addr)
                    : output_semaphore_noc_addr_in_pkt;
            const uint32_t device_id = (current_device_id + device_offset + num_devices) % num_devices;

            volatile PACKET_HEADER_TYPE* pkt_hdr = nullptr;
            if (device_offset > 0) {
                pkt_hdr = pkt_hdr_forward;
            } else if (device_offset < 0) {
                pkt_hdr = pkt_hdr_backward;
            }
            if (device_offset != 0) {
                set_target_unicast_route(pkt_hdr, route_info, custom_route_num_commands, custom_route_args_idx);
            }
            auto* target_connection = device_offset == 0 ? nullptr
                                                         : &select_connection(
                                                               fabric_connections,
                                                               device_offset,
                                                               route_info.dst_mesh_id,
                                                               route_info.dst_chip_id,
                                                               custom_route_initial_direction);

            auto calculate_params = [&](int b) {
                const uint32_t o = b / (concat_num_tiles * inner_dims_size);
                const uint32_t c = (b / inner_dims_size) % concat_num_tiles;
                const uint32_t i = b % inner_dims_size;
                const uint32_t dest_tile_id =
                    o * inner_dims_size * concat_dim_size + (c + full_block_offset) * inner_dims_size + i;
                uint16_t payload_size =
                    ((has_pre_half_tile && c == 0) || (has_post_half_tile && c == concat_num_tiles - 1))
                        ? output_page_size / 2
                        : output_page_size;
                uint32_t offset = (has_pre_half_tile && c == 0) ? (current_device_id % 2) * output_page_size / 2 : 0;
                uint64_t dst_addr =
                    (current_device_id == device_id)
                        ? output_addrgen.get_noc_addr(dest_tile_id, offset)
                        : tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, dest_tile_id, offset);
                return std::tuple{dst_addr, payload_size};
            };

            // The generic work split can assign an empty range when there are fewer blocks than links. The reader
            // publishes no CB entry for that range, and the host excludes it from the completion-semaphore count.
            if (block_idx >= block_end_id) {
                continue;
            }

            cb0.wait_front(1);
            size_t l1_read_addr = cb0.get_read_ptr();
            uint32_t pages_in_cb = 0;
            uint32_t current_package_payload = 0;
            uint32_t current_part = 0;
            uint64_t dst_addrs[4] = {0};
            uint16_t payload_sizes[4] = {0};
            auto flush_packet = [&](bool last) {
                write_data(
                    noc_obj,
                    dst_addrs,
                    payload_sizes,
                    current_part,
                    pkt_hdr,
                    target_connection,
                    l1_read_addr,
                    target_output_semaphore_noc_addr,
                    device_offset,
                    last);
                l1_read_addr += current_package_payload;
                current_package_payload = 0;
                current_part = 0;
            };

            while (block_idx < block_end_id) {
                auto [dst_addr, payload_size] = calculate_params(block_idx);
                const bool extends_previous_run =
                    current_part > 0 && dst_addr == dst_addrs[current_part - 1] + payload_sizes[current_part - 1];
                if (!extends_previous_run && current_part == 4) {
                    // Four is a scatter-header limit, not a page limit. Flush these runs and continue consuming the
                    // same CB page; a bank-owned batch normally remains one run for all seven pages.
                    flush_packet(false);
                }
                current_package_payload += payload_size;
                if (extends_previous_run) {
                    payload_sizes[current_part - 1] += payload_size;
                } else {
                    payload_sizes[current_part] = payload_size;
                    dst_addrs[current_part] = dst_addr;
                    current_part++;
                }
                pages_in_cb++;
                block_idx += block_stride;

                const bool schedule_complete = block_idx >= block_end_id;
                if (pages_in_cb == max_pages_per_packet || schedule_complete) {
                    flush_packet(schedule_complete && signal_completion);
                    cb0.pop_front(1);
                    pages_in_cb = 0;
                    if (!schedule_complete) {
                        cb0.wait_front(1);
                        l1_read_addr = cb0.get_read_ptr();
                    }
                }
            }
        }
    }

    noc_obj.async_write_barrier();
    {
        if constexpr ((is_fabric_2d && !stream_uses_mux) || has_fabric_connections) {
            close_all_to_all_connections(fabric_connections);
        }
    }
    if (core_id == 0 && link_id == 0) {
        noc_semaphore_wait(global_semaphore_addr_ptr, semaphore_expected_value);
        noc_semaphore_set(global_semaphore_addr_ptr, 0);
    }
}
