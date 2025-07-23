// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"

namespace ttnn::operations::ccl::common {

template <size_t Size>
inline void open_direction_connections(
    const std::array<bool, Size>& directions,
    std::array<WorkerToFabricEdmSender, Size>& connections,
    size_t rt_args_idx) {
    for (uint32_t i = 0; i < Size; i++) {
        if (directions[i]) {
            connections[i] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
            connections[i].open();
        }
    }
}

template <size_t Size>
inline void close_direction_connections(
    const std::array<bool, Size>& directions, std::array<WorkerToFabricEdmSender, Size>& connections) {
    for (size_t i = 0; i < Size; ++i) {
        if (directions[i]) {
            connections[i].close();
        }
    }
}

// Batch is shared among all devices, or replicated along each row or each column.
enum class ReplicateGroup : int {
    NONE = -1,
    ROWS = 1,
    COLS = 0,
};

// check if the target is along the same axis as the source if the axis is configured
template <uint32_t LinearizedSrcMeshCoord, uint32_t MeshRows, uint32_t MeshCols, ReplicateGroup Axis>
bool is_configured_target(uint32_t linearized_dest_mesh_coord) {
    // axis is the direction along which we are allowed to send packets
    // axis = 1; means we are allowed to send packets in the row direction
    // axis = 0; means we are allowed to send packets in the column direction
    // axis = -1; means we are allowed to send packets in all directions
    if constexpr (Axis == ReplicateGroup::COLS) {  // check if they're on the same column
        return LinearizedSrcMeshCoord % MeshCols == linearized_dest_mesh_coord % MeshCols;
    } else if constexpr (Axis == ReplicateGroup::ROWS) {  // check if they're on the same row
        return LinearizedSrcMeshCoord / MeshCols == linearized_dest_mesh_coord / MeshCols;
    } else {
        return true;  // if axis is not configured, we assume the target is configured, which is the default case, which
                      // is all directions
    }
}

template <tt::tt_fabric::Topology Topology>
constexpr bool has_wrap_around() {
    return Topology == tt::tt_fabric::Topology::Ring || Topology == tt::tt_fabric::Topology::Torus;
}

template <tt::tt_fabric::Topology Topology>
constexpr bool is_1d_topology() {
    return Topology == tt::tt_fabric::Topology::Linear || Topology == tt::tt_fabric::Topology::Ring;
}

template <tt::tt_fabric::Topology Topology>
constexpr bool is_2d_topology() {
    return Topology == tt::tt_fabric::Topology::Mesh || Topology == tt::tt_fabric::Topology::Torus;
}

template <uint32_t MeshRows, uint32_t MeshCols>
std::pair<uint32_t, uint32_t> get_mesh_coords(uint32_t linearized_mesh_coord) {
    // {row, column}
    return {linearized_mesh_coord / MeshCols, linearized_mesh_coord % MeshCols};
}

template <tt::tt_fabric::Topology Topology>
uint32_t distance(uint32_t position_1, uint32_t position_2, uint32_t axis_size) {
    if (position_1 == position_2) {
        return 0;
    }
    uint32_t line_distance = std::abs(int(position_2) - int(position_1));
    if constexpr (has_wrap_around<Topology>()) {
        return std::min(line_distance, axis_size - line_distance);
    } else {
        return line_distance;
    }
}

template <tt::tt_fabric::Topology Topology, uint32_t MeshRows, uint32_t MeshCols>
uint32_t manhattan_distance(uint32_t linearized_src_mesh_coord, uint32_t linearized_dest_mesh_coord) {
    auto [src_row, src_col] = get_mesh_coords<MeshRows, MeshCols>(linearized_src_mesh_coord);
    auto [dest_row, dest_col] = get_mesh_coords<MeshRows, MeshCols>(linearized_dest_mesh_coord);
    return distance<Topology>(src_row, dest_row, MeshRows) + distance<Topology>(src_col, dest_col, MeshCols);
}

template <tt::tt_fabric::Topology Topology, uint32_t MeshRows, uint32_t MeshCols>
uint32_t get_route(uint32_t linearized_src_mesh_coord, uint32_t linearized_dest_mesh_coord) {
    auto [src_row, src_col] = get_mesh_coords<MeshRows, MeshCols>(linearized_src_mesh_coord);
    auto [dest_row, dest_col] = get_mesh_coords<MeshRows, MeshCols>(linearized_dest_mesh_coord);

    if (src_row == dest_row) {
        if constexpr (!has_wrap_around<Topology>()) {
            return src_col < dest_col ? eth_chan_directions::EAST : eth_chan_directions::WEST;
        } else {
            // with wrap around, we can go either East or West. Choose the shorter route
            uint32_t east_distance = distance<tt::tt_fabric::Topology::Mesh>(src_col, dest_col, MeshCols);
            uint32_t west_distance = distance<tt::tt_fabric::Topology::Mesh>(src_col, dest_col, MeshCols);
            return east_distance < west_distance ? eth_chan_directions::EAST : eth_chan_directions::WEST;
        }
    } else if (src_col == dest_col) {
        if constexpr (!has_wrap_around<Topology>()) {
            return src_row < dest_row ? eth_chan_directions::SOUTH : eth_chan_directions::NORTH;
        } else {
            // with wrap around, we can go either North or South. Choose the shorter route
            uint32_t north_distance = distance<tt::tt_fabric::Topology::Mesh>(src_row, dest_row, MeshRows);
            uint32_t south_distance = distance<tt::tt_fabric::Topology::Mesh>(src_row, dest_row, MeshRows);
            return north_distance < south_distance ? eth_chan_directions::NORTH : eth_chan_directions::SOUTH;
        }
    } else {
        // when diagonal, we go North or South first, then East or West
        // so we route either North or South
        if constexpr (!has_wrap_around<Topology>()) {
            return src_row < dest_row ? eth_chan_directions::SOUTH : eth_chan_directions::NORTH;
        } else {
            // with wrap around, we can go either North or South. Choose the shorter route
            uint32_t north_distance = distance<tt::tt_fabric::Topology::Mesh>(src_row, dest_row, MeshRows);
            uint32_t south_distance = distance<tt::tt_fabric::Topology::Mesh>(src_row, dest_row, MeshRows);
            return north_distance < south_distance ? eth_chan_directions::NORTH : eth_chan_directions::SOUTH;
        }
    }
}

template <uint32_t FabricMaxPacketSzBytes>
inline void fabric_send_noc_unicast(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    int32_t size_bytes,
    uint32_t alignment) {
    while (size_bytes > 0) {
        uint32_t curr_packet_size = std::min(FabricMaxPacketSzBytes, (uint32_t)size_bytes);

        packet_header->to_noc_unicast_write(
            NocUnicastCommandHeader{noc_payload_write_address}, align(curr_packet_size, alignment));

        fabric_connection.wait_for_empty_write_slot();

        fabric_connection.send_payload_without_header_non_blocking_from_address(payload_l1_address, curr_packet_size);

        fabric_connection.send_payload_flush_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));

        payload_l1_address += curr_packet_size;
        noc_payload_write_address += curr_packet_size;
        size_bytes -= curr_packet_size;
    }
}

/*
enum eth_chan_directions {
    EAST = 0,
    WEST = 1,
    NORTH = 2,
    SOUTH = 3,
    COUNT = 4,
};*/
template <uint32_t SrcChipId, uint32_t MeshRows, uint32_t MeshCols, int32_t FabricMaxPacketSzBytes>
inline void fabric_send_chip_unicast_noc_unicast(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    const uint32_t dest_chip_id,
    const uint32_t dest_mesh_id,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    int32_t size_bytes,
    const uint32_t alignment) {
    const uint32_t route = get_next_hop_router_direction(dest_mesh_id, dest_chip_id);

    // Populate packet header with routing information
    fabric_set_unicast_route(
        (LowLatencyMeshPacketHeader*)packet_header,
        static_cast<eth_chan_directions>(fabric_connections[route].direction),
        SrcChipId,
        dest_chip_id,
        dest_mesh_id,
        MeshCols);

    fabric_send_noc_unicast<FabricMaxPacketSzBytes>(
        fabric_connections[route], packet_header, payload_l1_address, noc_payload_write_address, size_bytes, alignment);
}

template <uint32_t FabricMaxPacketSzBytes>
inline void fabric_send_noc_unicast_with_semaphore(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    uint64_t noc_remote_semaphore_address,
    int32_t size_bytes,
    uint32_t alignment,
    uint16_t increment_value,
    bool flush) {
    while (size_bytes > 0) {
        uint32_t curr_packet_size = std::min(FabricMaxPacketSzBytes, (uint32_t)size_bytes);

        if ((uint32_t)size_bytes == curr_packet_size) {
            // Fill header for fused unicast + atomic increment command when it is the last packet
            packet_header->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader(
                    noc_payload_write_address, noc_remote_semaphore_address, increment_value, 32, flush),
                align(curr_packet_size, alignment));
        } else {
            // Fill header for fused unicast + atomic increment command when it is not the last packet
            packet_header->to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{noc_payload_write_address}, align(curr_packet_size, alignment));
        }

        // Send payload followed by header over the fabric.
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_without_header_non_blocking_from_address(payload_l1_address, curr_packet_size);
        fabric_connection.send_payload_flush_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));

        payload_l1_address += curr_packet_size;
        noc_payload_write_address += curr_packet_size;
        size_bytes -= curr_packet_size;
    }
}

// Insert helper that handles the remote-device metadata path with fused atomic increment
template <uint32_t SrcChipId, uint32_t MeshRows, uint32_t MeshCols, uint32_t FabricMaxPacketSzBytes>
inline void fabric_send_chip_unicast_noc_unicast_with_semaphore(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t dest_chip_id,
    uint32_t dest_mesh_id,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    uint64_t noc_remote_semaphore_address,
    int32_t size_bytes,
    uint32_t alignment,
    uint16_t increment_value,
    bool flush) {
    uint32_t route = get_next_hop_router_direction(dest_mesh_id, dest_chip_id);

    // Populate packet header with routing information
    fabric_set_unicast_route(
        (LowLatencyMeshPacketHeader*)packet_header,
        static_cast<eth_chan_directions>(fabric_connections[route].direction),
        SrcChipId,
        dest_chip_id,
        dest_mesh_id,
        MeshCols);

    return fabric_send_noc_unicast_with_semaphore<FabricMaxPacketSzBytes>(
        fabric_connections[route],
        packet_header,
        payload_l1_address,
        noc_payload_write_address,
        noc_remote_semaphore_address,
        size_bytes,
        alignment,
        increment_value,
        flush);
}

// Fabric send for NOC unicast semaphore increment only (no payload)
template <uint32_t SrcChipId, uint32_t MeshRows, uint32_t MeshCols>
inline void fabric_send_chip_unicast_noc_unicast_semaphore_only(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t dest_chip_id,
    uint32_t dest_mesh_id,
    uint64_t noc_remote_semaphore_address,
    uint16_t increment_value,
    bool flush) {
    // Set up packet header for semaphore increment
    packet_header->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{noc_remote_semaphore_address, increment_value, 32, flush});

    uint32_t route = get_next_hop_router_direction(dest_mesh_id, dest_chip_id);

    // Populate packet header with routing information
    fabric_set_unicast_route(
        (LowLatencyMeshPacketHeader*)packet_header,
        static_cast<eth_chan_directions>(fabric_connections[route].direction),
        SrcChipId,
        dest_chip_id,
        dest_mesh_id,
        MeshCols);

    // Send only the packet header (for semaphore increment)
    fabric_connections[route].wait_for_empty_write_slot();
    fabric_connections[route].send_payload_flush_blocking_from_address(
        reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));
}

template <
    uint32_t LinearizedSrcMeshCoord,
    tt::tt_fabric::Topology Topology,
    uint32_t MeshRows,
    uint32_t MeshCols,
    int32_t FabricMaxPacketSzBytes>
inline void fabric_send_chip_unicast_noc_unicast_1d(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    const uint32_t linearized_dest_mesh_coord,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    int32_t size_bytes,
    const uint32_t alignment) {
    uint32_t distance =
        manhattan_distance<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    packet_header->to_chip_unicast(distance);

    uint32_t route = get_route<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    fabric_send_noc_unicast<FabricMaxPacketSzBytes>(
        fabric_connections[route], packet_header, payload_l1_address, noc_payload_write_address, size_bytes, alignment);
}

template <
    uint32_t LinearizedSrcMeshCoord,
    tt::tt_fabric::Topology Topology,
    uint32_t MeshRows,
    uint32_t MeshCols,
    int32_t FabricMaxPacketSzBytes>
inline void fabric_send_chip_unicast_noc_unicast_with_semaphore_1d(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    const uint32_t linearized_dest_mesh_coord,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    uint64_t noc_remote_semaphore_address,
    int32_t size_bytes,
    const uint32_t alignment,
    uint16_t increment_value,
    bool flush) {
    uint32_t distance =
        manhattan_distance<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    packet_header->to_chip_unicast(distance);

    uint32_t route = get_route<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    return fabric_send_noc_unicast_with_semaphore<FabricMaxPacketSzBytes>(
        fabric_connections[route],
        packet_header,
        payload_l1_address,
        noc_payload_write_address,
        noc_remote_semaphore_address,
        size_bytes,
        alignment,
        increment_value,
        flush);
}

// Fabric send for NOC unicast semaphore increment only in 1D topology (no payload)
template <uint32_t LinearizedSrcMeshCoord, tt::tt_fabric::Topology Topology, uint32_t MeshRows, uint32_t MeshCols>
inline void fabric_send_chip_unicast_noc_unicast_semaphore_only_1d(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    const uint32_t linearized_dest_mesh_coord,
    uint64_t noc_remote_semaphore_address,
    uint16_t increment_value,
    bool flush) {
    // Set up packet header for semaphore increment
    packet_header->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{noc_remote_semaphore_address, increment_value, 32, flush});

    uint32_t distance =
        manhattan_distance<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    packet_header->to_chip_unicast(distance);

    uint32_t route = get_route<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);

    // Send only the packet header (for semaphore increment)
    fabric_connections[route].wait_for_empty_write_slot();
    fabric_connections[route].send_payload_flush_blocking_from_address(
        reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));
}

}  // namespace ttnn::operations::ccl::common
