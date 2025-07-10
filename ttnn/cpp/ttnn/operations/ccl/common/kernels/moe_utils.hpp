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

template <uint32_t src_chip_id, uint32_t mesh_cols, uint32_t mesh_rows, ReplicateGroup Axis>
bool is_configured_target(uint32_t dest_chip_id) {
    // axis is the direction along which we are allowed to send packets
    // axis = 1; means we are allowed to send packets in the row direction
    // axis = 0; means we are allowed to send packets in the column direction
    // axis = -1; means we are allowed to send packets in all directions
    if constexpr (Axis == ReplicateGroup::COLS) {  // check if they're on the same column
        return src_chip_id % mesh_cols == dest_chip_id % mesh_cols;
    } else if constexpr (Axis == ReplicateGroup::ROWS) {  // check if they're on the same row
        return src_chip_id / mesh_cols == dest_chip_id / mesh_cols;
    } else {
        return true;  // if axis is not configured, we assume the target is configured, which is the default case, which
                      // is all directions
    }
}

template <uint32_t linearized_src_mesh_coord, uint32_t mesh_cols, uint32_t mesh_rows, ReplicateGroup Axis>
bool is_configured_target_mesh(uint32_t linearized_dest_mesh_coord) {
    // axis is the direction along which we are allowed to send packets
    // axis = 1; means we are allowed to send packets in the row direction
    // axis = 0; means we are allowed to send packets in the column direction
    // axis = -1; means we are allowed to send packets in all directions
    if constexpr (Axis == ReplicateGroup::COLS) {  // check if they're on the same column
        return linearized_src_mesh_coord % mesh_cols == linearized_dest_mesh_coord % mesh_cols;
    } else if constexpr (Axis == ReplicateGroup::ROWS) {  // check if they're on the same row
        return linearized_src_mesh_coord / mesh_cols == linearized_dest_mesh_coord / mesh_cols;
    } else {
        return true;  // if axis is not configured, we assume the target is configured, which is the default case, which
                      // is all directions
    }
}

template <uint32_t MaxPacketSzBytes>
inline void dispatch_noc_uni(
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    int32_t size_bytes,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    uint32_t alignment,
    volatile PACKET_HEADER_TYPE* packet_header) {
    while (size_bytes > 0) {
        uint32_t curr_packet_size = std::min(MaxPacketSzBytes, (uint32_t)size_bytes);

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
template <uint32_t SrcChipId, uint32_t MeshCols, uint32_t MeshRows, int32_t MaxPacketSzBytes>
inline void dispatch_input_remote_device(
    const uint32_t dest_chip_id,
    const uint32_t dest_mesh_id,
    const uint32_t alignment,
    int32_t size_bytes,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header) {
    const uint32_t route = get_next_hop_router_direction(dest_mesh_id, dest_chip_id);

    // Populate packet header with routing information
    fabric_set_unicast_route(
        (LowLatencyMeshPacketHeader*)packet_header,
        static_cast<eth_chan_directions>(fabric_connections[route].direction),
        SrcChipId,
        dest_chip_id,
        dest_mesh_id,
        MeshCols);

    dispatch_noc_uni<MaxPacketSzBytes>(
        payload_l1_address, noc_payload_write_address, size_bytes, fabric_connections[route], alignment, packet_header);
}

template <uint32_t FabricMaxPacketSize>
inline void dispatch_noc_uni_fused_sem_inc(
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    uint64_t noc_remote_semaphore_address,
    int32_t size,
    uint16_t increment_value,
    bool flush,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t alignment) {
    while (size > 0) {
        uint32_t curr_packet_size = std::min(FabricMaxPacketSize, (uint32_t)size);

        if ((uint32_t)size == curr_packet_size) {
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
        size -= curr_packet_size;
    }
}

// Insert helper that handles the remote-device metadata path with fused atomic increment
template <uint32_t SrcChipId, uint32_t MeshCols, uint32_t MeshRows, uint32_t FabricMaxPacketSize>
inline void dispatch_chip_uni_noc_uni_fused_sem_inc(
    uint32_t dest_chip_id,
    uint32_t dest_mesh_id,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    uint64_t noc_remote_semaphore_address,
    int32_t size,
    uint16_t increment_value,
    bool flush,
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t alignment) {
    uint32_t route = get_next_hop_router_direction(dest_mesh_id, dest_chip_id);

    // Populate packet header with routing information
    fabric_set_unicast_route(
        (LowLatencyMeshPacketHeader*)packet_header,
        static_cast<eth_chan_directions>(fabric_connections[route].direction),
        SrcChipId,
        dest_chip_id,
        dest_mesh_id,
        MeshCols);

    return dispatch_noc_uni_fused_sem_inc<FabricMaxPacketSize>(
        payload_l1_address,
        noc_payload_write_address,
        noc_remote_semaphore_address,
        size,
        increment_value,
        flush,
        fabric_connections[route],
        packet_header,
        alignment);
}

bool has_wrap_around(tt::tt_fabric::Topology topology) {
    return topology == tt::tt_fabric::Topology::Ring || topology == tt::tt_fabric::Topology::Torus;
}

bool is_1d_topology(tt::tt_fabric::Topology topology) {
    return topology == tt::tt_fabric::Topology::Linear || topology == tt::tt_fabric::Topology::Ring;
}

bool is_2d_topology(tt::tt_fabric::Topology topology) {
    return topology == tt::tt_fabric::Topology::Mesh || topology == tt::tt_fabric::Topology::Torus;
}

template <uint32_t MeshCols, uint32_t MeshRows>
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
    if (has_wrap_around(Topology)) {
        return std::min(line_distance, axis_size - line_distance);
    } else {
        return line_distance;
    }
}

template <tt::tt_fabric::Topology Topology, uint32_t MeshCols, uint32_t MeshRows>
uint32_t manhattan_distance(uint32_t linearized_src_mesh_coord, uint32_t linearized_dest_mesh_coord) {
    auto [src_row, src_col] = get_mesh_coords<MeshCols, MeshRows>(linearized_src_mesh_coord);
    auto [dest_row, dest_col] = get_mesh_coords<MeshCols, MeshRows>(linearized_dest_mesh_coord);
    return distance<Topology>(src_row, dest_row, MeshRows) + distance<Topology>(src_col, dest_col, MeshCols);
}

template <tt::tt_fabric::Topology Topology, uint32_t MeshCols, uint32_t MeshRows>
uint32_t get_route(uint32_t linearized_src_mesh_coord, uint32_t linearized_dest_mesh_coord) {
    auto [src_row, src_col] = get_mesh_coords<MeshCols, MeshRows>(linearized_src_mesh_coord);
    auto [dest_row, dest_col] = get_mesh_coords<MeshCols, MeshRows>(linearized_dest_mesh_coord);

    if (src_row == dest_row) {
        if (!has_wrap_around(Topology)) {
            return src_col < dest_col ? eth_chan_directions::EAST : eth_chan_directions::WEST;
        } else {
            // with wrap around, we can go either East or West. Choose the shorter route
            uint32_t east_distance = distance<tt::tt_fabric::Topology::Mesh>(src_col, dest_col, MeshCols);
            uint32_t west_distance = distance<tt::tt_fabric::Topology::Mesh>(src_col, dest_col, MeshCols);
            return east_distance < west_distance ? eth_chan_directions::EAST : eth_chan_directions::WEST;
        }
    } else if (src_col == dest_col) {
        if (!has_wrap_around(Topology)) {
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
        if (!has_wrap_around(Topology)) {
            return src_row < dest_row ? eth_chan_directions::SOUTH : eth_chan_directions::NORTH;
        } else {
            // with wrap around, we can go either North or South. Choose the shorter route
            uint32_t north_distance = distance<tt::tt_fabric::Topology::Mesh>(src_row, dest_row, MeshRows);
            uint32_t south_distance = distance<tt::tt_fabric::Topology::Mesh>(src_row, dest_row, MeshRows);
            return north_distance < south_distance ? eth_chan_directions::NORTH : eth_chan_directions::SOUTH;
        }
    }
}

template <
    uint32_t LinearizedSrcMeshCoord,
    uint32_t MeshCols,
    uint32_t MeshRows,
    int32_t MaxPacketSzBytes,
    tt::tt_fabric::Topology Topology>
inline void dispatch_input_remote_device_1d(
    const uint32_t linearized_dest_mesh_coord,
    const uint32_t alignment,
    int32_t size_bytes,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header) {
    uint32_t distance =
        manhattan_distance<Topology, MeshCols, MeshRows>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    packet_header->to_chip_unicast(distance);

    uint32_t route = get_route<Topology, MeshCols, MeshRows>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    dispatch_noc_uni<MaxPacketSzBytes>(
        payload_l1_address, noc_payload_write_address, size_bytes, fabric_connections[route], alignment, packet_header);
}

template <
    uint32_t LinearizedSrcMeshCoord,
    uint32_t MeshCols,
    uint32_t MeshRows,
    int32_t MaxPacketSzBytes,
    tt::tt_fabric::Topology Topology>
inline void dispatch_chip_uni_noc_uni_fused_sem_inc_1d(
    const uint32_t linearized_dest_mesh_coord,
    const uint32_t alignment,
    int32_t size_bytes,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    uint64_t noc_remote_semaphore_address,
    uint16_t increment_value,
    bool flush,
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header) {
    uint32_t distance =
        manhattan_distance<Topology, MeshCols, MeshRows>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    packet_header->to_chip_unicast(distance);

    uint32_t route = get_route<Topology, MeshCols, MeshRows>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    return dispatch_noc_uni_fused_sem_inc<MaxPacketSzBytes>(
        payload_l1_address,
        noc_payload_write_address,
        noc_remote_semaphore_address,
        size_bytes,
        increment_value,
        flush,
        fabric_connections[route],
        packet_header,
        alignment);
}

}  // namespace ttnn::operations::ccl::common
