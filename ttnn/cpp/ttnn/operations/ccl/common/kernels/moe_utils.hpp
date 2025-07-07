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
    uint32_t input_token_read_addr,
    uint64_t output_token_write_addr,
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* token_unicast_packet_header) {
    // Clear the header buffer region.

    const uint32_t route = get_next_hop_router_direction(dest_mesh_id, dest_chip_id);

    // Populate packet header with routing information
    fabric_set_unicast_route(
        const_cast<LowLatencyMeshPacketHeader*>(token_unicast_packet_header),
        static_cast<eth_chan_directions>(fabric_connections[route].direction),
        SrcChipId,
        dest_chip_id,
        dest_mesh_id,
        MeshCols);

    while (size_bytes > 0) {
        uint32_t curr_packet_size = std::min(MaxPacketSzBytes, size_bytes);

        token_unicast_packet_header->to_noc_unicast_write(
            NocUnicastCommandHeader{output_token_write_addr}, align(curr_packet_size, alignment));

        fabric_connections[route].wait_for_empty_write_slot();

        fabric_connections[route].send_payload_without_header_non_blocking_from_address(
            input_token_read_addr, curr_packet_size);

        fabric_connections[route].send_payload_flush_blocking_from_address(
            (uint32_t)token_unicast_packet_header, sizeof(PACKET_HEADER_TYPE));

        input_token_read_addr += curr_packet_size;
        output_token_write_addr += curr_packet_size;
        size_bytes -= curr_packet_size;
    }
}
}  // namespace ttnn::operations::ccl::common
