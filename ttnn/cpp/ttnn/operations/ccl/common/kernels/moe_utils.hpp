// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <tuple>

#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

namespace ttnn::operations::ccl::common {

enum class Polarity : uint8_t {
    NEGATIVE,
    POSITIVE,
};
namespace routing_state {

struct PolarState {
    std::array<Polarity, 2> polarity_table = {Polarity::NEGATIVE, Polarity::NEGATIVE};

    inline Polarity reverse(Polarity p) { return p == Polarity::POSITIVE ? Polarity::NEGATIVE : Polarity::POSITIVE; }

    inline uint32_t polar_compare_stateful(
        uint32_t positive_distance,
        uint32_t positive_direction,
        uint32_t negative_distance,
        uint32_t negative_direction,
        uint32_t axis) {
        auto polarity = polarity_table[axis];
        uint32_t result = 0;
        if (polarity == Polarity::POSITIVE) {
            result = positive_distance <= negative_distance ? positive_direction : negative_direction;
        } else {
            result = positive_distance < negative_distance ? positive_direction : negative_direction;
        }
        if (positive_distance == negative_distance) {
            polarity_table[axis] = reverse(polarity);
        }
        return result;
    }
};

PolarState polar_state;
}

struct MuxSyncCoreArgs {
    static constexpr uint16_t is_sync_core_arg_offset = 1;
    static constexpr uint16_t termination_sync_address_arg_offset = 10;
    static constexpr uint16_t termination_master_noc_x_arg_offset = 15;
    static constexpr uint16_t termination_master_noc_y_arg_offset = 16;

public:
    const bool is_sync_core;
    uint32_t termination_sync_address;
    const uint32_t termination_master_noc_x;
    const uint32_t termination_master_noc_y;

    MuxSyncCoreArgs(const uint32_t arg_idx) :
        is_sync_core(get_arg_val<uint32_t>(arg_idx + is_sync_core_arg_offset)),
        termination_sync_address(get_semaphore(get_arg_val<uint32_t>(arg_idx + termination_sync_address_arg_offset))),
        termination_master_noc_x(get_arg_val<uint32_t>(arg_idx + termination_master_noc_x_arg_offset)),
        termination_master_noc_y(get_arg_val<uint32_t>(arg_idx + termination_master_noc_y_arg_offset)) {};
};

// 17 RT args per active mux sender from ttnn::ccl::fabric_mux_connection_rt_args
template <uint32_t RtArgIncrement, uint32_t MxXRtOffset, uint32_t MxYRtOffset>
struct MuxBaseArgsImpl {
private:
    static constexpr uint16_t fabric_mux_x_arg_offset = MxXRtOffset;
    static constexpr uint16_t fabric_mux_y_arg_offset = MxYRtOffset;

protected:
    static constexpr uint32_t all_rt_arg_count = RtArgIncrement;

public:
    const uint8_t fabric_mux_x;
    const uint8_t fabric_mux_y;

    MuxBaseArgsImpl(const uint32_t arg_idx) :
        fabric_mux_x(get_arg_val<uint32_t>(arg_idx + fabric_mux_x_arg_offset)),
        fabric_mux_y(get_arg_val<uint32_t>(arg_idx + fabric_mux_y_arg_offset)) {};

    static void increment(uint32_t& arg_idx) { arg_idx += all_rt_arg_count; };
};
// 17 RT args per active mux sender from ttnn::ccl::fabric_mux_connection_rt_args
using MuxConnectionBaseArgs = MuxBaseArgsImpl<17, 2, 3>;

template <size_t MuxStatusAddress>
struct MuxConnectionStatusArgs : MuxConnectionBaseArgs {
private:
    // rt arg offsets from ttnn::ccl::fabric_mux_connection_rt_args
    static constexpr uint16_t local_fabric_mux_status_address_arg_offset = 11;

public:
    static constexpr size_t fabric_mux_status_address = MuxStatusAddress;
    uint32_t local_fabric_mux_status_address;

    MuxConnectionStatusArgs(const uint32_t arg_idx) :
        MuxConnectionBaseArgs(arg_idx),
        local_fabric_mux_status_address(
            get_semaphore(get_arg_val<uint32_t>(arg_idx + local_fabric_mux_status_address_arg_offset))) {};
};

template <uint8_t MuxNumBuffersPerChannel, size_t MuxChannelBufferSize, size_t MuxStatusAddress>
struct MuxConnectionArgs : MuxConnectionStatusArgs<MuxStatusAddress> {
private:
    // rt arg offsets from ttnn::ccl::fabric_mux_connection_rt_args
    static constexpr uint32_t fabric_mux_channel_id_arg_offset = 9;
    static constexpr uint32_t fabric_mux_channel_base_address_arg_offset = 4;
    static constexpr uint32_t fabric_mux_connection_info_address_arg_offset = 5;
    static constexpr uint32_t fabric_mux_connection_handshake_address_arg_offset = 6;
    static constexpr uint32_t fabric_mux_flow_control_address_arg_offset = 7;
    static constexpr uint32_t fabric_mux_buffer_index_address_arg_offset = 8;
    static constexpr uint32_t local_flow_control_address_arg_offset = 12;
    static constexpr uint32_t local_teardown_address_arg_offset = 13;
    static constexpr uint32_t local_buffer_index_address_arg_offset = 14;

public:
    const uint8_t fabric_mux_channel_id;
    static constexpr uint8_t fabric_mux_num_buffers_per_channel = MuxNumBuffersPerChannel;
    static constexpr size_t fabric_mux_channel_buffer_size_bytes = MuxChannelBufferSize;
    const size_t fabric_mux_channel_base_address;
    const size_t fabric_mux_connection_info_address;
    const size_t fabric_mux_connection_handshake_address;
    const size_t fabric_mux_flow_control_address;
    const size_t fabric_mux_buffer_index_address;
    uint32_t local_flow_control_address;
    uint32_t local_teardown_address;
    uint32_t local_buffer_index_address;

    MuxConnectionArgs(const uint32_t arg_idx) :
        MuxConnectionStatusArgs<MuxStatusAddress>(arg_idx),
        fabric_mux_channel_id(get_arg_val<uint32_t>(arg_idx + fabric_mux_channel_id_arg_offset)),
        fabric_mux_channel_base_address(get_arg_val<uint32_t>(arg_idx + fabric_mux_channel_base_address_arg_offset)),
        fabric_mux_connection_info_address(
            get_arg_val<uint32_t>(arg_idx + fabric_mux_connection_info_address_arg_offset)),
        fabric_mux_connection_handshake_address(
            get_arg_val<uint32_t>(arg_idx + fabric_mux_connection_handshake_address_arg_offset)),
        fabric_mux_flow_control_address(get_arg_val<uint32_t>(arg_idx + fabric_mux_flow_control_address_arg_offset)),
        fabric_mux_buffer_index_address(get_arg_val<uint32_t>(arg_idx + fabric_mux_buffer_index_address_arg_offset)),
        local_flow_control_address(
            get_semaphore(get_arg_val<uint32_t>(arg_idx + local_flow_control_address_arg_offset))),
        local_teardown_address(get_semaphore(get_arg_val<uint32_t>(arg_idx + local_teardown_address_arg_offset))),
        local_buffer_index_address(
            get_semaphore(get_arg_val<uint32_t>(arg_idx + local_buffer_index_address_arg_offset))) {};
};

// 2 RT args per mux worker from ttnn::operations::ccl::moe::detail::add_termination_master_rt_args
using MuxDisconnectionBaseArgs = MuxBaseArgsImpl<2, 0, 1>;

template <size_t TerminationSignalAddress>
struct MuxTerminationArgs : MuxDisconnectionBaseArgs {
    static constexpr size_t fabric_mux_termination_signal_address = TerminationSignalAddress;

    MuxTerminationArgs(const uint32_t arg_idx) : MuxDisconnectionBaseArgs(arg_idx) {};
};

template <size_t Size, uint8_t MuxNumBuffersPerChannel, size_t MuxChannelBufferSize, size_t MuxStatusAddress>
inline void open_direction_connections_async(
    const std::array<bool, Size>& directions,
    std::array<WorkerToFabricMuxSender<MuxNumBuffersPerChannel>, Size>& connections,
    uint32_t rt_args_idx) {
    for (uint32_t i = 0; i < Size; ++i) {
        if (directions[i]) {
            MuxConnectionArgs<MuxNumBuffersPerChannel, MuxChannelBufferSize, MuxStatusAddress> args(rt_args_idx);

            connections[i] = tt::tt_fabric::build_connection_to_fabric_endpoint<MuxNumBuffersPerChannel>(
                args.fabric_mux_x,
                args.fabric_mux_y,
                args.fabric_mux_channel_id,
                args.fabric_mux_num_buffers_per_channel,
                args.fabric_mux_channel_buffer_size_bytes,
                args.fabric_mux_channel_base_address,
                args.fabric_mux_connection_info_address,
                args.fabric_mux_connection_handshake_address,
                args.fabric_mux_flow_control_address,
                args.fabric_mux_buffer_index_address,
                args.local_flow_control_address,
                args.local_teardown_address,
                args.local_buffer_index_address);

            args.increment(rt_args_idx);
        }
    }
}

template <size_t Size, uint8_t MuxNumBuffersPerChannel, size_t MuxStatusAddress>
inline void open_direction_connections_barrier(
    const std::array<bool, Size>& directions,
    std::array<WorkerToFabricMuxSender<MuxNumBuffersPerChannel>, Size>& connections,
    uint32_t rt_args_idx) {
    for (uint32_t i = 0; i < Size; ++i) {
        if (directions[i]) {
            MuxConnectionStatusArgs<MuxStatusAddress> args(rt_args_idx);

            // DPRINT << "OPENING MUX CORE: " << (uint32_t)args.fabric_mux_x << ", " << (uint32_t)args.fabric_mux_y
            //                    << "\n";

            tt::tt_fabric::wait_for_fabric_endpoint_ready(
                args.fabric_mux_x,
                args.fabric_mux_y,
                args.fabric_mux_status_address,
                args.local_fabric_mux_status_address);
            tt::tt_fabric::fabric_client_connect(connections[i]);
            args.increment(rt_args_idx);
        }
    }
}

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
inline void open_direction_connections_async(
    const std::array<bool, Size>& directions,
    std::array<WorkerToFabricEdmSender, Size>& connections,
    size_t& rt_args_idx) {
    for (uint32_t i = 0; i < Size; i++) {
        if (directions[i]) {
            connections[i] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
            connections[i].open_start();
        }
    }
}

template <size_t Size>
inline void open_direction_connections_barrier(
    const std::array<bool, Size>& directions, std::array<WorkerToFabricEdmSender, Size>& connections) {
    for (uint32_t i = 0; i < Size; i++) {
        if (directions[i]) {
            connections[i].open_finish();
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

template <size_t Size, uint8_t MuxNumBuffersPerChannel, size_t TerminationSignalAddress, size_t NumMuxWorkers = 0>
inline void close_direction_connections(
    const std::array<bool, Size>& directions,
    std::array<WorkerToFabricMuxSender<MuxNumBuffersPerChannel>, Size>& connections,
    bool is_sync_core,
    uint32_t arg_idx = 0) {
    for (uint32_t i = 0; i < Size; ++i) {
        if (directions[i]) {
            tt::tt_fabric::fabric_client_disconnect(connections[i]);
        }
    }

    if (is_sync_core) {
        // fast forward past the normal mux rt args to the list of coordinates needed by termination master
        for (uint32_t i = 0; i < Size; ++i) {
            if (directions[i]) {
                MuxConnectionBaseArgs::increment(arg_idx);
            }
        }
        for (uint32_t i = 0; i < NumMuxWorkers; ++i) {
            MuxTerminationArgs<TerminationSignalAddress> args(arg_idx);

            // DPRINT << "CLOSING MUX CORE: " << (uint32_t)args.fabric_mux_x << ", " << (uint32_t)args.fabric_mux_y
            //                    << "\n";

            tt::tt_fabric::fabric_endpoint_terminate(
                args.fabric_mux_x, args.fabric_mux_y, args.fabric_mux_termination_signal_address);

            args.increment(arg_idx);
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
uint32_t topological_distance(uint32_t position_1, uint32_t position_2, uint32_t axis_size) {
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

template <uint32_t AxisSize>
uint32_t directional_wrap_distance(uint32_t position_1, uint32_t position_2, Polarity polarity) {
    if (polarity == Polarity::POSITIVE) {
        return (position_2 - position_1 + AxisSize) % AxisSize;
    } else {
        return (position_1 - position_2 + AxisSize) % AxisSize;
    }
}

template <tt::tt_fabric::Topology Topology, uint32_t MeshRows, uint32_t MeshCols>
uint32_t manhattan_distance(uint32_t linearized_src_mesh_coord, uint32_t linearized_dest_mesh_coord) {
    auto [src_row, src_col] = get_mesh_coords<MeshRows, MeshCols>(linearized_src_mesh_coord);
    auto [dest_row, dest_col] = get_mesh_coords<MeshRows, MeshCols>(linearized_dest_mesh_coord);
    return topological_distance<Topology>(src_row, dest_row, MeshRows) +
           topological_distance<Topology>(src_col, dest_col, MeshCols);
}

template <tt::tt_fabric::Topology Topology, uint32_t MeshRows, uint32_t MeshCols>
uint32_t get_route(uint32_t linearized_src_mesh_coord, uint32_t linearized_dest_mesh_coord) {
    auto [src_row, src_col] = get_mesh_coords<MeshRows, MeshCols>(linearized_src_mesh_coord);
    auto [dest_row, dest_col] = get_mesh_coords<MeshRows, MeshCols>(linearized_dest_mesh_coord);
    // default_polary is for ties in a ring
    // if default is positive, then for a E-W tie, we go East, and for a N-S tie, we go South
    // if default is negative, then for a E-W tie, we go West, and for a N-S tie, we go North
    if (src_row == dest_row) {
        if constexpr (!has_wrap_around<Topology>()) {
            return src_col < dest_col ? eth_chan_directions::EAST : eth_chan_directions::WEST;
        } else {
            // with wrap around, we can go either East or West. Choose the shorter route
            uint32_t east_distance = directional_wrap_distance<MeshCols>(src_col, dest_col, Polarity::POSITIVE);
            uint32_t west_distance = directional_wrap_distance<MeshCols>(src_col, dest_col, Polarity::NEGATIVE);
            return routing_state::polar_state.polar_compare_stateful(
                east_distance, eth_chan_directions::EAST, west_distance, eth_chan_directions::WEST, 1);
        }
    } else {
        if constexpr (!has_wrap_around<Topology>()) {
            return src_row < dest_row ? eth_chan_directions::SOUTH : eth_chan_directions::NORTH;
        } else {
            // with wrap around, we can go either North or South. Choose the shorter route
            uint32_t south_distance = directional_wrap_distance<MeshRows>(src_row, dest_row, Polarity::POSITIVE);
            uint32_t north_distance = directional_wrap_distance<MeshRows>(src_row, dest_row, Polarity::NEGATIVE);
            return routing_state::polar_state.polar_compare_stateful(
                south_distance, eth_chan_directions::SOUTH, north_distance, eth_chan_directions::NORTH, 0);
        }
    }
}

template <uint32_t FabricMaxPacketSzBytes, typename AddrGenType, class SenderType>
inline void fabric_send_noc_unicast(
    AddrGenType addrgen,
    SenderType& fabric_connection,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t payload_l1_address,
    uint64_t noc_page,
    int32_t size_bytes,
    uint32_t alignment,
    uint32_t offset = 0) {
    while (size_bytes > 0) {
        uint32_t curr_packet_size = std::min(FabricMaxPacketSzBytes, (uint32_t)size_bytes);

        tt::tt_fabric::linear::to_noc_unicast_write(
            align(curr_packet_size, alignment), packet_header, noc_page, addrgen, offset);
        perform_payload_send<false, false, SenderType>(
            fabric_connection, payload_l1_address, curr_packet_size, packet_header);

        payload_l1_address += curr_packet_size;
        offset += curr_packet_size;
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
template <
    uint32_t SrcChipId,
    uint32_t MeshRows,
    uint32_t MeshCols,
    int32_t FabricMaxPacketSzBytes,
    typename AddrGenType,
    typename SenderType>
inline void fabric_send_chip_unicast_noc_unicast(
    AddrGenType addrgen,
    std::array<SenderType, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    const uint32_t dest_chip_id,
    const uint32_t dest_mesh_id,
    uint32_t payload_l1_address,
    uint32_t noc_payload_page,
    int32_t size_bytes,
    const uint32_t alignment,
    uint32_t offset = 0) {
    const uint32_t route = get_next_hop_router_direction(dest_mesh_id, dest_chip_id);

    // Populate packet header with routing information
    fabric_set_unicast_route((HybridMeshPacketHeader*)packet_header, dest_chip_id, dest_mesh_id);

    fabric_send_noc_unicast<FabricMaxPacketSzBytes>(
        addrgen,
        fabric_connections[route],
        packet_header,
        payload_l1_address,
        noc_payload_page,
        size_bytes,
        alignment,
        offset);
}

template <uint32_t FabricMaxPacketSzBytes>
inline void l1_only_fabric_send_noc_unicast_with_semaphore(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    uint64_t noc_remote_semaphore_address,
    int32_t size_bytes,
    uint32_t alignment,
    uint32_t increment_value,
    bool flush) {
    // This api is only for L1 as it can cause a DRAM hang in blackhole
    while (size_bytes > 0) {
        uint32_t curr_packet_size = std::min(FabricMaxPacketSzBytes, (uint32_t)size_bytes);

        if ((uint32_t)size_bytes == curr_packet_size) {
            // Fill header for fused unicast + atomic increment command when it is the last packet
            packet_header->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader(
                    noc_payload_write_address, noc_remote_semaphore_address, increment_value, flush),
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

template <uint32_t FabricMaxPacketSzBytes, typename AddrGenType>
inline void fabric_send_noc_unicast_with_semaphore(
    AddrGenType addrgen,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t payload_l1_address,
    uint32_t payload_page_id,
    uint64_t noc_remote_semaphore_address,
    int32_t size_bytes,
    uint32_t alignment,
    uint32_t increment_value,
    bool flush,
    uint32_t offset = 0) {
    while (size_bytes > 0) {
        uint32_t curr_packet_size = std::min(FabricMaxPacketSzBytes, (uint32_t)size_bytes);

        if ((uint32_t)size_bytes == curr_packet_size) {
            // Fill header for fused unicast + atomic increment command when it is the last packet
            tt::tt_fabric::linear::to_noc_fused_unicast_write_atomic_inc(
                align(curr_packet_size, alignment),
                packet_header,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{noc_remote_semaphore_address, increment_value, flush},
                payload_page_id,
                addrgen,
                offset);
        } else {
            // Fill header for unicast write command when it is not the last packet
            tt::tt_fabric::linear::to_noc_unicast_write(
                align(curr_packet_size, alignment), packet_header, payload_page_id, addrgen, offset);
        }

        // Send payload followed by header over the fabric.
        perform_payload_send<true>(fabric_connection, payload_l1_address, curr_packet_size, packet_header);
        payload_l1_address += curr_packet_size;
        offset += curr_packet_size;
        size_bytes -= curr_packet_size;
    }
}

// Insert helper that handles the remote-device metadata path with fused atomic increment
template <uint32_t SrcChipId, uint32_t MeshRows, uint32_t MeshCols, uint32_t FabricMaxPacketSzBytes>
inline void l1_only_fabric_send_chip_unicast_noc_unicast_with_semaphore(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t dest_chip_id,
    uint32_t dest_mesh_id,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    uint64_t noc_remote_semaphore_address,
    int32_t size_bytes,
    uint32_t alignment,
    uint32_t increment_value,
    bool flush) {
    // This api is only for L1 as it can cause a DRAM hang in blackhole
    uint32_t route = get_next_hop_router_direction(dest_mesh_id, dest_chip_id);

    // Populate packet header with routing information
    fabric_set_unicast_route((HybridMeshPacketHeader*)packet_header, dest_chip_id, dest_mesh_id);

    return l1_only_fabric_send_noc_unicast_with_semaphore<FabricMaxPacketSzBytes>(
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

// Insert helper that handles the remote-device metadata path with fused atomic increment
template <
    uint32_t SrcChipId,
    uint32_t MeshRows,
    uint32_t MeshCols,
    uint32_t FabricMaxPacketSzBytes,
    typename AddrGenType>
inline void fabric_send_chip_unicast_noc_unicast_with_semaphore(
    AddrGenType addrgen,
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t dest_chip_id,
    uint32_t dest_mesh_id,
    uint32_t payload_l1_address,
    uint64_t noc_payload_page,
    uint64_t noc_remote_semaphore_address,
    int32_t size_bytes,
    uint32_t alignment,
    uint32_t increment_value,
    bool flush,
    uint32_t offset = 0) {
    uint32_t route = get_next_hop_router_direction(dest_mesh_id, dest_chip_id);

    // Populate packet header with routing information
    fabric_set_unicast_route((HybridMeshPacketHeader*)packet_header, dest_chip_id, dest_mesh_id);

    return fabric_send_noc_unicast_with_semaphore<FabricMaxPacketSzBytes>(
        addrgen,
        fabric_connections[route],
        packet_header,
        payload_l1_address,
        noc_payload_page,
        noc_remote_semaphore_address,
        size_bytes,
        alignment,
        increment_value,
        flush,
        offset);
}

// Fabric send for NOC unicast semaphore increment only (no payload)
template <uint32_t SrcChipId, uint32_t MeshRows, uint32_t MeshCols, typename SenderType>
inline void fabric_send_chip_unicast_noc_unicast_semaphore_only(
    std::array<SenderType, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t dest_chip_id,
    uint32_t dest_mesh_id,
    uint64_t noc_remote_semaphore_address,
    uint32_t increment_value,
    bool flush) {
    // Set up packet header for semaphore increment
    packet_header->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{noc_remote_semaphore_address, increment_value, flush});

    uint32_t route = get_next_hop_router_direction(dest_mesh_id, dest_chip_id);

    // Populate packet header with routing information
    fabric_set_unicast_route((HybridMeshPacketHeader*)packet_header, dest_chip_id, dest_mesh_id);

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
    int32_t FabricMaxPacketSzBytes,
    typename AddrGenType,
    typename SenderType>
inline void fabric_send_chip_unicast_noc_unicast_1d(
    AddrGenType addrgen,
    std::array<SenderType, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    const uint32_t linearized_dest_mesh_coord,
    uint32_t payload_l1_address,
    uint32_t noc_payload_page,
    int32_t size_bytes,
    const uint32_t alignment,
    uint32_t offset = 0) {
    uint32_t distance =
        manhattan_distance<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    fabric_set_unicast_route<false>((volatile tt_l1_ptr LowLatencyPacketHeader*)packet_header, distance);

    uint32_t route = get_route<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    fabric_send_noc_unicast<FabricMaxPacketSzBytes>(
        addrgen,
        fabric_connections[route],
        packet_header,
        payload_l1_address,
        noc_payload_page,
        size_bytes,
        alignment,
        offset);
}

template <
    uint32_t LinearizedSrcMeshCoord,
    tt::tt_fabric::Topology Topology,
    uint32_t MeshRows,
    uint32_t MeshCols,
    int32_t FabricMaxPacketSzBytes>
inline void l1_only_fabric_send_chip_unicast_noc_unicast_with_semaphore_1d(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    const uint32_t linearized_dest_mesh_coord,
    uint32_t payload_l1_address,
    uint64_t noc_payload_write_address,
    uint64_t noc_remote_semaphore_address,
    int32_t size_bytes,
    const uint32_t alignment,
    uint32_t increment_value,
    bool flush) {
    // This api is only for L1 as it can cause a DRAM hang in blackhole
    uint32_t distance =
        manhattan_distance<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    fabric_set_unicast_route<false>((volatile tt_l1_ptr LowLatencyPacketHeader*)packet_header, distance);

    uint32_t route = get_route<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    return l1_only_fabric_send_noc_unicast_with_semaphore<FabricMaxPacketSzBytes>(
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

template <
    uint32_t LinearizedSrcMeshCoord,
    tt::tt_fabric::Topology Topology,
    uint32_t MeshRows,
    uint32_t MeshCols,
    int32_t FabricMaxPacketSzBytes,
    typename AddrGenType>
inline void fabric_send_chip_unicast_noc_unicast_with_semaphore_1d(
    AddrGenType addrgen,
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    const uint32_t linearized_dest_mesh_coord,
    uint32_t payload_l1_address,
    uint32_t noc_payload_page,
    uint64_t noc_remote_semaphore_address,
    int32_t size_bytes,
    const uint32_t alignment,
    uint32_t increment_value,
    bool flush,
    uint32_t offset = 0) {
    uint32_t distance =
        manhattan_distance<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    fabric_set_unicast_route<false>((volatile tt_l1_ptr LowLatencyPacketHeader*)packet_header, distance);

    uint32_t route = get_route<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    return fabric_send_noc_unicast_with_semaphore<FabricMaxPacketSzBytes>(
        addrgen,
        fabric_connections[route],
        packet_header,
        payload_l1_address,
        noc_payload_page,
        noc_remote_semaphore_address,
        size_bytes,
        alignment,
        increment_value,
        flush,
        offset);
}

// Fabric send for NOC unicast semaphore increment only in 1D topology (no payload)
template <
    uint32_t LinearizedSrcMeshCoord,
    tt::tt_fabric::Topology Topology,
    uint32_t MeshRows,
    uint32_t MeshCols,
    typename SenderType>
inline void fabric_send_chip_unicast_noc_unicast_semaphore_only_1d(
    std::array<SenderType, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    const uint32_t linearized_dest_mesh_coord,
    uint64_t noc_remote_semaphore_address,
    uint32_t increment_value,
    bool flush) {
    // Set up packet header for semaphore increment
    packet_header->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{noc_remote_semaphore_address, increment_value, flush});

    uint32_t distance =
        manhattan_distance<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);
    fabric_set_unicast_route<false>((volatile tt_l1_ptr LowLatencyPacketHeader*)packet_header, distance);

    uint32_t route = get_route<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coord);

    // Send only the packet header (for semaphore increment)
    fabric_connections[route].wait_for_empty_write_slot();
    fabric_connections[route].send_payload_flush_blocking_from_address(
        reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));
}

template <typename T, uint32_t Size, bool ReturnIdx>
inline auto find_if(volatile tt_l1_ptr T* ptr, const uint32_t val) {
    for (uint32_t i = 0; i < Size; ++i) {
        invalidate_l1_cache();
        if (ptr[i] == val) {
            if constexpr (ReturnIdx) {
                return std::make_tuple(true, i);
            } else {
                return true;
            }
        }
    }
    if constexpr (ReturnIdx) {
        return std::make_tuple(false, 0ul);
    } else {
        return false;
    }
}

// Send initialization semaphore to configured target devices for synchronization
template <
    uint32_t LinearizedSrcMeshCoord,
    tt::tt_fabric::Topology Topology,
    uint32_t SrcChipId,
    uint32_t MeshRows,
    uint32_t MeshCols,
    ReplicateGroup Axis,
    uint32_t NumDevices,
    typename SenderType>
inline void send_init_semaphore_to_configured_targets(
    std::array<SenderType, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    const uint8_t dest_chip_ids[NumDevices],
    const uint8_t dest_mesh_ids[NumDevices],
    uint64_t init_noc_semaphore_addr) {
    uint32_t device_begin_idx = 0;
    uint32_t device_end_idx = NumDevices;
    uint32_t device_stride = 1;

    constexpr uint32_t row = LinearizedSrcMeshCoord / MeshCols;
    constexpr uint32_t col = LinearizedSrcMeshCoord % MeshCols;

    if constexpr (Axis == ReplicateGroup::COLS) {
        device_begin_idx = col;
        device_end_idx = col + MeshRows * MeshCols;
        device_stride = MeshCols;
    } else if constexpr (Axis == ReplicateGroup::ROWS) {
        device_begin_idx = row * MeshCols;
        device_end_idx = row * MeshCols + MeshCols;
        device_stride = 1;
    }

    for (uint32_t device_idx = device_begin_idx; device_idx < device_end_idx; device_idx += device_stride) {
        if (device_idx == LinearizedSrcMeshCoord) {
            continue;
        } else if (is_configured_target<LinearizedSrcMeshCoord, MeshRows, MeshCols, Axis>(device_idx)) {
            if constexpr (is_1d_topology<Topology>()) {
                fabric_send_chip_unicast_noc_unicast_semaphore_only_1d<
                    LinearizedSrcMeshCoord,
                    Topology,
                    MeshRows,
                    MeshCols>(fabric_connections, packet_header, device_idx, init_noc_semaphore_addr, 1, false);
            } else {
                const auto& dest_chip_id = dest_chip_ids[device_idx];
                const auto& dest_mesh_id = dest_mesh_ids[device_idx];
                fabric_send_chip_unicast_noc_unicast_semaphore_only<SrcChipId, MeshRows, MeshCols>(
                    fabric_connections, packet_header, dest_chip_id, dest_mesh_id, init_noc_semaphore_addr, 1, false);
            }
        }
    }
}

// Send a sparse multicast packet to destinations along a specific direction
// Currently only supports 1D Ring topology
// Supply hop mask and direction to the function
// Hop mask is a bitmask of which destination hops to write to, read from LSb to MSb.
// For example, if we want to write East from device 0 to devices 1 and 4 only:
// 0 --> 1 --> 2 --> 3 --> 4 --- 5
//      [X]               [X]
// We would set a hop mask of 0b01001 and set direction to eth_chan_directions::EAST.
template <int32_t FabricMaxPacketSzBytes, typename AddrGenType, typename FabricConnectionsArrayType>
inline void fabric_send_chip_sparse_multicast_noc_unicast_1d_in_direction(
    // Fabric parameters
    AddrGenType addrgen,
    FabricConnectionsArrayType& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint16_t fabric_mcast_hop_mask,
    uint32_t fabric_direction,  // eth_chan_directions index
    // NoC Parameters
    uint32_t payload_l1_address,
    uint32_t noc_payload_page,
    int32_t size_bytes,
    const uint32_t alignment,
    uint32_t offset = 0) {
    // Separate the list of destinations by direction relative to the source chip
    // Set the sparse multicast fabric route in the packet header
    fabric_set_sparse_multicast_route((volatile tt_l1_ptr LowLatencyPacketHeader*)packet_header, fabric_mcast_hop_mask);

    // Configure and send the packet
    fabric_send_noc_unicast<FabricMaxPacketSzBytes>(
        addrgen,
        fabric_connections[fabric_direction],
        packet_header,
        payload_l1_address,
        noc_payload_page,
        size_bytes,
        alignment,
        offset);
}

// Calculate the distance between two mesh coordinates, with the packet travelling in a specific direction.
// Positive Polarity: Refers to going "forward" (ascending order of coordinates).
// Eg. Going East (0->3) or South (0->2)
// Negative Polarity: Refers to going "backward" (descending order of coordinates).
// Eg. Going West (3->0) or North (2->0)
template <tt::tt_fabric::Topology Topology, uint32_t AxisSize>
inline uint32_t calculate_hops_direction_enforced_1D(uint32_t src_coord, uint32_t dest_coord, Polarity polarity) {
    // Currently this function has only been tested for 1D Ring topology.
    static_assert(
        has_wrap_around<Topology>(),
        "calculate_hops_direction_enforced_1D has only been tested for 1D topologies with wraparound links");
    return directional_wrap_distance<AxisSize>(src_coord, dest_coord, polarity);
}

// Given a list of destinations, generates a hop mask relative to the source chip
// Also determines the direction out of which the source chip should send the packet
// Assumes that all destinations are along the same axis as the source chip
template <
    uint32_t LinearizedSrcMeshCoord,
    tt::tt_fabric::Topology Topology,
    uint32_t MeshRows,
    uint32_t MeshCols,
    uint32_t MaxNumDestinations>
inline std::pair<uint16_t, uint32_t> get_fabric_mcast_hop_mask_and_direction(
    uint32_t linearized_dest_mesh_coords[MaxNumDestinations], uint32_t NumDestinations) {
    // Guard to make sure the number of destinations is not greater than the maximum number of destinations
    ASSERT(NumDestinations > 0 && NumDestinations <= MaxNumDestinations);

    // Figure out direction of this multicast packet
    // Direction will follow the path used for the first destination
    uint32_t routing_direction =
        get_route<Topology, MeshRows, MeshCols>(LinearizedSrcMeshCoord, linearized_dest_mesh_coords[0]);

    // Determine polarity of the direction
    // Positive polarity means going "forward" (ascending order of coordinates). Eg. Going East (0->3) or South (0->2)
    // Negative polarity means going "backward" (descending order of coordinates). Eg. Going West (3->0) or North (2->0)
    Polarity direction_polarity =
        (routing_direction == eth_chan_directions::EAST || routing_direction == eth_chan_directions::SOUTH)
            ? Polarity::POSITIVE
            : Polarity::NEGATIVE;
    bool travelling_ew =
        (routing_direction == eth_chan_directions::EAST || routing_direction == eth_chan_directions::WEST);
    // Generate the multicast packet's hop mask
    auto [src_row, src_col] = get_mesh_coords<MeshRows, MeshCols>(LinearizedSrcMeshCoord);
    uint16_t fabric_mcast_hop_mask = 0;
    for (uint32_t i = 0; i < NumDestinations; i++) {
        auto [dest_row, dest_col] = get_mesh_coords<MeshRows, MeshCols>(linearized_dest_mesh_coords[i]);
        // Calculate the number of hops between source chip and destination
        uint32_t num_hops;
        if (travelling_ew) {
            num_hops = calculate_hops_direction_enforced_1D<Topology, MeshCols>(src_col, dest_col, direction_polarity);
        } else {
            num_hops = calculate_hops_direction_enforced_1D<Topology, MeshRows>(src_row, dest_row, direction_polarity);
        }
        // Guard to make sure that num_hops is a valid result
        ASSERT(num_hops > 0 && num_hops <= 16);  // Currently fabric sparse multicast only supports up to 16 hops
        // Set the corresponding bit in the fabric multicast hop mask
        fabric_mcast_hop_mask |= (1 << (num_hops - 1));
    }
    return std::make_pair(fabric_mcast_hop_mask, routing_direction);
}

// Sends a fabric sparse multicast packet to destinations along a specific direction
// Currently only supports 1D Ring topology
// Supply the list of destinations and the number of destinations to the function
template <
    uint32_t LinearizedSrcMeshCoord,
    tt::tt_fabric::Topology Topology,
    uint32_t MeshRows,
    uint32_t MeshCols,
    uint32_t FabricMaxNumDestinations,
    int32_t FabricMaxPacketSzBytes,
    typename AddrGenType,
    typename FabricConnectionsArrayType>
inline void fabric_send_chip_sparse_multicast_noc_unicast_1d(
    // Fabric parameters
    AddrGenType addrgen,
    FabricConnectionsArrayType& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t linearized_dest_mesh_coords[FabricMaxNumDestinations],
    uint32_t NumDestinations,
    // NoC Parameters
    uint32_t payload_l1_address,
    uint32_t noc_payload_page,
    int32_t size_bytes,
    const uint32_t alignment,
    uint32_t offset = 0) {
    // Generate the fabric multicast hop mask and get the direction the packet should be sent
    uint16_t fabric_mcast_hop_mask;
    uint32_t routing_direction;
    std::tie(fabric_mcast_hop_mask, routing_direction) = get_fabric_mcast_hop_mask_and_direction<
        LinearizedSrcMeshCoord,
        Topology,
        MeshRows,
        MeshCols,
        FabricMaxNumDestinations>(linearized_dest_mesh_coords, NumDestinations);

    // Send the packet
    fabric_send_chip_sparse_multicast_noc_unicast_1d_in_direction<FabricMaxPacketSzBytes>(
        addrgen,
        fabric_connections,
        packet_header,
        fabric_mcast_hop_mask,
        routing_direction,
        payload_l1_address,
        noc_payload_page,
        size_bytes,
        alignment,
        offset);
}

}  // namespace ttnn::operations::ccl::common
