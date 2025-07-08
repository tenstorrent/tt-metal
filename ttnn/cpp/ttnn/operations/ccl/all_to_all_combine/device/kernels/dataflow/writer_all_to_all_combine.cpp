// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"

#include "../common.hpp"

using tt::tt_fabric::NocUnicastAtomicIncCommandHeader;
using tt::tt_fabric::NocUnicastCommandHeader;
using tt::tt_fabric::WorkerToFabricEdmSender;

namespace detail {

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

template <uint32_t SourceChipId, uint32_t BatchSize, uint32_t MeshCols, uint32_t MeshRows, ReplicateGroup Axis>
inline uint32_t get_device_idx_from_batch_idx(const uint32_t b) {
    constexpr uint32_t Replicate_Group = (Axis == ReplicateGroup::NONE)   ? MeshCols * MeshRows
                                         : (Axis == ReplicateGroup::COLS) ? MeshRows
                                                                          : MeshCols;

    constexpr uint32_t Batch_Per_Device = BatchSize / Replicate_Group;
    const uint32_t device_in_group = b / Batch_Per_Device;

    if constexpr (Axis == ReplicateGroup::NONE) {
        return device_in_group;
    } else if (Axis == ReplicateGroup::ROWS) {
        return (SourceChipId / MeshCols) * MeshCols + device_in_group;
    } else {
        return device_in_group * MeshCols + SourceChipId % MeshCols;
    }
}

// output per device is [K, B/replicate_group, 1, H]
template <uint32_t BatchSize, uint32_t MeshCols, uint32_t MeshRows, ReplicateGroup Axis>
inline uint32_t get_output_page_idx(const uint32_t b, const uint32_t k) {
    uint32_t batch_devices;
    if constexpr (Axis == ReplicateGroup::NONE) {
        batch_devices = MeshCols * MeshRows;
    } else if constexpr (Axis == ReplicateGroup::ROWS) {
        batch_devices = MeshCols;
    } else {
        batch_devices = MeshRows;
    }

    const uint32_t batch_per_device = BatchSize / batch_devices;
    return k * batch_per_device + b % batch_per_device;
}

// commonize me!
template <size_t Size>
inline void open_direction_connections(
    const std::array<bool, Size>& directions,
    std::array<WorkerToFabricEdmSender, Size>& connections,
    size_t rt_args_idx) {
    for (uint32_t i = 0; i < 4; i++) {
        if (directions[i]) {
            connections[i] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
            connections[i].open();
        }
    }
}

// commonize me!

template <size_t Size>
inline void close_direction_connections(
    const std::array<bool, Size>& directions, std::array<WorkerToFabricEdmSender, Size>& connections) {
    for (size_t i = 0; i < Size; ++i) {
        if (directions[i]) {
            connections[i].close();
        }
    }
}

// ! commonize me!
/*
enum eth_chan_directions {
    EAST = 0,
    WEST = 1,
    NORTH = 2,
    SOUTH = 3,
    COUNT = 4,
};*/

template <uint32_t SrcChipId, uint32_t MeshCols, uint32_t MeshRows>
inline eth_chan_directions get_direction(uint32_t dest_chip_id) {
    // if along the same row, we go east or west
    eth_chan_directions direction = eth_chan_directions::COUNT;

    constexpr uint32_t Src_Row = SrcChipId / MeshCols, Src_Col = SrcChipId % MeshCols;
    if (Src_Row == dest_chip_id / MeshCols) {
        direction = SrcChipId < dest_chip_id ? eth_chan_directions::EAST : eth_chan_directions::WEST;
    }
    // if along the same column, we go north or south
    else if (Src_Col == dest_chip_id % MeshCols) {
        direction = SrcChipId > dest_chip_id ? eth_chan_directions::NORTH : eth_chan_directions::SOUTH;
    }
    // if not along the same row or column, we go north or south; north if dest_chip_id is smaller than src_chip_id
    else {
        direction = SrcChipId > dest_chip_id ? eth_chan_directions::NORTH : eth_chan_directions::SOUTH;
    }
    return direction;
}

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

    const uint32_t route = get_direction<SrcChipId, MeshCols, MeshRows>(dest_chip_id);

    // Populate packet header with routing information
    zero_l1_buf((uint32_t*)token_unicast_packet_header, sizeof(PACKET_HEADER_TYPE));
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
}  // namespace detail

void kernel_main() {
    using detail::ReplicateGroup;

    constexpr uint32_t metadata_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t local_experts_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t data_cb_id = get_compile_time_arg_val(3);

    constexpr uint32_t batch_size = get_compile_time_arg_val(4);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(5);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(6);
    constexpr uint32_t num_devices = get_compile_time_arg_val(7);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(8);
    constexpr uint32_t data_size_bytes = get_compile_time_arg_val(9);  // hidden dim * datum size
    constexpr uint32_t alignment = get_compile_time_arg_val(10);
    constexpr uint32_t output_is_dram = get_compile_time_arg_val(11);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(12);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(13);  // ew_dim
    constexpr uint32_t fabric_max_packet_size_bytes = get_compile_time_arg_val(14);

#ifdef REPLICATE_GROUP_AXIS
    constexpr ReplicateGroup replicate_axis = ReplicateGroup(REPLICATE_GROUP_AXIS);
    constexpr uint8_t replicate_group_devices =
        num_devices / (replicate_axis == ReplicateGroup::COLS ? mesh_cols : mesh_rows);
#else
    constexpr ReplicateGroup replicate_axis = ReplicateGroup::NONE;
    constexpr uint8_t replicate_group_devices = num_devices;
#endif

    constexpr uint8_t Num_Directions = 4;
    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    const std::array<bool, Num_Directions> directions = DIRECTIONS;

    const auto output_base_addr = get_arg_val<uint32_t>(0);
    const auto global_semaphore_addr = get_arg_val<uint32_t>(1);

    const uint32_t rt_arg_count = 2;
    std::array<WorkerToFabricEdmSender, Num_Directions> fabric_connections;
    detail::open_direction_connections(directions, fabric_connections, rt_arg_count);

    InterleavedAddrGen<output_is_dram> output_addrgen{
        .bank_base_address = output_base_addr, .page_size = data_size_bytes};

    volatile PACKET_HEADER_TYPE * packet_headers[2];
    for(uint8_t i =0;i<2;++i){
        cb_reserve_back(packet_header_cb_id,1);
        const uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
        packet_headers[i] = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
        cb_push_back(packet_header_cb_id,1);
    }

    cb_wait_front(local_experts_cb_id,1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(local_experts_cb_id));

    for (uint32_t b = 0; b < batch_size; ++b) {
        cb_wait_front(metadata_cb_id, 1);
        const uint32_t metadata_l1_addr = get_write_ptr(metadata_cb_id);
        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);

        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const auto & expert_idx = local_experts_ptr[e];
            const auto [found, k] = detail::find_if<uint16_t, selected_experts_k, true>(metadata_ptr, expert_idx);

            if (found) {
                cb_wait_front(data_cb_id,1);
                const uint32_t src_data_l1_ptr = get_read_ptr(data_cb_id);

                // figure out output page index, noc address.
                const uint32_t output_page_idx =
                    detail::get_output_page_idx<batch_size, mesh_cols, mesh_rows, replicate_axis>(b, k);
                const uint64_t output_noc_addr = get_noc_addr(output_page_idx, output_addrgen);

                // figure out which device to send data to and routing
                const auto dest_device_idx = detail::
                    get_device_idx_from_batch_idx<src_chip_id, batch_size, mesh_cols, mesh_rows, replicate_axis>(b);
                const auto& dest_chip_id = dest_chip_ids[dest_device_idx];

                if (dest_chip_id == src_chip_id) {
                    noc_async_write(src_data_l1_ptr,output_noc_addr,data_size_bytes);
                    noc_async_write_barrier();
                } else {
                    const auto route = detail::get_direction<src_chip_id, mesh_cols, mesh_rows>(dest_chip_id);
                    const auto& dest_mesh_id = dest_mesh_ids[dest_device_idx];
                    detail::
                        dispatch_input_remote_device<src_chip_id, mesh_cols, mesh_rows, fabric_max_packet_size_bytes>(
                            dest_chip_id,
                            dest_mesh_id,
                            alignment,
                            data_size_bytes,
                            src_data_l1_ptr,
                            output_noc_addr,
                            fabric_connections,
                            packet_headers[0]);
                }
                cb_pop_front(data_cb_id,1);
            }
        }
        cb_pop_front(metadata_cb_id, 1);
    }
    cb_pop_front(local_experts_cb_id, 1);

    const uint64_t global_noc_semaphore_addr = get_noc_addr(global_semaphore_addr);
    // "multicast" semaphore increment to let other devices know we are done
    for(uint32_t device_idx=0;device_idx < num_devices;++device_idx){
        const auto & dest_chip_id = dest_chip_ids[device_idx];

        if (dest_chip_id == src_chip_id) {
            noc_semaphore_inc(get_noc_addr(global_semaphore_addr), 1);
            noc_async_atomic_barrier();
        } else if (detail::is_configured_target<src_chip_id, mesh_cols, mesh_rows, replicate_axis>(dest_chip_id)) {
            const auto & dest_mesh_id = dest_mesh_ids[device_idx];
            const auto route = detail::get_direction<src_chip_id, mesh_cols, mesh_rows>(dest_chip_id);

            packet_headers[1]->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{global_noc_semaphore_addr, 1, 32, true});

            fabric_set_unicast_route(
                const_cast<LowLatencyMeshPacketHeader*>(packet_headers[1]),
                static_cast<eth_chan_directions>(fabric_connections[route].direction),
                src_chip_id,
                dest_chip_id,
                dest_mesh_id,
                mesh_cols);

            fabric_connections[route].wait_for_empty_write_slot();
            fabric_connections[route].send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(packet_headers[1]), sizeof(PACKET_HEADER_TYPE));
        }
    }

    detail::close_direction_connections(directions, fabric_connections);

    auto semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);
    noc_semaphore_wait(semaphore_ptr, replicate_group_devices);
    noc_semaphore_set(semaphore_ptr, 0);
}
