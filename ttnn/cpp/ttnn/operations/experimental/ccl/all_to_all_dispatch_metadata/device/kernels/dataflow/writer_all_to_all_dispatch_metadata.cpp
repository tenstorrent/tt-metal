// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

namespace detail {

inline void dispatch_input_local_device_flushed(
    uint32_t input_token_read_addr, uint64_t output_token_write_addr, uint32_t output_page_size) {
    noc_async_write(input_token_read_addr, output_token_write_addr, output_page_size);
    noc_async_writes_flushed();
}

// Insert helper that handles the local-device metadata path
inline void dispatch_metadata_local_device(
    uint32_t token_indices_address,
    uint64_t metadata_write_addr,
    uint32_t metadata_page_size,
    uint64_t global_noc_semaphore_address) {
    // send metadata to local device output buffer
    noc_async_write(token_indices_address, metadata_write_addr, metadata_page_size);
    noc_async_write_barrier();
    noc_semaphore_inc(global_noc_semaphore_address, 1);
    noc_async_atomic_barrier();
}

void zero_buffer_async(uint32_t write_addr, int bytes) {
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        uint32_t curr_bytes = std::min(bytes, MEM_ZEROS_SIZE);
        noc_async_read(zeros_noc_addr, write_addr, curr_bytes);
        write_addr += curr_bytes;
        bytes -= curr_bytes;
    }
}

void zero_buffer_barrier() { noc_async_read_barrier(); }

// Bidirectional fabric multicast atomic increment - sends to both positive and negative directions
// For a 1D ring with even number of devices, we multicast in both directions to cover all devices
// with just 2 packets instead of (dispatch_devices - 1) unicast packets
template <
    uint32_t LinearizedSrcMeshCoord,
    uint32_t MeshRows,
    uint32_t MeshCols,
    ttnn::operations::ccl::common::ReplicateGroup Axis,
    bool DoubleAntipodalAtomicInc = false>
FORCE_INLINE void fabric_multicast_bidirectional_atomic_inc_ring_1d(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint64_t semaphore_noc_addr) {
    using ttnn::operations::ccl::common::ReplicateGroup;
    const auto cmd_header = tt::tt_fabric::NocUnicastAtomicIncCommandHeader{semaphore_noc_addr, 1, true};

    // ReplicateGroup::COLS (axis=0): targets on same column, dispatch vertically (SOUTH/NORTH),
    // dispatch_devices=MeshRows ReplicateGroup::ROWS (axis=1): targets on same row, dispatch horizontally (EAST/WEST),
    // dispatch_devices=MeshCols
    constexpr uint32_t dispatch_devices =
        Axis == ttnn::operations::ccl::common::ReplicateGroup::COLS ? MeshRows : MeshCols;

    // Split the ring: positive direction gets half, negative direction gets the other half
    // For dispatch_devices = 16: positive gets 8, negative gets 7 (total 15 = dispatch_devices - 1) if
    // DoubleAntipodalAtomicInc is false For dispatch_devices = 16: positive gets 8, negative gets 8 (total 16 =
    // dispatch_devices) if DoubleAntipodalAtomicInc is true
    constexpr uint32_t positive_range = DoubleAntipodalAtomicInc ? (dispatch_devices + 1) / 2 : dispatch_devices / 2;
    constexpr uint32_t negative_range =
        DoubleAntipodalAtomicInc ? dispatch_devices - positive_range : (dispatch_devices - 1) - positive_range;

    // Determine directions based on axis:
    // COLS (axis=0): dispatch along column → SOUTH is positive, NORTH is negative
    // ROWS (axis=1): dispatch along row → EAST is positive, WEST is negative
    constexpr uint32_t positive_direction =
        Axis == ReplicateGroup::COLS ? eth_chan_directions::SOUTH : eth_chan_directions::EAST;
    constexpr uint32_t negative_direction =
        Axis == ReplicateGroup::COLS ? eth_chan_directions::NORTH : eth_chan_directions::WEST;

    // Send multicast in positive direction (start_distance=1, range=positive_range)
    if constexpr (positive_range > 0) {
        tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_atomic_inc(
            &fabric_connections[positive_direction],
            packet_header,
            cmd_header,
            static_cast<uint8_t>(1),
            static_cast<uint8_t>(positive_range));
    }

    // Send multicast in negative direction (start_distance=1, range=negative_range)
    if constexpr (negative_range > 0) {
        tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_atomic_inc(
            &fabric_connections[negative_direction],
            packet_header,
            cmd_header,
            static_cast<uint8_t>(1),
            static_cast<uint8_t>(negative_range));
    }
}

// Bidirectional multicast write - sends same payload to all devices on ring via multicast in both directions
// Handles payloads larger than max packet size by splitting into multiple packets
template <
    uint32_t FabricMaxPacketSzBytes,
    uint32_t LinearizedSrcMeshCoord,
    uint32_t MeshRows,
    uint32_t MeshCols,
    ttnn::operations::ccl::common::ReplicateGroup Axis>
FORCE_INLINE void fabric_multicast_bidirectional_write_ring_1d_async(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header_pos,
    volatile PACKET_HEADER_TYPE* packet_header_neg,
    uint32_t src_addr,
    uint64_t noc_addr,
    int32_t size_bytes,
    uint32_t alignment) {
    using ttnn::operations::ccl::common::ReplicateGroup;

    // ReplicateGroup::COLS (axis=0): targets on same column, dispatch vertically (SOUTH/NORTH),
    // dispatch_devices=MeshRows ReplicateGroup::ROWS (axis=1): targets on same row, dispatch horizontally (EAST/WEST),
    // dispatch_devices=MeshCols
    constexpr uint32_t dispatch_devices = Axis == ReplicateGroup::COLS ? MeshRows : MeshCols;

    // Split the ring: positive direction gets half, negative direction gets the other half
    // For dispatch_devices = 16: positive gets 8, negative gets 7 (total 15 = dispatch_devices - 1)
    constexpr uint32_t positive_range = dispatch_devices / 2;
    constexpr uint32_t negative_range = (dispatch_devices - 1) - positive_range;

    constexpr uint32_t positive_direction =
        Axis == ReplicateGroup::COLS ? eth_chan_directions::SOUTH : eth_chan_directions::EAST;
    constexpr uint32_t negative_direction =
        Axis == ReplicateGroup::COLS ? eth_chan_directions::NORTH : eth_chan_directions::WEST;

    // Track the original total size for the local write at the end
    const uint32_t total_size = static_cast<uint32_t>(size_bytes);
    const uint64_t original_noc_addr = noc_addr;
    const uint32_t original_src_addr = src_addr;

    // Send multicast packets, splitting payload if larger than max packet size
    bool negative_polarity = true;
    while (size_bytes > 0) {
        uint32_t curr_packet_size = std::min(FabricMaxPacketSzBytes, static_cast<uint32_t>(size_bytes));
        // DPRINT << "curr_packet_size: " << curr_packet_size << ENDL();

        const auto noc_command_header = tt::tt_fabric::NocUnicastCommandHeader{noc_addr};

        // Send multicast in positive direction (start_distance=1, range=positive_range)
        // IMPORTANT: Use separate packet headers for each direction to avoid race condition
        // where the second call overwrites the header while first's DMA is still in flight
        if constexpr (positive_range > 0) {
            tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_write(
                &fabric_connections[positive_direction],
                packet_header_pos,
                src_addr,
                curr_packet_size,
                noc_command_header,
                static_cast<uint8_t>(1),
                negative_polarity ? static_cast<uint8_t>(positive_range) : static_cast<uint8_t>(negative_range));
        }

        // Send multicast in negative direction (start_distance=1, range=negative_range)
        if constexpr (negative_range > 0) {
            tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_write(
                &fabric_connections[negative_direction],
                packet_header_neg,
                src_addr,
                curr_packet_size,
                noc_command_header,
                static_cast<uint8_t>(1),
                negative_polarity ? static_cast<uint8_t>(negative_range) : static_cast<uint8_t>(positive_range));
        }

        negative_polarity = !negative_polarity;
        // Update addresses and remaining size for next iteration
        src_addr += curr_packet_size;
        noc_addr += curr_packet_size;
        size_bytes -= curr_packet_size;

        // Wait for header DMAs to complete before modifying headers in next iteration
        // The fabric API uses non-blocking header sends, so we need to ensure the
        // header memory is no longer being read before we overwrite it
        noc_async_writes_flushed();
    }

    // Also write to local device (use original addresses and total size)
    noc_async_write(original_src_addr, original_noc_addr, total_size);
    noc_async_writes_flushed();
}

// Fabric multicast metadata write helper - handles scatter writes in both directions along a 1D ring
// Similar to fabric_multicast_bidirectional_atomic_inc_ring_1d but for scatter writes
template <
    uint32_t LinearizedSrcMeshCoord,
    uint32_t MeshRows,
    uint32_t MeshCols,
    ttnn::operations::ccl::common::ReplicateGroup Axis>
FORCE_INLINE void fabric_multicast_bidirectional_scatter_write_ring_1d_async(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    const std::array<uint64_t, 2>& noc_addresses,
    const std::array<uint16_t, 2>& chunk_sizes) {
    using ttnn::operations::ccl::common::ReplicateGroup;

    // ReplicateGroup::COLS (axis=0): targets on same column, dispatch vertically (SOUTH/NORTH),
    // dispatch_devices=MeshRows ReplicateGroup::ROWS (axis=1): targets on same row, dispatch horizontally (EAST/WEST),
    // dispatch_devices=MeshCols
    constexpr uint32_t dispatch_devices = Axis == ReplicateGroup::COLS ? MeshRows : MeshCols;

    // Split the ring: positive direction gets half, negative direction gets the other half
    // For dispatch_devices = 16: positive gets 8, negative gets 7 (total 15 = dispatch_devices - 1)
    constexpr uint32_t positive_range = dispatch_devices / 2;
    constexpr uint32_t negative_range = (dispatch_devices - 1) - positive_range;

    constexpr uint32_t positive_direction =
        Axis == ReplicateGroup::COLS ? eth_chan_directions::SOUTH : eth_chan_directions::EAST;
    constexpr uint32_t negative_direction =
        Axis == ReplicateGroup::COLS ? eth_chan_directions::NORTH : eth_chan_directions::WEST;

    // Use pointer-based constructor: (addresses*, chunk_sizes*, num_addresses)
    const auto scatter_cmd_header =
        tt::tt_fabric::NocUnicastScatterCommandHeader{noc_addresses.data(), chunk_sizes.data(), 2};
    const uint32_t total_payload_size = chunk_sizes[0] + chunk_sizes[1];

    // Send multicast scatter write in positive direction
    if constexpr (positive_range > 0) {
        tt::tt_fabric::linear::experimental::fabric_multicast_noc_scatter_write(
            &fabric_connections[positive_direction],
            packet_header,
            src_addr,
            total_payload_size,
            scatter_cmd_header,
            static_cast<uint8_t>(1),
            static_cast<uint8_t>(positive_range));
    }

    // Send multicast scatter write in negative direction
    if constexpr (negative_range > 0) {
        tt::tt_fabric::linear::experimental::fabric_multicast_noc_scatter_write(
            &fabric_connections[negative_direction],
            packet_header,
            src_addr,
            total_payload_size,
            scatter_cmd_header,
            static_cast<uint8_t>(1),
            static_cast<uint8_t>(negative_range));
    }

    // Also write to local device - scatter means contiguous src, separate destinations
    // First chunk: src_addr -> noc_addresses[0], chunk_sizes[0] bytes
    // Second chunk: src_addr + chunk_sizes[0] -> noc_addresses[1], chunk_sizes[1] bytes
    noc_async_write(src_addr, noc_addresses[0], chunk_sizes[0]);
    noc_async_write(src_addr + chunk_sizes[0], noc_addresses[1], chunk_sizes[1]);
    noc_async_writes_flushed();
}

// ============================================================================
// Point-to-Point Unicast Dispatch Algorithm
// ============================================================================
// Dispatches tokens to target devices based on expert selection using point-to-point
// unicast sends. For each token, iterates through selected experts and sends to the
// device that owns each expert (with deduplication to avoid sending twice to same device).
//
// Pros: Only sends to devices that need the token (sparse dispatch)
// Cons: In ring/torus topology, may send same token multiple times over same links
//       (e.g., token from device 0 to device 2 doesn't pass through device 1)
//
// Template parameters:
//   LinearizedSrcMeshCoord - Source device's linearized mesh coordinate
//   Topology - Network topology (RING, etc.)
//   MeshRows, MeshCols - Mesh dimensions
//   Axis - Dispatch axis (ROWS or COLS)
//   FabricMaxPacketSize - Maximum fabric packet size in bytes
//   NumDevices - Total number of devices
//   SelectedExpertsK - Number of experts selected per token
//   OutputAddrGenT - Type of the output address generator (TensorAccessor)
// ============================================================================
template <
    uint32_t LinearizedSrcMeshCoord,
    tt::tt_fabric::Topology Topology,
    uint32_t MeshRows,
    uint32_t MeshCols,
    ttnn::operations::ccl::common::ReplicateGroup Axis,
    uint32_t FabricMaxPacketSize,
    uint32_t NumDevices,
    uint32_t SelectedExpertsK,
    typename OutputAddrGenT>
FORCE_INLINE bool dispatch_token_point_to_point_unicast(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* unicast_packet_header,
    const OutputAddrGenT& output_addr_gen,
    const uint16_t* expert_mapping,
    uint8_t* send_preparation_buffer,
    const uint16_t* token_indices,
    uint32_t input_token_read_addr,
    uint64_t output_token_write_addr,
    uint32_t output_page_size,
    uint32_t global_token,
    uint32_t local_token,
    uint32_t token_start_idx,
    uint32_t alignment) {
    using ttnn::operations::ccl::common::fabric_send_chip_unicast_noc_unicast_1d;
    using ttnn::operations::ccl::common::is_configured_target;

    bool needs_barrier = false;

    for (uint32_t k = 0; k < SelectedExpertsK; k++) {
        // Get the expert that is chosen for the current token
        uint16_t expert_chosen = token_indices[k];
        // Direct lookup: get target device from expert mapping
        uint16_t target_device = expert_mapping[expert_chosen];

        // Check if we've already sent to this device for this token (avoid duplicate sends)
        if (send_preparation_buffer[(local_token - token_start_idx) * NumDevices + target_device] == 0) {
            send_preparation_buffer[(local_token - token_start_idx) * NumDevices + target_device] = 1;

            if (target_device == LinearizedSrcMeshCoord) {
                // If the expert lives on the current device, we dispatch the input token to it
                dispatch_input_local_device_flushed(input_token_read_addr, output_token_write_addr, output_page_size);
                needs_barrier = true;
            } else if (is_configured_target<LinearizedSrcMeshCoord, MeshRows, MeshCols, Axis>(target_device)) {
                // If the expert lives on a remote device, we dispatch the input token to it
                // If axis is specified then we only send to the devices that are along the axis
                // If axis is not specified then we send to all devices
                fabric_send_chip_unicast_noc_unicast_1d<
                    LinearizedSrcMeshCoord,
                    Topology,
                    MeshRows,
                    MeshCols,
                    FabricMaxPacketSize>(
                    output_addr_gen,
                    fabric_connections,
                    unicast_packet_header,
                    target_device,
                    input_token_read_addr,
                    global_token,
                    (int)output_page_size,
                    alignment);
            }
        }
    }

    return needs_barrier;
}

// ============================================================================
// Bidirectional Multicast Dispatch Algorithm (Broadcast All)
// ============================================================================
// Dispatches all tokens to all devices via bidirectional multicast in a ring topology.
// Sends in both directions to minimize hop count (tokens travel at most half the ring).
//
// Pros: Optimal for ring topology - each packet traverses minimum hops
//       Simple implementation - no per-token routing decisions
// Cons: Every device receives every token, even if no expert on that device needs it
//       Relies on downstream filtering (selective_tilize) to ignore unneeded tokens
//
// Template parameters:
//   FabricMaxPacketSize - Maximum fabric packet size in bytes
//   LinearizedSrcMeshCoord - Source device's linearized mesh coordinate
//   MeshRows, MeshCols - Mesh dimensions
//   Axis - Dispatch axis (ROWS or COLS)
// ============================================================================
template <
    uint32_t FabricMaxPacketSize,
    uint32_t LinearizedSrcMeshCoord,
    uint32_t MeshRows,
    uint32_t MeshCols,
    ttnn::operations::ccl::common::ReplicateGroup Axis>
FORCE_INLINE void dispatch_token_bidirectional_multicast(
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header_pos,
    volatile PACKET_HEADER_TYPE* packet_header_neg,
    uint32_t input_token_read_addr,
    uint64_t output_token_write_addr,
    uint32_t input_page_size,
    uint32_t alignment) {
    fabric_multicast_bidirectional_write_ring_1d_async<
        FabricMaxPacketSize,
        LinearizedSrcMeshCoord,
        MeshRows,
        MeshCols,
        Axis>(
        fabric_connections,
        packet_header_pos,
        packet_header_neg,
        input_token_read_addr,
        output_token_write_addr,
        (int32_t)input_page_size,
        alignment);
}

}  // namespace detail

using namespace ttnn::operations::ccl::common;

void kernel_main() {
    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t indices_tensor_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t mapping_tensor_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t send_preparation_buffer_cb_id = get_compile_time_arg_val(4);

    constexpr uint32_t input_pages = get_compile_time_arg_val(5);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(6);
    constexpr uint32_t mapping_pages = get_compile_time_arg_val(7);
    constexpr uint32_t output_pages = get_compile_time_arg_val(8);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(9);

    constexpr uint32_t input_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t mapping_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(14);

    constexpr uint32_t num_devices = get_compile_time_arg_val(15);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(16);
    constexpr uint32_t batch_size = get_compile_time_arg_val(17);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(18);
    constexpr uint32_t experts = get_compile_time_arg_val(19);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(20);

    constexpr uint32_t num_links = get_compile_time_arg_val(21);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(22);

    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(23);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(24);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(25);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(26);  // ew_dim
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(27);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(28);
    constexpr uint32_t aligned_mapping_page_size = get_compile_time_arg_val(29);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(30);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(31);

    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(32);
    constexpr uint32_t alignment = get_compile_time_arg_val(33);
    constexpr uint32_t metadata_buffer_id = get_compile_time_arg_val(34);
    constexpr uint32_t write_page_by_page = get_compile_time_arg_val(35);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(36);

    // scores tensor compile time args
    constexpr uint32_t scores_tensor_cb_id = get_compile_time_arg_val(37);
    constexpr uint32_t scores_pages = get_compile_time_arg_val(38);
    constexpr uint32_t scores_page_size = get_compile_time_arg_val(39);
    constexpr uint32_t aligned_scores_page_size = get_compile_time_arg_val(40);

    constexpr auto input_args = TensorAccessorArgs<41>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto scores_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto scores_out_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();

    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t scores_out_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t global_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_args_idx++);
    // Drain sync tilizer core coordinates for direct metadata output
    uint32_t drain_sync_tilizer_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t drain_sync_tilizer_noc_y = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;

    constexpr uint32_t num_directions = 4;
    constexpr std::array<bool, num_directions> directions = DIRECTIONS;

    std::array<tt::tt_fabric::WorkerToFabricEdmSender, num_directions> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);

    uint32_t send_preparation_buffer_address = get_write_ptr(send_preparation_buffer_cb_id);
    detail::zero_buffer_async(
        send_preparation_buffer_address, (token_end_idx - token_start_idx) * num_devices * sizeof(uint8_t));

    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t col = linearized_mesh_coord % mesh_cols;

    constexpr uint32_t dispatch_index = axis == ReplicateGroup::COLS ? row : col;
    // Based on cluster axis, we only need to dispatch to the devices that are along the axis
    // If ReplicateGroup is COLs/AXIS is 1, then we dispatch alonw the ROW, and vice versa
    // For ReplicateGroup COLs/AXIS is 1, the device_begin_idx is the start of the row, and the device_end_idx is the
    // end of the row For ReplicateGroup ROWs/AXIS is 0, the device_begin_idx is the start of the column, and the
    // device_end_idx is the end of the column
    constexpr uint32_t device_begin_idx = axis == ReplicateGroup::COLS ? col : row * mesh_cols;
    constexpr uint32_t device_end_idx =
        (axis == ReplicateGroup::COLS)
            ? (col + mesh_rows * mesh_cols)   // last is col+(mesh_rows-1)*mesh_cols; add one stride
            : (row * mesh_cols + mesh_cols);  // last is row*mesh_cols+(mesh_cols-1); add one
    constexpr uint32_t device_stride = axis == ReplicateGroup::COLS ? mesh_cols : 1;

    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address, output_page_size);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, metadata_page_size);
    const auto output_scores_addr_gen = TensorAccessor(scores_out_args, scores_out_tensor_address, scores_page_size);

    uint32_t packet_header_buffer_address = get_read_ptr(packet_header_cb_id);
    auto* unicast_packet_header_pos = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* unicast_packet_header_neg =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));
    auto* metadata_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + 2 * sizeof(PACKET_HEADER_TYPE));
    auto* scores_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + 3 * sizeof(PACKET_HEADER_TYPE));

    uint32_t base_indices_addr = get_read_ptr(indices_tensor_cb_id);
    uint32_t base_scores_addr = get_read_ptr(scores_tensor_cb_id);

    detail::zero_buffer_barrier();
    open_direction_connections_barrier(directions, fabric_connections);

    // Send initialization semaphore to configured targets for synchronization
    // Use bidirectional multicast for 1D ring topology (2 packets instead of dispatch_devices-1 unicasts)
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
    detail::fabric_multicast_bidirectional_atomic_inc_ring_1d<linearized_mesh_coord, mesh_rows, mesh_cols, axis>(
        fabric_connections, metadata_packet_header, init_noc_semaphore_addr);

    // Wait for all devices to complete initialization synchronization
    bool needs_barrier = false;
    noc_semaphore_wait((uint32_t*)init_semaphore_address, dispatch_devices - 1);
    noc_semaphore_set((uint32_t*)init_semaphore_address, 0);

    // Based on the selected experts, we dispatch the input tokens to the corresponding devices
    cb_wait_front(mapping_tensor_cb_id, 1);
    uint16_t* expert_mapping = (uint16_t*)(get_read_ptr(mapping_tensor_cb_id));
    uint8_t* send_preparation_buffer = (uint8_t*)send_preparation_buffer_address;

    // ============================================================================
    // Token Dispatch Algorithm Selection
    // ============================================================================
    // USE_BIDIRECTIONAL_MULTICAST: Broadcasts all tokens to all devices via bidirectional
    //   multicast. Simple and optimal for ring topology hop count, but sends tokens to
    //   devices that may not need them. Downstream selective_tilize filters unneeded tokens.
    //
    // USE_POINT_TO_POINT_UNICAST: Sends tokens only to devices that need them based on
    //   expert selection. More selective/sparse, but in ring topology may send same token
    //   over same links multiple times (e.g., device 0 -> device 2 doesn't pass device 1).
    //
    // TODO: Implement sparse multicast that collects target devices and sends optimally
    //   through the ring (e.g., device 0 sends to device 1 which forwards to device 2).
    // ============================================================================
    constexpr bool USE_BIDIRECTIONAL_MULTICAST = false;
    constexpr bool USE_POINT_TO_POINT_UNICAST = true;

    for (uint32_t local_token = token_start_idx; local_token < token_end_idx; local_token++) {
        // global_token is the global token index for the current token
        // we need the global token index to write to the output buffer – each global token that could potentially be
        // sent has a unique output buffer address to ensure that it is not overwritten by another token
        uint32_t global_token = (local_token + (tokens_per_device * dispatch_index));
        uint64_t output_token_write_addr = get_noc_addr(global_token, output_addr_gen);
        cb_wait_front(indices_tensor_cb_id, 1);
        cb_wait_front(scores_tensor_cb_id, 1);
        cb_wait_front(input_tensor_cb_id, 1);
        uint32_t input_token_read_addr = get_read_ptr(input_tensor_cb_id);
        uint16_t* token_indices = (uint16_t*)(get_read_ptr(indices_tensor_cb_id));

        if constexpr (USE_BIDIRECTIONAL_MULTICAST) {
            // Broadcast token to all devices via bidirectional multicast
            detail::dispatch_token_bidirectional_multicast<
                fabric_max_packet_size,
                linearized_mesh_coord,
                mesh_rows,
                mesh_cols,
                axis>(
                fabric_connections,
                unicast_packet_header_pos,
                unicast_packet_header_neg,
                input_token_read_addr,
                output_token_write_addr,
                input_page_size,
                alignment);
        } else if constexpr (USE_POINT_TO_POINT_UNICAST) {
            // Send token only to devices that need it based on expert selection
            needs_barrier |= detail::dispatch_token_point_to_point_unicast<
                linearized_mesh_coord,
                topology,
                mesh_rows,
                mesh_cols,
                axis,
                fabric_max_packet_size,
                num_devices,
                selected_experts_k>(
                fabric_connections,
                unicast_packet_header_neg,
                output_addr_gen,
                expert_mapping,
                send_preparation_buffer,
                token_indices,
                input_token_read_addr,
                output_token_write_addr,
                output_page_size,
                global_token,
                local_token,
                token_start_idx,
                alignment);
        }

        cb_pop_front(indices_tensor_cb_id, 1);
        cb_pop_front(scores_tensor_cb_id, 1);
        cb_pop_front(input_tensor_cb_id, 1);
    }
    if (needs_barrier) {
        noc_async_write_barrier();
    }
    // Send our selected experts tensor to all other devices and signal that we are done dispatching the input tokens
    // with a semaphore. Write directly to the output metadata tensor on the drain sync tilizer core.
    uint64_t global_noc_semaphore_address = get_noc_addr(global_semaphore_address);

    // IMPORTANT: Use aligned_metadata_page_size for OUTPUT metadata tensor offsets.
    // The input indices tensor MUST be in L1 (not DRAM) to ensure its alignment matches the output
    // metadata tensor's alignment. DRAM uses 32B alignment while L1 uses 16B alignment.
    // If alignments don't match, the padding bytes in the CB will corrupt the output.
    // The metadata tensor layout is: [device_0_tokens, device_1_tokens, ..., device_N_tokens]
    // Each device section is: tokens_per_device * aligned_metadata_page_size bytes
    uint32_t metadata_size_per_device = aligned_metadata_page_size * tokens_per_device;
    uint32_t metadata_size_per_core = aligned_metadata_page_size * (token_end_idx - token_start_idx);

    // Write directly to the sharded output metadata tensor on the drain sync tilizer core
    // The tensor is contiguous in L1 at metadata_tensor_address on the drain core
    // NOTE: We manually construct the NOC address using drain core coords rather than using
    // metadata_addr_gen because our offset logic (dispatch_index * metadata_size) assumes
    // a specific memory layout that may not match the accessor's page indexing.
    uint64_t metadata_output_base_addr =
        get_noc_addr(drain_sync_tilizer_noc_x, drain_sync_tilizer_noc_y, metadata_tensor_address);
    uint64_t noc_core_offset_md_write_addr = metadata_output_base_addr + dispatch_index * metadata_size_per_device +
                                             token_start_idx * aligned_metadata_page_size;

    uint64_t scores_output_base_addr =
        get_noc_addr(drain_sync_tilizer_noc_x, drain_sync_tilizer_noc_y, scores_out_tensor_address);
    uint64_t noc_core_offset_scores_write_addr = scores_output_base_addr + dispatch_index * metadata_size_per_device +
                                                 token_start_idx * aligned_scores_page_size;

    cb_wait_front(metadata_buffer_id, tokens_per_device);
    uint32_t base_metadata_addr = get_read_ptr(metadata_buffer_id);
    detail::
        fabric_multicast_bidirectional_scatter_write_ring_1d_async<linearized_mesh_coord, mesh_rows, mesh_cols, axis>(
            fabric_connections,
            metadata_packet_header,
            base_metadata_addr,
            {noc_core_offset_md_write_addr, noc_core_offset_scores_write_addr},
            {static_cast<uint16_t>(metadata_size_per_core), static_cast<uint16_t>(metadata_size_per_core)});
    cb_pop_front(metadata_buffer_id, tokens_per_device);
    // Use DoubleAntipodalAtomicInc=true to increment semaphore on all devices including antipodal
    detail::fabric_multicast_bidirectional_atomic_inc_ring_1d<linearized_mesh_coord, mesh_rows, mesh_cols, axis, true>(
        fabric_connections, metadata_packet_header, global_noc_semaphore_address);

    cb_pop_front(mapping_tensor_cb_id, mapping_pages);

    close_direction_connections(directions, fabric_connections);
    noc_async_write_barrier();
}
