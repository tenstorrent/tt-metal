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
    bool DoubleAntipodalAtomicInc = false,
    typename FabricConnectionsType>
FORCE_INLINE void fabric_multicast_bidirectional_atomic_inc_ring_1d(
    FabricConnectionsType& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header_pos,
    volatile PACKET_HEADER_TYPE* packet_header_neg,
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
            packet_header_pos,
            cmd_header,
            static_cast<uint8_t>(1),
            static_cast<uint8_t>(positive_range));
    }

    // Send multicast in negative direction (start_distance=1, range=negative_range)
    if constexpr (negative_range > 0) {
        tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_atomic_inc(
            &fabric_connections[negative_direction],
            packet_header_neg,
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
    ttnn::operations::ccl::common::ReplicateGroup Axis,
    typename FabricConnectionsType>
FORCE_INLINE void fabric_multicast_bidirectional_write_ring_1d_async(
    FabricConnectionsType& fabric_connections,
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
    ttnn::operations::ccl::common::ReplicateGroup Axis,
    typename FabricConnectionsType>
FORCE_INLINE void fabric_multicast_bidirectional_scatter_write_ring_1d_async(
    FabricConnectionsType& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header_pos,
    volatile PACKET_HEADER_TYPE* packet_header_neg,
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
            packet_header_pos,
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
            packet_header_neg,
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
    typename OutputAddrGenT,
    typename FabricConnectionsType>
FORCE_INLINE bool dispatch_token_point_to_point_unicast(
    FabricConnectionsType& fabric_connections,
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
    uint32_t alignment,
    uint32_t payload_offset = 0) {
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
                noc_async_write(input_token_read_addr, output_token_write_addr, output_page_size);
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
                    alignment,
                    payload_offset);
            }
        }
    }

    noc_async_writes_flushed();
    return needs_barrier;
}

// ============================================================================
// Sparse Multicast Dispatch Algorithm
// ============================================================================
// Collects all unique target devices for a token's selected experts, then sends
// the token to all of them via sparse multicast (single packet with multiple destinations).
//
// Pros: Only sends to devices that need the token (based on expert selection)
//       Efficient for cases where tokens select experts on few devices
// Cons: Requires building destination list per token
//       May have higher latency than point-to-point for single destinations
//
// Template parameters:
//   LinearizedSrcMeshCoord - Source device's linearized mesh coordinate
//   Topology - Fabric topology (Ring1D, etc.)
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
    typename OutputAddrGenT,
    typename FabricConnectionsType>
FORCE_INLINE bool dispatch_token_sparse_multicast(
    FabricConnectionsType& fabric_connections,
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
    uint32_t alignment,
    uint32_t payload_offset = 0) {
    using ttnn::operations::ccl::common::fabric_send_chip_sparse_multicast_noc_unicast_1d;
    using ttnn::operations::ccl::common::is_configured_target;

    bool needs_barrier = false;

    // Collect all unique remote destinations for this token
    uint32_t remote_token_destinations[NumDevices];
    uint32_t num_remote_token_destinations = 0;

    for (uint32_t k = 0; k < SelectedExpertsK; k++) {
        uint16_t expert_chosen = token_indices[k];
        // Direct lookup: get target device from expert mapping
        uint16_t target_device = expert_mapping[expert_chosen];

        // Check if we've already processed this device for this token
        if (send_preparation_buffer[(local_token - token_start_idx) * NumDevices + target_device] == 0) {
            send_preparation_buffer[(local_token - token_start_idx) * NumDevices + target_device] = 1;

            if (target_device == LinearizedSrcMeshCoord) {
                // If the expert lives on the current device, dispatch locally
                noc_async_write(input_token_read_addr, output_token_write_addr, output_page_size);
                needs_barrier = true;
            } else if (is_configured_target<LinearizedSrcMeshCoord, MeshRows, MeshCols, Axis>(target_device)) {
                // Add to remote destinations list
                remote_token_destinations[num_remote_token_destinations++] = target_device;
            }
        }
    }

    // If there are any remote destinations, send via sparse multicast
    if (num_remote_token_destinations > 0) {
        fabric_send_chip_sparse_multicast_noc_unicast_1d<
            LinearizedSrcMeshCoord,
            Topology,
            MeshRows,
            MeshCols,
            NumDevices,
            FabricMaxPacketSize>(
            output_addr_gen,
            fabric_connections,
            unicast_packet_header,
            remote_token_destinations,
            num_remote_token_destinations,
            input_token_read_addr,
            global_token,
            (int)output_page_size,
            alignment,
            payload_offset);
    }
    noc_async_writes_flushed();
    return needs_barrier;
}

// ============================================================================
// Bidirectional Sparse Multicast Dispatch Algorithm (Shortest Path)
// ============================================================================
// Builds hop masks for both directions using bit manipulation, then sends
// sparse multicast packets in both directions based on shortest path routing.
// For antipodal ties (equal distance both ways), direction alternates based on token index.
//
// Uses OR-based deduplication: if the same device is selected multiple times,
// the bit is already set so OR-ing again is a no-op.
//
// Pros: Only sends to devices that need the token
//       Uses shortest path for each destination (optimal hop count)
//       Balances antipodal traffic across both directions
//       Fast bit manipulation without send_preparation_buffer
// Cons: May send two packets per token (one in each direction)
//       Requires two packet headers
//
// Template parameters:
//   LinearizedSrcMeshCoord - Source device's linearized mesh coordinate
//   Topology - Fabric topology (Ring1D, etc.)
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
    typename OutputAddrGenT,
    typename FabricConnectionsType>
FORCE_INLINE bool dispatch_token_sparse_multicast_bidirectional(
    FabricConnectionsType& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header_pos,
    volatile PACKET_HEADER_TYPE* packet_header_neg,
    const OutputAddrGenT& output_addr_gen,
    const uint16_t* expert_mapping,
    const uint16_t* token_indices,
    uint32_t input_token_read_addr,
    uint64_t output_token_write_addr,
    uint32_t output_page_size,
    uint32_t global_token,
    ttnn::operations::ccl::common::Polarity antipodal_polarity,  // Direction for antipodal tie-breaking
    uint32_t alignment,
    uint32_t payload_offset = 0) {
    using ttnn::operations::ccl::common::fabric_send_chip_sparse_multicast_noc_unicast_1d_in_direction;
    using ttnn::operations::ccl::common::is_configured_target;
    using ttnn::operations::ccl::common::Polarity;

    bool needs_barrier = false;

    // Build hop masks for both directions directly via bit manipulation
    // OR handles deduplication: same bit set twice is still just set once
    uint16_t pos_hop_mask = 0;  // EAST (positive direction)
    uint16_t neg_hop_mask = 0;  // WEST (negative direction)
    bool sent_local = false;

    for (uint32_t k = 0; k < SelectedExpertsK; k++) {
        uint16_t expert_chosen = token_indices[k];
        uint16_t target_device = expert_mapping[expert_chosen];

        if (target_device == LinearizedSrcMeshCoord) {
            // Local device - dispatch once
            if (!sent_local) {
                noc_async_write(input_token_read_addr, output_token_write_addr, output_page_size);
                needs_barrier = true;
                sent_local = true;
            }
        } else if (is_configured_target<LinearizedSrcMeshCoord, MeshRows, MeshCols, Axis>(target_device)) {
            // Remote device on our axis - calculate distance in both directions
            // pos_distance: going EAST/SOUTH (ascending, with wrap)
            // neg_distance: going WEST/NORTH (descending, with wrap)
            uint32_t pos_distance = (target_device - LinearizedSrcMeshCoord + NumDevices) % NumDevices;
            uint32_t neg_distance = (LinearizedSrcMeshCoord - target_device + NumDevices) % NumDevices;
            // Determine shortest path direction
            if (pos_distance < neg_distance) {
                // Shorter via positive direction (EAST/SOUTH)
                pos_hop_mask |= (1 << (pos_distance - 1));
            } else if (neg_distance < pos_distance) {
                // Shorter via negative direction (WEST/NORTH)
                neg_hop_mask |= (1 << (neg_distance - 1));
            } else {
                // Antipodal tie - use provided polarity
                if (antipodal_polarity == Polarity::POSITIVE) {
                    pos_hop_mask |= (1 << (pos_distance - 1));
                } else {
                    neg_hop_mask |= (1 << (neg_distance - 1));
                }
            }
        }
        // else: target_device is on a different axis - skip (handled by another dispatch on that axis)
    }

    // Derive direction constants from Axis template parameter
    // ROWS axis (or 1xN mesh): EAST/WEST, COLS axis (or Nx1 mesh): SOUTH/NORTH
    /*
        constexpr uint32_t positive_direction =
        Axis == ReplicateGroup::COLS ? eth_chan_directions::SOUTH : eth_chan_directions::EAST;
    constexpr uint32_t negative_direction =
        Axis == ReplicateGroup::COLS ? eth_chan_directions::NORTH : eth_chan_directions::WEST;*/
    constexpr bool is_row_axis = (Axis == ttnn::operations::ccl::common::ReplicateGroup::ROWS) ||
                                 (Axis == ttnn::operations::ccl::common::ReplicateGroup::NONE && MeshRows == 1);
    constexpr uint32_t pos_direction = is_row_axis ? eth_chan_directions::EAST : eth_chan_directions::SOUTH;
    constexpr uint32_t neg_direction = is_row_axis ? eth_chan_directions::WEST : eth_chan_directions::NORTH;

    // Send in positive direction if any destinations
    if (pos_hop_mask != 0) {
        fabric_send_chip_sparse_multicast_noc_unicast_1d_in_direction<FabricMaxPacketSize>(
            output_addr_gen,
            fabric_connections,
            packet_header_pos,
            pos_hop_mask,
            pos_direction,
            input_token_read_addr,
            global_token,
            (int)output_page_size,
            alignment,
            payload_offset);
    }

    // Send in negative direction if any destinations
    if (neg_hop_mask != 0) {
        fabric_send_chip_sparse_multicast_noc_unicast_1d_in_direction<FabricMaxPacketSize>(
            output_addr_gen,
            fabric_connections,
            packet_header_neg,
            neg_hop_mask,
            neg_direction,
            input_token_read_addr,
            global_token,
            (int)output_page_size,
            alignment,
            payload_offset);
    }

    noc_async_writes_flushed();
    return needs_barrier;
}

// ============================================================================
// Split-Bandwidth Sparse Multicast Dispatch Algorithm
// ============================================================================
// Splits the token DATA in half and sends each half in opposite directions to
// only the devices that need the token. This balances bandwidth usage across
// both ring directions while still being selective about destinations.
//
// For token 0 (POSITIVE polarity): first half → positive dir, second half → negative dir
// For token 1 (NEGATIVE polarity): first half → negative dir, second half → positive dir
// This alternates to avoid systematic bias.
//
// Both halves go to the SAME set of destinations - just via different directions.
// Each destination receives both halves of the token.
//
// Pros: Guaranteed 50/50 bandwidth split for every token
//       Only sends to devices that need the token (selective)
// Cons: Each destination receives data via both directions
//
// Template parameters:
//   LinearizedSrcMeshCoord - Source device's linearized mesh coordinate
//   MeshRows, MeshCols - Mesh dimensions
//   Axis - Dispatch axis (ROWS or COLS)
//   FabricMaxPacketSize - Maximum fabric packet size in bytes
//   NumDevices - Total number of devices
//   SelectedExpertsK - Number of experts selected per token
//   OutputAddrGenT - Type of the output address generator (TensorAccessor)
// ============================================================================
template <
    uint32_t LinearizedSrcMeshCoord,
    uint32_t MeshRows,
    uint32_t MeshCols,
    ttnn::operations::ccl::common::ReplicateGroup Axis,
    uint32_t FabricMaxPacketSize,
    uint32_t NumDevices,
    uint32_t SelectedExpertsK,
    typename OutputAddrGenT,
    typename FabricConnectionsType>
FORCE_INLINE bool dispatch_token_split_bandwidth(
    FabricConnectionsType& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header_pos,
    volatile PACKET_HEADER_TYPE* packet_header_neg,
    const OutputAddrGenT& output_addr_gen,
    const uint16_t* expert_mapping,
    const uint16_t* token_indices,
    uint32_t input_token_read_addr,
    uint64_t output_token_write_addr,
    uint32_t input_page_size,
    uint32_t global_token,
    ttnn::operations::ccl::common::Polarity first_half_polarity,
    uint32_t alignment) {
    using ttnn::operations::ccl::common::fabric_send_chip_sparse_multicast_noc_unicast_1d_in_direction;
    using ttnn::operations::ccl::common::is_configured_target;
    using ttnn::operations::ccl::common::Polarity;

    bool needs_barrier = false;

    // Derive direction constants from Axis template parameter
    constexpr bool is_row_axis = (Axis == ttnn::operations::ccl::common::ReplicateGroup::ROWS) ||
                                 (Axis == ttnn::operations::ccl::common::ReplicateGroup::NONE && MeshRows == 1);
    constexpr uint32_t pos_direction = is_row_axis ? eth_chan_directions::EAST : eth_chan_directions::SOUTH;
    constexpr uint32_t neg_direction = is_row_axis ? eth_chan_directions::WEST : eth_chan_directions::NORTH;

    // Build hop masks for both directions - same destinations, different routes
    // pos_hop_mask: bits set for hops needed to reach each destination going positive
    // neg_hop_mask: bits set for hops needed to reach each destination going negative
    // OR handles deduplication: setting a bit that's already 1 to 1 is a no-op
    uint16_t pos_hop_mask = 0;
    uint16_t neg_hop_mask = 0;
    bool sent_local = false;

    for (uint32_t k = 0; k < SelectedExpertsK; k++) {
        uint16_t expert_chosen = token_indices[k];
        uint16_t target_device = expert_mapping[expert_chosen];

        if (target_device == LinearizedSrcMeshCoord) {
            // Local device - dispatch once
            if (!sent_local) {
                noc_async_write(input_token_read_addr, output_token_write_addr, input_page_size);
                needs_barrier = true;
                sent_local = true;
            }
        } else if (is_configured_target<LinearizedSrcMeshCoord, MeshRows, MeshCols, Axis>(target_device)) {
            // Remote device on our axis - add to BOTH masks (same dest, different directions)
            // OR handles dedup: if bit already set, setting again is harmless

            // Calculate distance in positive direction (e.g., EAST/SOUTH)
            uint32_t pos_distance = (target_device - LinearizedSrcMeshCoord + NumDevices) % NumDevices;
            // Calculate distance in negative direction (e.g., WEST/NORTH)
            uint32_t neg_distance = (LinearizedSrcMeshCoord - target_device + NumDevices) % NumDevices;

            // Add to both masks - each destination reachable from both directions
            pos_hop_mask |= (1 << (pos_distance - 1));
            neg_hop_mask |= (1 << (neg_distance - 1));
        }
        // else: target_device is on a different axis - skip (handled by another dispatch)
    }

    // If no remote destinations, we're done
    if (pos_hop_mask == 0 && neg_hop_mask == 0) {
        return needs_barrier;
    }

    // Split token in half
    uint32_t half_size = input_page_size / 2;
    uint32_t first_half_addr = input_token_read_addr;
    uint32_t second_half_addr = input_token_read_addr + half_size;

    // Determine which half goes which direction based on polarity
    uint32_t pos_addr, neg_addr;
    uint32_t pos_offset, neg_offset;
    if (first_half_polarity == Polarity::POSITIVE) {
        pos_addr = first_half_addr;   // First half goes positive
        neg_addr = second_half_addr;  // Second half goes negative
        pos_offset = 0;
        neg_offset = half_size;
    } else {
        pos_addr = second_half_addr;  // Second half goes positive
        neg_addr = first_half_addr;   // First half goes negative
        pos_offset = half_size;
        neg_offset = 0;
    }

    // Send one half in positive direction to all destinations
    fabric_send_chip_sparse_multicast_noc_unicast_1d_in_direction<FabricMaxPacketSize>(
        output_addr_gen,
        fabric_connections,
        packet_header_pos,
        pos_hop_mask,
        pos_direction,
        pos_addr,
        global_token,
        (int)half_size,
        alignment,
        pos_offset);

    // Send other half in negative direction to all destinations
    fabric_send_chip_sparse_multicast_noc_unicast_1d_in_direction<FabricMaxPacketSize>(
        output_addr_gen,
        fabric_connections,
        packet_header_neg,
        neg_hop_mask,
        neg_direction,
        neg_addr,
        global_token,
        (int)half_size,
        alignment,
        neg_offset);
    noc_async_writes_flushed();
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
    ttnn::operations::ccl::common::ReplicateGroup Axis,
    typename FabricConnectionsType>
FORCE_INLINE void dispatch_token_bidirectional_multicast(
    FabricConnectionsType& fabric_connections,
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

#ifdef USE_MUX
    // Mux compile-time args (appended after TensorAccessorArgs when USE_MUX is defined)
    constexpr uint32_t mux_ct_args_offset = scores_out_args.next_compile_time_args_offset();
    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(mux_ct_args_offset + 0);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(mux_ct_args_offset + 1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(mux_ct_args_offset + 2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(mux_ct_args_offset + 3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(mux_ct_args_offset + 4);
#endif

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

    // Payload split parameters (always read from RT args)
    // In payload split mode: each worker sends a portion of the payload
    // In non-split mode: defaults are payload_offset=0, payload_size=input_page_size, is_primary=true
    uint32_t payload_offset = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t payload_size = get_arg_val<uint32_t>(rt_args_idx++);
    bool is_primary_payload_worker = get_arg_val<uint32_t>(rt_args_idx++) == 1;

    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;

    constexpr uint32_t num_directions = 4;
    constexpr std::array<bool, num_directions> directions = DIRECTIONS;

#ifdef USE_MUX
    // ========================================================================
    // MUX PATH: Use WorkerToFabricMuxSender connections via fabric mux
    // ========================================================================
    // Read mux runtime args for each direction (neighbors)
    std::array<bool, num_directions> mux_connection_valid_arr = {false, false, false, false};
    std::array<bool, num_directions> is_termination_master_arr = {false, false, false, false};
    std::array<uint8_t, num_directions> fabric_mux_x_arr = {0, 0, 0, 0};
    std::array<uint8_t, num_directions> fabric_mux_y_arr = {0, 0, 0, 0};
    std::array<size_t, num_directions> fabric_mux_channel_base_address_arr = {0, 0, 0, 0};
    std::array<size_t, num_directions> fabric_mux_connection_info_address_arr = {0, 0, 0, 0};
    std::array<size_t, num_directions> fabric_mux_connection_handshake_address_arr = {0, 0, 0, 0};
    std::array<size_t, num_directions> fabric_mux_flow_control_address_arr = {0, 0, 0, 0};
    std::array<size_t, num_directions> fabric_mux_buffer_index_address_arr = {0, 0, 0, 0};
    std::array<uint8_t, num_directions> fabric_mux_channel_id_arr = {0, 0, 0, 0};
    std::array<uint32_t, num_directions> termination_sync_address_arr = {0, 0, 0, 0};
    std::array<uint32_t, num_directions> local_fabric_mux_status_address_arr = {0, 0, 0, 0};
    std::array<uint32_t, num_directions> local_flow_control_address_arr = {0, 0, 0, 0};
    std::array<uint32_t, num_directions> local_teardown_address_arr = {0, 0, 0, 0};
    std::array<uint32_t, num_directions> local_buffer_index_address_arr = {0, 0, 0, 0};
    std::array<uint32_t, num_directions> termination_master_noc_x_arr = {0, 0, 0, 0};
    std::array<uint32_t, num_directions> termination_master_noc_y_arr = {0, 0, 0, 0};

    bool any_is_termination_master = false;
    for (uint32_t dir = 0; dir < num_directions; dir++) {
        if (directions[dir]) {
            mux_connection_valid_arr[dir] = get_arg_val<uint32_t>(rt_args_idx++) == 1;
            is_termination_master_arr[dir] = get_arg_val<uint32_t>(rt_args_idx++) == 1;
            fabric_mux_x_arr[dir] = get_arg_val<uint32_t>(rt_args_idx++);
            fabric_mux_y_arr[dir] = get_arg_val<uint32_t>(rt_args_idx++);
            fabric_mux_channel_base_address_arr[dir] = get_arg_val<uint32_t>(rt_args_idx++);
            fabric_mux_connection_info_address_arr[dir] = get_arg_val<uint32_t>(rt_args_idx++);
            fabric_mux_connection_handshake_address_arr[dir] = get_arg_val<uint32_t>(rt_args_idx++);
            fabric_mux_flow_control_address_arr[dir] = get_arg_val<uint32_t>(rt_args_idx++);
            fabric_mux_buffer_index_address_arr[dir] = get_arg_val<uint32_t>(rt_args_idx++);
            fabric_mux_channel_id_arr[dir] = get_arg_val<uint32_t>(rt_args_idx++);
            termination_sync_address_arr[dir] = get_semaphore(get_arg_val<uint32_t>(rt_args_idx++));
            local_fabric_mux_status_address_arr[dir] = get_semaphore(get_arg_val<uint32_t>(rt_args_idx++));
            local_flow_control_address_arr[dir] = get_semaphore(get_arg_val<uint32_t>(rt_args_idx++));
            local_teardown_address_arr[dir] = get_semaphore(get_arg_val<uint32_t>(rt_args_idx++));
            local_buffer_index_address_arr[dir] = get_semaphore(get_arg_val<uint32_t>(rt_args_idx++));
            termination_master_noc_x_arr[dir] = get_arg_val<uint32_t>(rt_args_idx++);
            termination_master_noc_y_arr[dir] = get_arg_val<uint32_t>(rt_args_idx++);
            if (is_termination_master_arr[dir]) {
                any_is_termination_master = true;
            }
        }
    }

    // Build mux connections for each direction
    std::array<tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>, num_directions>
        fabric_connections;
    for (uint32_t dir = 0; dir < num_directions; dir++) {
        if (directions[dir] && mux_connection_valid_arr[dir]) {
            fabric_connections[dir] =
                tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
                    fabric_mux_x_arr[dir],
                    fabric_mux_y_arr[dir],
                    fabric_mux_channel_id_arr[dir],
                    fabric_mux_num_buffers_per_channel,
                    fabric_mux_channel_buffer_size_bytes,
                    fabric_mux_channel_base_address_arr[dir],
                    fabric_mux_connection_info_address_arr[dir],
                    fabric_mux_connection_handshake_address_arr[dir],
                    fabric_mux_flow_control_address_arr[dir],
                    fabric_mux_buffer_index_address_arr[dir],
                    local_flow_control_address_arr[dir],
                    local_teardown_address_arr[dir],
                    local_buffer_index_address_arr[dir]);
        }
    }
#else
    // ========================================================================
    // DIRECT EDM PATH: Use WorkerToFabricEdmSender connections directly
    // ========================================================================
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, num_directions> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);
#endif

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
    auto* metadata_packet_header_pos =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + 2 * sizeof(PACKET_HEADER_TYPE));
    auto* metadata_packet_header_neg =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + 3 * sizeof(PACKET_HEADER_TYPE));
    auto* atomic_inc_packet_header_pos =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + 4 * sizeof(PACKET_HEADER_TYPE));
    auto* atomic_inc_packet_header_neg =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + 5 * sizeof(PACKET_HEADER_TYPE));
    // packet headers at +6 and +7 are currently unused (reserved for future split sends)

    uint32_t base_indices_addr = get_read_ptr(indices_tensor_cb_id);
    uint32_t base_scores_addr = get_read_ptr(scores_tensor_cb_id);

    detail::zero_buffer_barrier();

#ifdef USE_MUX
    // Wait for mux to be ready and connect
    for (uint32_t dir = 0; dir < num_directions; dir++) {
        if (directions[dir] && mux_connection_valid_arr[dir]) {
            tt::tt_fabric::wait_for_fabric_endpoint_ready(
                fabric_mux_x_arr[dir],
                fabric_mux_y_arr[dir],
                fabric_mux_status_address,
                local_fabric_mux_status_address_arr[dir]);
            tt::tt_fabric::fabric_client_connect(fabric_connections[dir]);
        }
    }
#else
    open_direction_connections_barrier(directions, fabric_connections);
#endif

    // Send initialization semaphore to configured targets for synchronization
    // Use bidirectional multicast for 1D ring topology (2 packets instead of dispatch_devices-1 unicasts)
    // In persistent mode (SKIP_INIT_SEMAPHORE defined), we skip this barrier because:
    // - All output buffers are persistent, so remote writes won't corrupt data being used
    // - The cross_device_semaphore is double-buffered externally to avoid races between iterations
#ifndef SKIP_INIT_SEMAPHORE
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
    detail::fabric_multicast_bidirectional_atomic_inc_ring_1d<linearized_mesh_coord, mesh_rows, mesh_cols, axis>(
        fabric_connections, atomic_inc_packet_header_pos, atomic_inc_packet_header_neg, init_noc_semaphore_addr);
    noc_async_writes_flushed();

    // Wait for all devices to complete initialization synchronization
    noc_semaphore_wait((uint32_t*)init_semaphore_address, dispatch_devices - 1);
    noc_semaphore_set((uint32_t*)init_semaphore_address, 0);
#endif
    bool needs_barrier = false;
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
    // ============================================================================
    enum class DispatchAlgorithm : uint8_t {
        BROADCAST = 0,                   // Broadcast all tokens to ALL devices (bidirectional multicast)
        SPARSE_UNICAST = 1,              // Send to each target device individually (point-to-point)
        SPARSE_MCAST_LINEAR = 2,         // Sparse multicast in single direction
        SPARSE_MCAST_SHORTEST_PATH = 3,  // Sparse multicast with bidirectional shortest path routing
        SPARSE_MCAST_SPLIT_BW = 4        // Sparse multicast, split token data 50/50 between directions
    };
    // DISPATCH_ALGORITHM is passed as a define from the host
    constexpr DispatchAlgorithm dispatch_algorithm = static_cast<DispatchAlgorithm>(DISPATCH_ALGORITHM);

    for (uint32_t local_token = token_start_idx; local_token < token_end_idx; local_token++) {
        // global_token is the global token index for the current token
        // we need the global token index to write to the output buffer – each global token that could potentially be
        // sent has a unique output buffer address to ensure that it is not overwritten by another token
        uint32_t global_token = (local_token + (tokens_per_device * dispatch_index));
        uint64_t output_token_write_addr = get_noc_addr(global_token, output_addr_gen);
        // All workers read indices (needed for routing decisions)
        // Only primary worker reads scores (only primary sends metadata)
        cb_wait_front(indices_tensor_cb_id, 1);
        if (is_primary_payload_worker) {
            cb_wait_front(scores_tensor_cb_id, 1);
        }
        cb_wait_front(input_tensor_cb_id, 1);
        uint32_t input_token_read_addr = get_read_ptr(input_tensor_cb_id);
        uint16_t* token_indices = (uint16_t*)(get_read_ptr(indices_tensor_cb_id));

        // In payload split mode: reader already reads only this worker's portion into CB
        // In non-split mode: reader reads full page, payload_offset=0
        // Either way, we read from the start of the CB
        uint32_t payload_read_addr = input_token_read_addr;
        // Write address includes offset to write to correct position in output
        uint64_t payload_write_addr = output_token_write_addr + payload_offset;

        if constexpr (dispatch_algorithm == DispatchAlgorithm::BROADCAST) {
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
                payload_read_addr,
                payload_write_addr,
                payload_size,
                alignment);
        } else if constexpr (dispatch_algorithm == DispatchAlgorithm::SPARSE_UNICAST) {
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
                payload_read_addr,
                payload_write_addr,
                payload_size,
                global_token,
                local_token,
                token_start_idx,
                alignment,
                payload_offset);
        } else if constexpr (dispatch_algorithm == DispatchAlgorithm::SPARSE_MCAST_LINEAR) {
            // Collect unique destinations and send via sparse multicast (single direction)
            needs_barrier |= detail::dispatch_token_sparse_multicast<
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
                payload_read_addr,
                payload_write_addr,
                payload_size,
                global_token,
                local_token,
                token_start_idx,
                alignment,
                payload_offset);
        } else if constexpr (dispatch_algorithm == DispatchAlgorithm::SPARSE_MCAST_SHORTEST_PATH) {
            // Collect unique destinations and send via bidirectional sparse multicast
            // Uses shortest path routing with antipodal tie-breaking
            needs_barrier |= detail::dispatch_token_sparse_multicast_bidirectional<
                linearized_mesh_coord,
                topology,
                mesh_rows,
                mesh_cols,
                axis,
                fabric_max_packet_size,
                num_devices,
                selected_experts_k>(
                fabric_connections,
                unicast_packet_header_pos,
                unicast_packet_header_neg,
                output_addr_gen,
                expert_mapping,
                token_indices,
                payload_read_addr,
                payload_write_addr,
                payload_size,
                global_token,
                static_cast<ttnn::operations::ccl::common::Polarity>(local_token % 2),
                alignment,
                payload_offset);
        } else if constexpr (dispatch_algorithm == DispatchAlgorithm::SPARSE_MCAST_SPLIT_BW) {
            // Split token data in half: first half one direction, second half other direction
            // Both halves go to same destinations, just via different directions
            // Token 0: first half → positive dir, second half → negative dir
            // Token 1: first half → negative dir, second half → positive dir
            // NOTE: SPARSE_MCAST_SPLIT_BW already splits the payload by direction, so it's
            // incompatible with PAYLOAD_SPLIT_MODE which splits by worker. Use only one mode.
            needs_barrier |= detail::dispatch_token_split_bandwidth<
                linearized_mesh_coord,
                mesh_rows,
                mesh_cols,
                axis,
                fabric_max_packet_size,
                num_devices,
                selected_experts_k>(
                fabric_connections,
                unicast_packet_header_pos,
                unicast_packet_header_neg,
                output_addr_gen,
                expert_mapping,
                token_indices,
                payload_read_addr,
                payload_write_addr,
                payload_size,
                global_token,
                static_cast<ttnn::operations::ccl::common::Polarity>(local_token % 2),
                alignment);
        }

        // All workers pop indices (all read them for routing)
        // Only primary pops scores (only primary reads them)
        cb_pop_front(indices_tensor_cb_id, 1);
        if (is_primary_payload_worker) {
            cb_pop_front(scores_tensor_cb_id, 1);
        }
        cb_pop_front(input_tensor_cb_id, 1);
    }
    if (needs_barrier) {
        noc_async_write_barrier();
    }

#ifdef PAYLOAD_SPLIT_MODE
    // In payload split mode, all workers must barrier before the primary sends atomic_inc
    // This ensures all payload portions have been sent before signaling completion
    noc_async_write_barrier();
#endif

    // Only the primary worker (or all workers in non-payload-split mode) sends metadata and atomic_inc
    // In payload split mode, non-primary workers skip metadata/atomic_inc entirely
    if (is_primary_payload_worker) {
        // Send our selected experts tensor to all other devices and signal that we are done dispatching the input
        // tokens with a semaphore. Write directly to the output metadata tensor on the drain sync tilizer core.
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
        uint64_t noc_core_offset_scores_write_addr = scores_output_base_addr +
                                                     dispatch_index * metadata_size_per_device +
                                                     token_start_idx * aligned_scores_page_size;

        cb_wait_front(metadata_buffer_id, tokens_per_device);
        uint32_t base_metadata_addr = get_read_ptr(metadata_buffer_id);
        detail::fabric_multicast_bidirectional_scatter_write_ring_1d_async<
            linearized_mesh_coord,
            mesh_rows,
            mesh_cols,
            axis>(
            fabric_connections,
            metadata_packet_header_pos,
            metadata_packet_header_neg,
            base_metadata_addr,
            {noc_core_offset_md_write_addr, noc_core_offset_scores_write_addr},
            {static_cast<uint16_t>(metadata_size_per_core), static_cast<uint16_t>(metadata_size_per_core)});
        cb_pop_front(metadata_buffer_id, tokens_per_device);
        // Use DoubleAntipodalAtomicInc=true to increment semaphore on all devices including twice on the antipodal
        // device
        detail::
            fabric_multicast_bidirectional_atomic_inc_ring_1d<linearized_mesh_coord, mesh_rows, mesh_cols, axis, true>(
                fabric_connections,
                atomic_inc_packet_header_pos,
                atomic_inc_packet_header_neg,
                global_noc_semaphore_address);
    }

    cb_pop_front(mapping_tensor_cb_id, mapping_pages);

#ifdef USE_MUX
    // MUX teardown for Phase 2: Multiple workers per link share the same mux cores
    // Each link has a termination master that waits for all other workers to disconnect
    // before terminating the mux cores.
    noc_async_write_barrier();
    noc_async_atomic_barrier();

    // Step 1: All workers disconnect from their mux connections
    for (uint32_t dir = 0; dir < num_directions; dir++) {
        if (directions[dir] && mux_connection_valid_arr[dir]) {
            tt::tt_fabric::fabric_client_disconnect(fabric_connections[dir]);
        }
    }

    // Step 2: Coordinate termination
    // - Termination masters wait for signals from other workers on their link
    // - Non-masters signal the termination master that they've disconnected
    //
    // Note: All directions for a given link share the same termination master,
    // so we only need to coordinate once per link (using any valid direction's semaphore).
    // We use the first valid direction's semaphore for coordination.

    // Find the first valid direction for this worker to use for coordination
    uint32_t coord_dir = num_directions;  // Invalid sentinel
    for (uint32_t dir = 0; dir < num_directions; dir++) {
        if (directions[dir] && mux_connection_valid_arr[dir]) {
            coord_dir = dir;
            break;
        }
    }

    if (coord_dir < num_directions) {
        if (any_is_termination_master) {
            // Termination master: wait for (num_mux_clients - 1) signals from other workers
            if (num_mux_clients > 1) {
                volatile uint32_t* termination_semaphore =
                    reinterpret_cast<volatile uint32_t*>(termination_sync_address_arr[coord_dir]);
                noc_semaphore_wait(termination_semaphore, num_mux_clients - 1);
            }

            // All workers have disconnected, now terminate the mux cores
            for (uint32_t dir = 0; dir < num_directions; dir++) {
                if (directions[dir] && mux_connection_valid_arr[dir]) {
                    tt::tt_fabric::fabric_endpoint_terminate(
                        fabric_mux_x_arr[dir], fabric_mux_y_arr[dir], fabric_mux_termination_signal_address);
                }
            }
        } else {
            // Non-master: signal the termination master that we've disconnected
            uint64_t termination_master_semaphore_noc_addr = get_noc_addr(
                termination_master_noc_x_arr[coord_dir],
                termination_master_noc_y_arr[coord_dir],
                termination_sync_address_arr[coord_dir]);
            noc_semaphore_inc(termination_master_semaphore_noc_addr, 1);
            noc_async_atomic_barrier();
        }
    }

    noc_async_write_barrier();
#else
    close_direction_connections(directions, fabric_connections);
    noc_async_write_barrier();
#endif
}
