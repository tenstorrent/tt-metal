// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "eth_l1_address_map.h"
#include "umd/device/tt_cluster_descriptor_types.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_types.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"


#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/program/program.hpp"

#include <vector>
#include <unordered_map>
namespace ttnn {
namespace ccl {


struct FabricEriscDatamoverConfig {
    static constexpr std::size_t field_size = 16;
    static constexpr std::size_t buffer_alignment = 32;
    static_assert(((buffer_alignment - 1) & buffer_alignment) == 0);

    // Global
    static constexpr std::size_t eth_channel_sync_size = 16;
    static constexpr std::size_t handshake_addr = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    static constexpr std::size_t edm_channel_ack_addr = handshake_addr + eth_channel_sync_size;
    static constexpr std::size_t termination_signal_address =
        edm_channel_ack_addr + (2 * eth_channel_sync_size);  // pad extra bytes to match old EDM so handshake logic will still work

    // Sender Channel 0
    static constexpr std::size_t sender_channel_0_buffer_index_address = termination_signal_address + field_size;
    static constexpr std::size_t sender_channel_0_worker_connection_info_address =
        sender_channel_0_buffer_index_address + field_size;
    static_assert(field_size >= sizeof(tt::fabric::EDMChannelWorkerLocationInfo));

    // Sender Channel 1
    static constexpr std::size_t sender_channel_1_buffer_index_address =
        sender_channel_0_worker_connection_info_address + field_size;
    static constexpr std::size_t sender_channel_1_worker_connection_info_address =
        sender_channel_1_buffer_index_address + field_size;

    // Receiver Channel
    static constexpr std::size_t receiver_channel_local_buffer_index_addr =
        sender_channel_1_worker_connection_info_address + field_size;

    // Channel Allocations
    static constexpr std::size_t buffer_region_start =
        (receiver_channel_local_buffer_index_addr + field_size + buffer_alignment) & ~(buffer_alignment - 1); // Align
    static constexpr std::size_t available_channel_buffering_space =
        eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - buffer_region_start;

    static_assert(sender_channel_1_buffer_index_address != sender_channel_0_buffer_index_address);

    FabricEriscDatamoverConfig(
        std::size_t channel_buffer_size_bytes, std::size_t sender_ratio_size, std::size_t receiver_ratio_size);

    std::size_t channel_buffer_size_bytes = 0;
    std::size_t channel_buffer_size_bytes_with_channel_sync = 0;
    std::size_t sender_0_channel_size_bytes = 0;
    std::size_t sender_0_num_buffers = 0;
    std::size_t sender_1_channel_size_bytes = 0;
    std::size_t sender_1_num_buffers = 0;
    std::size_t receiver_channel_size_bytes = 0;
    std::size_t receiver_num_buffers = 0;

    std::size_t sender_0_channel_base_address = 0;
    std::size_t sender_1_channel_base_address = 0;
    std::size_t receiver_channel_base_address = 0;
};

struct SenderWorkerAdapterSpec {
    size_t edm_noc_x = 0;
    size_t edm_noc_y = 0;
    size_t edm_buffer_base_addr = 0;
    size_t num_buffers_per_channel = 0;
    size_t edm_l1_sem_addr = 0;
    size_t edm_connection_handshake_addr = 0;
    size_t edm_worker_location_info_addr = 0;  // The EDM's location for `EDMChannelWorkerLocationInfo`
    size_t buffer_size_bytes = 0;
    size_t buffer_index_semaphore_id = 0; // the semaphore ID on the EDM, not the worker
};
class FabricEriscDatamoverBuilder {
   public:
    FabricEriscDatamoverBuilder(
        CoreCoord const& my_eth_core_logical,
        size_t my_noc_x,
        size_t my_noc_y,
        size_t my_chip_id,
        size_t peer_chip_id,

        std::optional<size_t> receiver_channel_downstream_flow_control_semaphore_id,
        size_t sender_channel_0_flow_control_semaphore_id,
        size_t sender_channel_1_flow_control_semaphore_id,
        size_t sender_channel_0_connection_semaphore_id,
        size_t sender_channel_1_connection_semaphore_id,
        size_t sender_channel_0_buffer_index_semaphore_id,
        size_t sender_channel_1_buffer_index_semaphore_id,

        FabricEriscDatamoverConfig const& config);

    static FabricEriscDatamoverBuilder build(
        tt::tt_metal::Device* device,
        tt::tt_metal::Program& program,
        CoreCoord const& ethernet_core,
        chip_id_t local_chip_id,
        chip_id_t peer_chip_id,
        FabricEriscDatamoverConfig const& config);

    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_worker_channel() const;
    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_fabric_channel() const;

    [[nodiscard]] std::vector<uint32_t> get_compile_time_args() const;

    [[nodiscard]] std::vector<uint32_t> get_runtime_args() const;

    void connect_to_downstream_edm(FabricEriscDatamoverBuilder const& downstream_edm);

    void dump_to_log() const {
        // TODO
    }

   private:
   friend class EdmLineFabricOpInterface;
    CoreCoord my_eth_core_logical;
    size_t my_noc_x = 0;
    size_t my_noc_y = 0;

    FabricEriscDatamoverConfig config;

    size_t my_chip_id = 0;
    size_t peer_chip_id = 0;
    size_t handshake_address = 0;
    size_t channel_buffer_size = 0;

    size_t sender_0_num_buffers = 0;
    size_t sender_1_num_buffers = 0;
    size_t receiver_num_buffers = 0;

    size_t local_sender_channel_0_buffer_address = 0;
    size_t local_sender_channel_0_connection_info_addr = 0;
    size_t local_sender_channel_1_buffer_address = 0;
    size_t local_sender_channel_1_connection_info_addr = 0;
    size_t local_receiver_channel_buffer_address = 0;

    size_t termination_signal_ptr = 0;

    // Semaphore IDs
    // this is the receiver channel's local sem for flow controlling with downstream fabric sender
    std::optional<size_t> receiver_channel_downstream_flow_control_semaphore_id;
    size_t sender_channel_0_flow_control_semaphore_id = 0;
    size_t sender_channel_1_flow_control_semaphore_id = 0;
    size_t sender_channel_0_connection_semaphore_id = 0;
    size_t sender_channel_1_connection_semaphore_id = 0;
    size_t sender_channel_0_buffer_index_semaphore_id = 0;
    size_t sender_channel_1_buffer_index_semaphore_id = 0;
    size_t receiver_channel_local_buffer_index_addr = 0;

    std::optional<size_t> downstream_edm_noc_x;
    std::optional<size_t> downstream_edm_noc_y;
    std::optional<size_t> downstream_edm_buffer_base_address;
    std::optional<size_t> downstream_edm_semaphore_address;
    std::optional<size_t> downstream_edm_worker_registration_address;
    std::optional<size_t> downstream_edm_worker_location_info_address;
    std::optional<size_t> downstream_sender_channel_buffer_index_semaphore_id;
};


struct edm_termination_info_t {
    uint32_t distance = 0;
    uint32_t edm_noc_x = 0;
    uint32_t edm_noc_y = 0;
    uint32_t termination_addr = 0;
};

struct EdmLineFabricOpInterface {
    enum Direction {
        // Ascending chips in the sequence
        FORWARD,

        // Descending chips in the sequence
        BACKWARD,
    };

    // Device ID -> EDM Builders
    std::unordered_map<size_t, std::vector<FabricEriscDatamoverBuilder>> edm_builders_forward_direction;
    std::unordered_map<size_t, std::vector<FabricEriscDatamoverBuilder>> edm_builders_backward_direction;

    // Device ID -> link index
    std::unordered_map<size_t, size_t> next_forward_direction_edm_available;
    std::unordered_map<size_t, size_t> next_backward_direction_edm_available;

    std::vector<Device*> device_sequence;
    std::vector<Program*> programs;

    size_t num_links = 0;

    //   The constructor will assemble/connect the line across the specified device sequence, for all available links.
    EdmLineFabricOpInterface (std::vector<Device*> const& device_sequence, std::vector<Program*> const& program_sequence, std::optional<size_t> desired_num_links = std::nullopt);


    // Will create a connection adapter for a worker which can be used to pass args to the worker kernel talking to the
    // corresponding fabric endpoint. This interface will guarantee unique connections only so requesting more unique connections
    // than available will result in an error.
    SenderWorkerAdapterSpec uniquely_connect_worker(tt::tt_metal::Device* device, Direction direction);

    // builds the ethernet kernels for all EDMs in the "fabric"
    void build_kernels() const;

    // Generates a list of target cores (for now assumed from chip 0 in the line) from farthest
    // to nearest for the sake of sending teardown/termination signals on workload completion.
    // Returns: A list of termination infos which can be passed to a terminate kernel
    // Note there is currently a small bug in that with multiple links, we don't currently know
    // who will be sending the termination signals (and which link(s) they are connected to)
    // and so a termination signal may be sent to our link first before the other eth core links
    // on the chip so multi-link isn't officially supported yet
    std::vector<edm_termination_info_t> generate_ordered_termination_info_farthest_to_nearest() const;
};

};  // namespace ccl
};  // namespace ttnn
