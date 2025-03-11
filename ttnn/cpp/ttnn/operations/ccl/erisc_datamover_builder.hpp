// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/distributed/types.hpp"
#include "umd/device/types/cluster_descriptor_types.h"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_types.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"

#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/edm_fabric_counters.hpp"

#include <tt-metalium/assert.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program_impl.hpp>
#include <tt-metalium/hal_exp.hpp>

#include <vector>
#include <unordered_map>
#include <optional>

namespace ttnn {
namespace ccl {

struct FabricEriscDatamoverConfig {
    static constexpr std::size_t num_sender_channels = 3;
    static constexpr std::size_t num_receiver_channels = 2;
    static constexpr bool constrain_to_power_of_2_buffer_slot_counts = true;

    static constexpr std::size_t field_size = 16;
    static constexpr std::size_t buffer_alignment = 32;
    static constexpr std::size_t eth_word_l1_alignment = 16;
    static_assert(((buffer_alignment - 1) & buffer_alignment) == 0);
    static constexpr bool enable_fabric_counters = false;
    static constexpr bool enable_fabric_pkt_header_recording = false;

    // Global
    static constexpr std::size_t eth_channel_sync_size = 16;
    std::size_t handshake_addr;
    std::size_t edm_channel_ack_addr;
    std::size_t termination_signal_address;  // pad extra bytes to match old EDM so handshake logic will still work

    // Debug and Counters
    static constexpr std::size_t receiver_channel_counters_size_bytes =
        (((tt::fabric::receiver_channel_counters_l1_size - 1) / field_size) + 1) * field_size;
    static constexpr std::size_t sender_channel_counters_size_bytes =
        (((tt::fabric::sender_channel_counters_l1_size - 1) / field_size) + 1) * field_size;

    std::array<std::size_t, num_receiver_channels> receiver_channels_counters_address;
    std::array<std::size_t, num_sender_channels> sender_channels_counters_address;

    // Packet header history buffer(s)
    static constexpr std::size_t receiver_completed_packet_header_cb_size_headers = 32;
    static constexpr std::size_t receiver_completed_packet_header_cb_size_bytes =
        sizeof(tt::fabric::PacketHeader) * receiver_completed_packet_header_cb_size_headers;
    static constexpr std::size_t sender_completed_packet_header_cb_size_headers = 32;
    static constexpr std::size_t sender_completed_packet_header_cb_size_bytes =
        sizeof(tt::fabric::PacketHeader) * sender_completed_packet_header_cb_size_headers;
    std::array<std::size_t, num_receiver_channels> receivers_completed_packet_header_cb_address;
    std::array<std::size_t, num_sender_channels> senders_completed_packet_header_cb_address;

    // ----------- Sender Channels
    std::array<std::size_t, num_sender_channels> sender_channels_buffer_index_address;
    // Connection info layout:
    // 0: buffer_index_rdptr -> Tells EDM the address in worker L1 to update EDM's copy of channel rdptr
    // 1: worker_teardown_semaphore_address -> Tells EDM where to signal connection teardown completion in worker's L1
    // 2: WorkerXY (as uint32_t)
    // 3: Hold's EDM's rdptr for the buffer index in the channel
    std::array<std::size_t, num_sender_channels> sender_channels_worker_conn_info_base_address;
    std::array<std::size_t, num_sender_channels> sender_channels_local_flow_control_semaphore_address;
    std::array<std::size_t, num_sender_channels> sender_channels_producer_terminate_connection_address;
    // persistent mode field
    std::array<std::size_t, num_sender_channels> sender_channels_connection_semaphore_address;
    // persistent mode field
    std::array<std::size_t, num_sender_channels> sender_channels_buffer_index_semaphore_address;

    static_assert(sizeof(tt::fabric::EDMChannelWorkerLocationInfo) % field_size == 0);

    // ----------- Receiver Channels
    std::array<std::size_t, num_receiver_channels> receiver_channels_local_buffer_index_address;
    // persistent mode field
    std::array<std::size_t, num_receiver_channels> receiver_channels_downstream_flow_control_semaphore_address;

    // Channel Allocations
    std::size_t max_l1_loading_size;
    ;
    std::size_t buffer_region_start;
    std::size_t available_channel_buffering_space;

    FabricEriscDatamoverConfig(
        std::size_t channel_buffer_size_bytes,
        std::size_t sender_ratio_size,
        std::size_t receiver_ratio_size,
        Topology topology = Topology::Linear);

    std::size_t channel_buffer_size_bytes = 0;
    std::size_t channel_buffer_size_bytes_with_channel_sync = 0;

    std::array<std::size_t, num_sender_channels> sender_channels_size_bytes;
    std::array<std::size_t, num_receiver_channels> receiver_channels_size_bytes;
    std::array<std::size_t, num_sender_channels> sender_channels_num_buffers;
    std::array<std::size_t, num_receiver_channels> receiver_channels_num_buffers;

    std::array<std::size_t, num_sender_channels> sender_channels_base_address;
    std::array<std::size_t, num_receiver_channels> receiver_channels_base_address;

    std::size_t num_used_sender_channels = 0;
    std::size_t num_used_receiver_channels = 0;

    Topology topology = Topology::Linear;

private:
    FabricEriscDatamoverConfig();
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
    size_t buffer_index_semaphore_id = 0;  // the semaphore ID on the EDM, not the worker
    bool persistent_fabric = false;
};

struct edm_termination_info_t {
    uint32_t distance = 0;
    uint32_t edm_noc_x = 0;
    uint32_t edm_noc_y = 0;
    uint32_t termination_addr = 0;
};

void get_runtime_args_for_edm_termination_infos(
    const std::vector<edm_termination_info_t>& edm_termination_infos, std::vector<uint32_t>& args_out);
void append_worker_to_fabric_edm_sender_rt_args(
    const SenderWorkerAdapterSpec& connection,
    size_t sender_worker_flow_control_semaphore_id,
    size_t sender_worker_teardown_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out);
size_t log_worker_to_fabric_edm_sender_rt_args(const std::vector<uint32_t>& args, size_t starting_arg_idx = 0);

class FabricEriscDatamoverBuilder {
public:
    static constexpr size_t default_firmware_context_switch_interval = 10000;
    // payload only, no header
    static constexpr size_t default_packet_payload_size_bytes = tt::tile_size(tt::DataFormat::Bfp8_b) * 4;

    static constexpr uint32_t num_virtual_channels = 2;

    FabricEriscDatamoverBuilder(
        const CoreCoord& my_eth_core_logical,
        size_t my_noc_x,
        size_t my_noc_y,
        size_t my_chip_id,
        size_t peer_chip_id,

        const std::array<std::optional<size_t>, FabricEriscDatamoverConfig::num_receiver_channels>&
            receiver_channels_downstream_flow_control_semaphore_id,
        const std::array<std::optional<size_t>, FabricEriscDatamoverConfig::num_receiver_channels>&
            receiver_channels_downstream_teardown_semaphore_id,
        const std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>&
            sender_channels_flow_control_semaphore_id,
        const std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>&
            sender_channels_connection_semaphore_id,
        const std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>&
            sender_channels_buffer_index_semaphore_id,

        const FabricEriscDatamoverConfig& config,
        bool enable_persistent_mode,
        bool build_in_worker_connection_mode = false);

    static FabricEriscDatamoverBuilder build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        const CoreCoord& ethernet_core,
        chip_id_t local_chip_id,
        chip_id_t peer_chip_id,
        const FabricEriscDatamoverConfig& config,
        bool enable_persistent_mode,
        bool build_in_worker_connection_mode = false);

    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_worker_channel() const;
    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t vc) const;

    [[nodiscard]] std::vector<uint32_t> get_compile_time_args() const;

    [[nodiscard]] std::vector<uint32_t> get_runtime_args() const;

    void connect_to_downstream_edm(const FabricEriscDatamoverBuilder& downstream_edm);

    void dump_to_log() const {
        // TODO
    }

    void teardown_from_host(
        tt::tt_metal::IDevice* d,
        tt::fabric::TerminationSignal termination_signal = tt::fabric::TerminationSignal::GRACEFULLY_TERMINATE) const;

    void set_firmware_context_switch_interval(size_t interval);

    //    protected:
    friend class EdmLineFabricOpInterface;
    CoreCoord my_eth_core_logical;
    size_t my_noc_x = 0;
    size_t my_noc_y = 0;

    FabricEriscDatamoverConfig config;

    size_t my_chip_id = 0;
    size_t peer_chip_id = 0;
    size_t handshake_address = 0;
    size_t channel_buffer_size = 0;

    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_num_buffers;
    std::array<size_t, FabricEriscDatamoverConfig::num_receiver_channels> receiver_channels_num_buffers;

    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> local_sender_channels_buffer_address;
    std::array<size_t, FabricEriscDatamoverConfig::num_receiver_channels> local_receiver_channels_buffer_address;

    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> local_sender_channels_connection_info_addr;

    size_t termination_signal_ptr = 0;

    // Semaphore IDs
    // this is the receiver channel's local sem for flow controlling with downstream fabric sender
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::num_receiver_channels>
        receiver_channels_downstream_flow_control_semaphore_id;
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::num_receiver_channels>
        receiver_channels_downstream_teardown_semaphore_id;
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_flow_control_semaphore_id;
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_connection_semaphore_id;
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_buffer_index_semaphore_id;
    std::array<size_t, FabricEriscDatamoverConfig::num_receiver_channels> receiver_channels_local_buffer_index_address;

    std::array<std::optional<size_t>, num_virtual_channels> downstream_edm_vcs_noc_x;
    std::array<std::optional<size_t>, num_virtual_channels> downstream_edm_vcs_noc_y;
    std::array<std::optional<size_t>, num_virtual_channels> downstream_edm_vcs_buffer_base_address;
    std::array<std::optional<size_t>, num_virtual_channels> downstream_edm_vcs_semaphore_address;
    std::array<std::optional<size_t>, num_virtual_channels> downstream_edm_vcs_worker_registration_address;
    std::array<std::optional<size_t>, num_virtual_channels> downstream_edm_vcs_worker_location_info_address;
    std::array<std::optional<size_t>, num_virtual_channels> downstream_vcs_sender_channel_buffer_index_semaphore_id;

    bool enable_persistent_mode = false;
    bool build_in_worker_connection_mode = false;
    size_t firmware_context_switch_interval = default_firmware_context_switch_interval;
    bool enable_first_level_ack = false;
    bool fuse_receiver_flush_and_completion_ptr = true;
};

class EdmLineFabricOpInterface {
public:
    enum Direction {
        // Ascending chips in the sequence
        FORWARD,

        // Descending chips in the sequence
        BACKWARD,
    };

    //   The constructor will assemble/connect the line across the specified device sequence, for all available links.
    EdmLineFabricOpInterface(
        const std::vector<tt::tt_metal::IDevice*>& device_sequence,
        const std::vector<tt::tt_metal::Program*>& program_sequence,
        bool enable_persistent_mode,
        std::optional<size_t> desired_num_links = std::nullopt,
        bool build_in_worker_connection_mode = false,
        Topology topology = Topology::Linear);

    // Invocable per chip if we want to collectively build the fabric by building this separately per chip
    // (and implicitly building the fabric that way)
    EdmLineFabricOpInterface(
        tt::tt_metal::IDevice* local_device,
        std::optional<tt::tt_metal::IDevice*> forward_device,
        std::optional<tt::tt_metal::IDevice*> backward_device,
        tt::tt_metal::Program* program,
        bool enable_persistent_mode,
        std::optional<size_t> desired_num_links,
        bool build_in_worker_connection_mode = false,
        Topology topology = Topology::Linear);

    static EdmLineFabricOpInterface build_program_builder_worker_connection_fabric(
        const std::vector<tt::tt_metal::IDevice*>& device_sequence,
        const std::vector<tt::tt_metal::Program*>& program_sequence,
        bool enable_persistent_mode,
        std::optional<size_t> desired_num_links = std::nullopt,
        Topology topology = Topology::Linear);
    static EdmLineFabricOpInterface build_program_builder_worker_connection_fabric(
        tt::tt_metal::IDevice* local_device,
        tt::tt_metal::IDevice* forward_device,
        tt::tt_metal::IDevice* backward_device,
        tt::tt_metal::Program* program,
        bool enable_persistent_mode,
        std::optional<size_t> desired_num_links = std::nullopt,
        Topology topology = Topology::Linear);

    // Will create a connection adapter for a worker which can be used to pass args to the worker kernel talking to the
    // corresponding fabric endpoint. This interface will guarantee unique connections only so requesting more unique
    // connections than available will result in an error.
    SenderWorkerAdapterSpec uniquely_connect_worker(tt::tt_metal::IDevice* device, Direction direction);

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

    // Generates a list of termination infos for the local chip's EDMs
    std::vector<edm_termination_info_t> generate_local_chip_fabric_termination_infos(
        tt::tt_metal::IDevice* device) const;

    // Accessors
    size_t get_num_links() const { return num_links; }

    size_t get_device_count() const { return device_sequence.size(); }

    size_t get_index_of_device(tt::tt_metal::IDevice* device) const {
        for (size_t i = 0; i < device_sequence.size(); i++) {
            if (device_sequence[i] == device) {
                return i;
            }
        }
        TT_THROW("Device {} not found in device sequence of line fabric", device->id());
        return -1;
    }

    size_t get_edm_buffer_size_bytes() const { return buffer_size_bytes; }

    void teardown_from_host(
        tt::fabric::TerminationSignal termination_signal = tt::fabric::TerminationSignal::GRACEFULLY_TERMINATE) const;

    static void launch_mesh_fabric(MeshDevice* mesh_device);
    static void teardown_edm_fabric(MeshDevice* mesh_device);

    void set_firmware_context_switch_interval(size_t interval);

    // Device ID -> EDM Builders
    std::unordered_map<size_t, std::vector<FabricEriscDatamoverBuilder>> edm_builders_forward_direction;
    std::unordered_map<size_t, std::vector<FabricEriscDatamoverBuilder>> edm_builders_backward_direction;

private:
    // Device ID -> link index
    std::unordered_map<size_t, size_t> next_forward_direction_edm_available;
    std::unordered_map<size_t, size_t> next_backward_direction_edm_available;

    std::vector<tt::tt_metal::IDevice*> device_sequence;
    std::vector<tt::tt_metal::Program*> programs;

    size_t num_links;
    size_t buffer_size_bytes;
    size_t firmware_context_switch_interval = FabricEriscDatamoverBuilder::default_firmware_context_switch_interval;
};

void initialize_edm_fabric(
    distributed::MeshDevice* mesh_device,
    bool wrap_fabric_around_mesh = false,
    std::optional<size_t> context_switch_interval_override = std::nullopt,
    Topology topology = Topology::Linear);
void teardown_edm_fabric(
    distributed::MeshDevice* mesh_device, bool wrap_fabric_around_mesh = false, Topology topology = Topology::Linear);

};  // namespace ccl
};  // namespace ttnn
