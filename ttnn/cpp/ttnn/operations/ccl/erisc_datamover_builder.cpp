// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/erisc_datamover_builder.hpp"

#include "common/math.hpp"
#include "erisc_datamover_builder.hpp"
#include "eth_l1_address_map.h"
#include "tt_metal/common/assert.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/program/program.hpp"

#include <vector>
namespace ttnn::ccl {


// The channel structure is as follows:
//              &header->  |----------------| channel_base_address
//                         |    header      |
//             &payload->  |----------------|
//                         |                |
//                         |    payload     |
//                         |                |
//        &channel_sync->  |----------------|
//                         |  channel_sync  |
//                         ------------------
//

FabricEriscDatamoverConfig::FabricEriscDatamoverConfig(
    std::size_t channel_buffer_size_bytes, std::size_t sender_ratio_size, std::size_t receiver_ratio_size) {
    const size_t min_buffer_size = sizeof(tt::fabric::PacketHeader) + 2 * FabricEriscDatamoverConfig::eth_channel_sync_size;
    TT_FATAL(channel_buffer_size_bytes >= min_buffer_size, "FabricEriscDatamoverConfig was constructed with `channel_buffer_size_bytes` argument set smaller than minimum size of {}", min_buffer_size);
    const std::size_t channel_buffer_size_with_channel_sync =
        channel_buffer_size_bytes + sizeof(tt::fabric::PacketHeader); // + 16 // sizeof(tt::fabric::PacketHeader);

    this->channel_buffer_size_bytes = channel_buffer_size_bytes;
    this->channel_buffer_size_bytes_with_channel_sync = channel_buffer_size_with_channel_sync;
    const std::size_t total_ratio_count = 2 * sender_ratio_size + receiver_ratio_size;
    this->sender_0_channel_size_bytes = tt::round_down(
        (available_channel_buffering_space / total_ratio_count) * sender_ratio_size,
        channel_buffer_size_with_channel_sync);
    this->sender_0_num_buffers = this->sender_0_channel_size_bytes / channel_buffer_size_with_channel_sync;
    this->sender_1_channel_size_bytes = tt::round_down(
        (available_channel_buffering_space / total_ratio_count) * sender_ratio_size,
        channel_buffer_size_with_channel_sync);
    this->sender_1_num_buffers = this->sender_1_channel_size_bytes / channel_buffer_size_with_channel_sync;
    this->receiver_channel_size_bytes = tt::round_down(
        (available_channel_buffering_space / total_ratio_count) * receiver_ratio_size,
        channel_buffer_size_with_channel_sync);
    this->receiver_num_buffers = this->receiver_channel_size_bytes / channel_buffer_size_with_channel_sync;

    this->sender_0_channel_base_address = buffer_region_start;
    this->sender_1_channel_base_address = this->sender_0_channel_base_address + this->sender_0_channel_size_bytes;
    this->receiver_channel_base_address = this->sender_1_channel_base_address + this->sender_1_channel_size_bytes;

    log_trace(tt::LogOp, "Sender 0 channel_start: {}", this->sender_0_channel_base_address);
    log_trace(tt::LogOp, "Sender 1 channel_start: {}", this->sender_1_channel_base_address);
    log_trace(tt::LogOp, "Receiver channel_start: {}", this->receiver_channel_base_address);

    static constexpr size_t total_num_channels = 3; // sender0, sender1, receiver
    const size_t max_channel_buffer_size = (available_channel_buffering_space / total_num_channels) - FabricEriscDatamoverConfig::eth_channel_sync_size - sizeof(tt::fabric::PacketHeader);
    TT_FATAL(channel_buffer_size_bytes <= max_channel_buffer_size, "Specified size of `channel_buffer_size_bytes` was too large. Maximum allowable size is {} B", max_channel_buffer_size);
    TT_FATAL(this->sender_0_channel_size_bytes > 0, "Internal error when computing `sender_0_channel_size_bytes` which was computed to be size 0");
    TT_FATAL(this->sender_1_channel_size_bytes > 0, "Internal error when computing `sender_1_channel_size_bytes` which was computed to be size 0");
    TT_FATAL(this->receiver_channel_size_bytes > 0, "Internal error when computing `receiver_channel_size_bytes` which was computed to be size 0");
    TT_FATAL(
        this->sender_0_channel_size_bytes + this->sender_1_channel_size_bytes + this->receiver_channel_size_bytes <=
        this->available_channel_buffering_space, "Internal error when computing channel sizes. Total channel size exceeds available space");
    TT_FATAL(
        this->receiver_channel_base_address + this->receiver_channel_size_bytes <
        eth_l1_mem::address_map::MAX_L1_LOADING_SIZE, "Internal error - channel buffers spilled past the end of usable L1 region.");
}

FabricEriscDatamoverBuilder::FabricEriscDatamoverBuilder(
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

    FabricEriscDatamoverConfig const& config) :
    my_eth_core_logical(my_eth_core_logical),
    my_noc_x(my_noc_x),
    my_noc_y(my_noc_y),
    config(config),
    my_chip_id(my_chip_id),
    peer_chip_id(peer_chip_id),
    handshake_address(tt::round_up(eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, FabricEriscDatamoverConfig::eth_channel_sync_size)),
    channel_buffer_size(config.channel_buffer_size_bytes),
    sender_0_num_buffers(config.sender_0_num_buffers),
    sender_1_num_buffers(config.sender_1_num_buffers),
    receiver_num_buffers(config.receiver_num_buffers),

    // this is the receiver channel's local sem for flow controlling with downstream fabric sender
    receiver_channel_downstream_flow_control_semaphore_id(receiver_channel_downstream_flow_control_semaphore_id),
    sender_channel_0_flow_control_semaphore_id(sender_channel_0_flow_control_semaphore_id),
    sender_channel_1_flow_control_semaphore_id(sender_channel_1_flow_control_semaphore_id),
    sender_channel_0_connection_semaphore_id(sender_channel_0_connection_semaphore_id),
    sender_channel_1_connection_semaphore_id(sender_channel_1_connection_semaphore_id),
    sender_channel_0_buffer_index_semaphore_id(sender_channel_0_buffer_index_semaphore_id),
    sender_channel_1_buffer_index_semaphore_id(sender_channel_1_buffer_index_semaphore_id),

    receiver_channel_local_buffer_index_addr(FabricEriscDatamoverConfig::receiver_channel_local_buffer_index_addr),

    local_sender_channel_0_buffer_address(config.sender_0_channel_base_address),
    local_sender_channel_0_connection_info_addr(
        FabricEriscDatamoverConfig::sender_channel_0_worker_connection_info_address),
    local_sender_channel_1_buffer_address(config.sender_1_channel_base_address),
    local_sender_channel_1_connection_info_addr(
        FabricEriscDatamoverConfig::sender_channel_1_worker_connection_info_address),
    local_receiver_channel_buffer_address(config.receiver_channel_base_address),

    termination_signal_ptr(FabricEriscDatamoverConfig::termination_signal_address) {}

std::vector<uint32_t> FabricEriscDatamoverBuilder::get_compile_time_args() const {
    const bool is_handshake_master = this->my_chip_id < this->peer_chip_id;
    TT_ASSERT(this->my_chip_id != this->peer_chip_id);
    TT_ASSERT(
        this->sender_0_num_buffers == this->sender_1_num_buffers);  //, "Implementation expects sender_0_num_buffers and
                                                                    // sender_1_num_buffers to be the same for now");
    log_trace(tt::LogTest, "Sender 0 num buffers: {}", this->sender_0_num_buffers);
    log_trace(tt::LogTest, "Sender 0 channel address: {}", this->local_sender_channel_0_buffer_address);
    log_trace(tt::LogTest, "Sender 1 num buffers: {}", this->sender_1_num_buffers);
    log_trace(tt::LogTest, "Sender 1 channel address: {}", this->local_sender_channel_1_buffer_address);
    log_trace(tt::LogTest, "Receiver num buffers: {}", this->receiver_num_buffers);
    log_trace(tt::LogTest, "Receiver channel address: {}", this->local_receiver_channel_buffer_address);
    return std::vector<uint32_t>{
        is_handshake_master,
        this->handshake_address,
        this->channel_buffer_size,

        this->sender_0_num_buffers,
        this->receiver_num_buffers,

        config.sender_0_channel_base_address,
        FabricEriscDatamoverConfig::sender_channel_0_worker_connection_info_address,
        config.sender_1_channel_base_address,
        FabricEriscDatamoverConfig::sender_channel_1_worker_connection_info_address,
        config.receiver_channel_base_address,
        config.receiver_channel_base_address,

        config.sender_0_channel_base_address,
        config.sender_1_channel_base_address,

        this->termination_signal_ptr};
}

std::vector<uint32_t> FabricEriscDatamoverBuilder::get_runtime_args() const {
    return std::vector<uint32_t>{
        this->sender_channel_0_connection_semaphore_id,
        this->sender_channel_1_connection_semaphore_id,
        this->sender_channel_0_buffer_index_semaphore_id,
        this->downstream_sender_channel_buffer_index_semaphore_id.value_or(-1),
        this->downstream_edm_buffer_base_address != std::nullopt,
        this->downstream_edm_buffer_base_address.value_or(0),
        this->downstream_edm_noc_x.value_or(0),
        this->downstream_edm_noc_y.value_or(0),
        this->downstream_edm_semaphore_address.value_or(-1),
        this->downstream_edm_worker_registration_address.value_or(0),
        this->downstream_edm_worker_location_info_address.value_or(0),
        this->receiver_channel_local_buffer_index_addr,
        // this is the receiver channel's local sem for flow controlling with downstream fabric sender
        this->receiver_channel_downstream_flow_control_semaphore_id.value_or(0),
        this->sender_channel_0_flow_control_semaphore_id,
        this->sender_channel_1_flow_control_semaphore_id
    };
}

FabricEriscDatamoverBuilder FabricEriscDatamoverBuilder::build(
    Device* device,
    Program& program,
    CoreCoord const& ethernet_core,
    chip_id_t local_chip_id,
    chip_id_t peer_chip_id,
    FabricEriscDatamoverConfig const& config) {
    std::optional<size_t> receiver_channel_downstream_flow_control_semaphore_id = std::nullopt;
    auto sender_channel_0_flow_control_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
    auto sender_channel_1_flow_control_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
    auto sender_channel_0_connection_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
    auto sender_channel_1_connection_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
    auto sender_channel_0_buffer_index_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
    auto sender_channel_1_buffer_index_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);

    return FabricEriscDatamoverBuilder(
        ethernet_core,
        device->translated_ethernet_core_from_logical_core(ethernet_core).x,
        device->translated_ethernet_core_from_logical_core(ethernet_core).y,
        local_chip_id,
        peer_chip_id,

        receiver_channel_downstream_flow_control_semaphore_id,
        sender_channel_0_flow_control_semaphore_id,
        sender_channel_1_flow_control_semaphore_id,
        sender_channel_0_connection_semaphore_id,
        sender_channel_1_connection_semaphore_id,
        sender_channel_0_buffer_index_semaphore_id,
        sender_channel_1_buffer_index_semaphore_id,

        config);
}

SenderWorkerAdapterSpec FabricEriscDatamoverBuilder::build_connection_to_worker_channel() const {
    return SenderWorkerAdapterSpec {
        this->my_noc_x,
        this->my_noc_y,
        this->local_sender_channel_0_buffer_address,
        this->sender_0_num_buffers,
        this->sender_channel_0_flow_control_semaphore_id,
        this->sender_channel_0_connection_semaphore_id,
        FabricEriscDatamoverConfig::sender_channel_0_worker_connection_info_address,
        this->config.channel_buffer_size_bytes,
        this->sender_channel_0_buffer_index_semaphore_id
    };
}


SenderWorkerAdapterSpec FabricEriscDatamoverBuilder::build_connection_to_fabric_channel() const {
    return SenderWorkerAdapterSpec {
        this->my_noc_x,
        this->my_noc_y,
        this->local_sender_channel_1_buffer_address,
        this->sender_1_num_buffers,
        this->sender_channel_1_flow_control_semaphore_id,
        this->sender_channel_1_connection_semaphore_id,
        FabricEriscDatamoverConfig::sender_channel_1_worker_connection_info_address,
        this->config.channel_buffer_size_bytes,
        this->sender_channel_1_buffer_index_semaphore_id
    };
}

void FabricEriscDatamoverBuilder::connect_to_downstream_edm(FabricEriscDatamoverBuilder const& downstream_edm) {
    auto const adapter_spec = downstream_edm.build_connection_to_fabric_channel();

    log_trace(tt::LogTest, "Connecting to downstream EDM at x={}, y={}", adapter_spec.edm_noc_x, adapter_spec.edm_noc_y);

    this->downstream_edm_noc_x = adapter_spec.edm_noc_x;
    this->downstream_edm_noc_y = adapter_spec.edm_noc_y;
    this->downstream_edm_buffer_base_address = adapter_spec.edm_buffer_base_addr;
    this->downstream_edm_semaphore_address = adapter_spec.edm_l1_sem_addr;
    this->downstream_edm_worker_registration_address = adapter_spec.edm_connection_handshake_addr;
    this->downstream_edm_worker_location_info_address = adapter_spec.edm_worker_location_info_addr;
    this->downstream_sender_channel_buffer_index_semaphore_id = adapter_spec.buffer_index_semaphore_id;
}



EdmLineFabricOpInterface::EdmLineFabricOpInterface (std::vector<Device*> const& device_sequence, std::vector<Program*> const& program_sequence, std::optional<size_t> desired_num_links) :
    device_sequence(device_sequence),
    programs(program_sequence) {
    static constexpr std::size_t edm_buffer_size = 4096 + sizeof(tt::fabric::PacketHeader);
    auto const config = FabricEriscDatamoverConfig(edm_buffer_size, 1, 2);
    TT_ASSERT(device_sequence.size() == program_sequence.size());

    for (size_t i = 0; i < device_sequence.size(); i++) {
        log_trace(tt::LogOp, "device[{}] id={}",  i, device_sequence[i]->id());
    }


    // Construct the builders
    for (size_t hop = 0; hop < device_sequence.size() - 1; hop++) {
        auto src_device = device_sequence[hop];
        auto dest_device = device_sequence[hop + 1];

        auto const& src_device_sockets = src_device->get_ethernet_sockets(dest_device->id());;
        auto const& dest_device_sockets = dest_device->get_ethernet_sockets(src_device->id());;
        std::vector<CoreCoord> local_link_cores; local_link_cores.reserve(src_device_sockets.size());
        std::vector<CoreCoord> remote_link_cores; remote_link_cores.reserve(dest_device_sockets.size());
        std::copy_if(src_device_sockets.begin(), src_device_sockets.end(), std::back_inserter(local_link_cores), [src_device](CoreCoord const& core) { return src_device->is_active_ethernet_core(core, true); });
        std::copy_if(dest_device_sockets.begin(), dest_device_sockets.end(), std::back_inserter(remote_link_cores), [dest_device](CoreCoord const& core) { return dest_device->is_active_ethernet_core(core, true); });

        this->num_links = std::min(desired_num_links.value_or(std::numeric_limits<std::size_t>::max()), local_link_cores.size());

        TT_ASSERT(local_link_cores.size() == remote_link_cores.size());

        edm_builders_forward_direction[src_device->id()].reserve(local_link_cores.size());
        edm_builders_forward_direction[dest_device->id()].reserve(local_link_cores.size());
        for (size_t l = 0; l < this->num_links; l++) {
            log_trace(tt::LogOp, "Building forward direction EDM on chip {} on link {}", src_device->id(), edm_builders_forward_direction[src_device->id()].size());
            edm_builders_forward_direction[src_device->id()].push_back(FabricEriscDatamoverBuilder::build(
                device_sequence[hop],
                *programs[hop],
                local_link_cores[l],
                src_device->id(),
                dest_device->id(),
                config));

            log_trace(tt::LogOp, "Building backward direction EDM on chip {} on link {}", dest_device->id(), edm_builders_backward_direction[dest_device->id()].size());
            edm_builders_backward_direction[dest_device->id()].push_back(FabricEriscDatamoverBuilder::build(
                device_sequence[hop + 1],
                *programs[hop + 1],
                remote_link_cores[l],
                dest_device->id(),
                src_device->id(),
                config));
        }
    }

    // Establish local connections between EDMs on the same chips to establish the lin fabric
    for (size_t i = 1; i < device_sequence.size() - 1; i++) {
        const size_t num_links = edm_builders_forward_direction.at(device_sequence[i]->id()).size();
        auto& forward_direction_edm = edm_builders_forward_direction.at(device_sequence[i]->id());
        auto& backward_direction_edm = edm_builders_backward_direction.at(device_sequence[i]->id());

        for (size_t l = 0; l < num_links; l++) {
            forward_direction_edm.at(l).connect_to_downstream_edm(backward_direction_edm.at(l));
            backward_direction_edm.at(l).connect_to_downstream_edm(forward_direction_edm.at(l));
        }
    }

}


SenderWorkerAdapterSpec EdmLineFabricOpInterface::uniquely_connect_worker(Device* device, Direction direction) {
    TT_ASSERT((direction == FORWARD) ? edm_builders_forward_direction.find(device->id()) != edm_builders_forward_direction.end()
                                     : edm_builders_backward_direction.find(device->id()) != edm_builders_backward_direction.end());
    auto& edm_builders = (direction == FORWARD) ? edm_builders_forward_direction.at(device->id())
                                                : edm_builders_backward_direction.at(device->id());
    auto &link_count_map = (direction == FORWARD) ? next_forward_direction_edm_available : next_backward_direction_edm_available;
    const auto next_link = link_count_map[device->id()];
    link_count_map[device->id()] = next_link + 1;

    TT_ASSERT(edm_builders.size() > 0);
    TT_ASSERT(next_link < edm_builders.size());
    return edm_builders.at(next_link).build_connection_to_worker_channel();
}

void EdmLineFabricOpInterface::build_kernels() const {
    auto generate_kernels_in_direction = [this](Device *device, Program *program, Direction direction) {
        auto &edm_builders = direction == FORWARD ? edm_builders_forward_direction : edm_builders_backward_direction;
        if (edm_builders.find(device->id()) != edm_builders.end()) {
            for (auto& edm_builder : edm_builders.at(device->id())) {
                auto local_edm_kernel = ttnn::ccl::generate_edm_kernel(
                    *program,
                    device,
                    edm_builder,
                    edm_builder.my_eth_core_logical,
                    NOC::NOC_0);
            }
        }
    };

    TT_ASSERT(device_sequence.size() == programs.size());
    for (size_t i = 0; i < device_sequence.size(); i++) {
        Program* program = programs[i];
        Device* device = device_sequence[i];
        generate_kernels_in_direction(device, program, Direction::FORWARD);
        generate_kernels_in_direction(device, program, Direction::BACKWARD);
    }
}



std::vector<edm_termination_info_t> EdmLineFabricOpInterface::generate_ordered_termination_info_farthest_to_nearest() const {
    TT_ASSERT(device_sequence.size() > 0);
    const size_t num_hops = device_sequence.size() - 1;
    TT_ASSERT(num_hops > 0);
    std::vector<edm_termination_info_t> edm_termination_infos;
    edm_termination_infos.reserve(num_hops * 2 * this->num_links);
    for (int i = num_hops - 1; i >= 0; i--) {
        log_trace(tt::LogOp, "Generating termination info for hop {}", i);
        TT_ASSERT(i + 1 != 0);
        TT_ASSERT(i + 1 < device_sequence.size());
        TT_ASSERT(edm_builders_backward_direction.find(device_sequence[i+1]->id()) != edm_builders_backward_direction.end(), "Device {} at index {} not found in `edm_builders_backward_direction` but it was expected there", i + 1, device_sequence[i+1]->id());
        TT_ASSERT(edm_builders_forward_direction.find(device_sequence[i]->id()) != edm_builders_forward_direction.end(), "Device {} at index {} not found in `edm_builders_forward_direction` but it was expected there", i, device_sequence[i]->id());
        auto &farther_edms = edm_builders_backward_direction.at(device_sequence[i+1]->id());
        auto &nearer_edms = edm_builders_forward_direction.at(device_sequence[i]->id());

        TT_ASSERT(farther_edms.size() <= this->num_links);
        TT_ASSERT(nearer_edms.size() <= this->num_links);
        for (size_t l = 0; l < this->num_links; l++) {
            auto &farther_edm = farther_edms.at(l);
            const std::size_t distance_receiver = i + 1;
            edm_termination_infos.push_back(
                {distance_receiver,
                farther_edm.my_noc_x,
                farther_edm.my_noc_y,
             ttnn::ccl::FabricEriscDatamoverConfig::termination_signal_address});
        }
        for (size_t l = 0; l < this->num_links; l++) {
            auto &nearer_edm = nearer_edms.at(l);
            const std::size_t distance_sender = i;
            edm_termination_infos.push_back(
                {distance_sender,
                nearer_edm.my_noc_x,
                nearer_edm.my_noc_y,
                ttnn::ccl::FabricEriscDatamoverConfig::termination_signal_address});
        }
    }
    log_trace(tt::LogOp, "Done Generating termination infos");
    return edm_termination_infos;
}





}  // namespace ttnn::ccl
