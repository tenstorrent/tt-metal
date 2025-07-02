// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "erisc_datamover_builder_helper.hpp"

namespace ttnn::ccl {

std::vector<CoreCoord> reorder_connected_sockets(
    const tt::tt_metal::IDevice* local_device, const std::vector<CoreCoord>& connected_sockets) {
    // Prepare storage
    std::vector<CoreCoord> reordered_connected_sockets;
    reordered_connected_sockets.reserve(connected_sockets.size());

    // Create a vector of <logical, virtual> pairs
    std::vector<std::pair<CoreCoord, CoreCoord>> ethernet_cores_logical_virtual;
    ethernet_cores_logical_virtual.reserve(connected_sockets.size());

    // Build the pair list
    for (auto core : connected_sockets) {
        auto core_physical = local_device->virtual_core_from_logical_core(core, CoreType::ETH);
        ethernet_cores_logical_virtual.emplace_back(core, core_physical);
    }

    // Sort by the 'x' coordinate of the virtual (physical) core
    std::sort(
        ethernet_cores_logical_virtual.begin(), ethernet_cores_logical_virtual.end(), [](const auto& a, const auto& b) {
            return a.second.x < b.second.x;
        });

    // Extract the reordered logical sockets
    for (auto& core_pair : ethernet_cores_logical_virtual) {
        reordered_connected_sockets.push_back(core_pair.first);
    }

    return reordered_connected_sockets;
}

EdmLineFabricOpInterface::EdmLineFabricOpInterface(
    const std::vector<tt::tt_metal::IDevice*>& device_sequence,
    const std::vector<tt::tt_metal::Program*>& program_sequence,
    std::optional<size_t> desired_num_links,
    bool build_in_worker_connection_mode,
    Topology topology,
    bool is_galaxy,
    const tt::tt_fabric::FabricRouterBufferConfig& edm_buffer_config) :
    device_sequence(device_sequence), programs(program_sequence) {
    if (topology == Topology::Ring) {
        TT_FATAL(device_sequence.size() > 2, "Ring topology only supports more than 2 devices");
    }

    static constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes +
        sizeof(tt::tt_fabric::PacketHeader);
    const auto config = tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size, topology);
    TT_ASSERT(device_sequence.size() == program_sequence.size());

    for (size_t i = 0; i < device_sequence.size(); i++) {
        log_trace(tt::LogOp, "device[{}] id={}", i, device_sequence[i]->id());
    }
    auto get_min_link_count = [&](IDevice* src_device, IDevice* dest_device, size_t min_link_count) {
        const auto& src_device_sockets = src_device->get_ethernet_sockets(dest_device->id());
        const auto& dest_device_sockets = dest_device->get_ethernet_sockets(src_device->id());
        if (src_device_sockets.size() > 0) {
            min_link_count = std::min(min_link_count, src_device_sockets.size());
        }
        if (src_device_sockets.size() > 0) {
            min_link_count = std::min(min_link_count, dest_device_sockets.size());
        }
        return min_link_count;
    };

    size_t min_link_count = desired_num_links.value_or(std::numeric_limits<size_t>::max());
    for (size_t hop = 0; hop < device_sequence.size() - 1; hop++) {
        auto src_device = device_sequence[hop];
        auto dest_device = device_sequence[hop + 1];
        min_link_count = get_min_link_count(src_device, dest_device, min_link_count);
    }
    if (topology == Topology::Ring) {
        auto src_device = device_sequence.back();
        auto dest_device = device_sequence.front();
        min_link_count = get_min_link_count(src_device, dest_device, min_link_count);
    }

    this->num_links = min_link_count;

    auto build_edm_directions =
        [&](IDevice* src_device, IDevice* dest_device, Program* src_program, Program* dest_program) {
            const auto& src_device_sockets = src_device->get_ethernet_sockets(dest_device->id());
            const auto& dest_device_sockets = dest_device->get_ethernet_sockets(src_device->id());
            // re-order the connected_sockets based on virtual coords
            auto reordered_src_device_sockets = reorder_connected_sockets(src_device, src_device_sockets);
            auto reordered_dest_device_sockets = reorder_connected_sockets(dest_device, dest_device_sockets);

            std::vector<CoreCoord> local_link_cores;
            local_link_cores.reserve(reordered_src_device_sockets.size());
            std::vector<CoreCoord> remote_link_cores;
            remote_link_cores.reserve(reordered_dest_device_sockets.size());
            std::copy_if(
                reordered_src_device_sockets.begin(),
                reordered_src_device_sockets.end(),
                std::back_inserter(local_link_cores),
                [src_device](const CoreCoord& core) { return src_device->is_active_ethernet_core(core, true); });
            std::copy_if(
                reordered_dest_device_sockets.begin(),
                reordered_dest_device_sockets.end(),
                std::back_inserter(remote_link_cores),
                [dest_device](const CoreCoord& core) { return dest_device->is_active_ethernet_core(core, true); });

            TT_ASSERT(local_link_cores.size() == remote_link_cores.size());
            // set edm types based on topology and device ids.
            bool dateline = false;
            auto src_device_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Default;
            auto dest_device_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Default;
            if (topology == Topology::Ring) {
                if (src_device->id() == device_sequence.back()->id() &&
                    dest_device->id() == device_sequence.front()->id()) {
                    src_device_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Dateline;
                    dest_device_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Dateline;
                    dateline = true;
                } else if (
                    src_device->id() == device_sequence.front()->id() &&
                    dest_device->id() != device_sequence.back()->id()) {
                    src_device_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstream;
                    dest_device_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice;
                } else if (
                    src_device->id() != device_sequence.front()->id() &&
                    dest_device->id() == device_sequence.back()->id()) {
                    src_device_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice;
                    dest_device_edm_type = tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstream;
                } else if (
                    src_device->id() == device_sequence.at(1)->id() &&
                    dest_device->id() != device_sequence.front()->id()) {
                    src_device_edm_type =
                        tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDeviceUpstream;
                    if (dest_device->id() == device_sequence.at(device_sequence.size() - 2)->id()) {
                        dest_device_edm_type =
                            tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDeviceUpstream;
                    }
                } else if (
                    src_device->id() == device_sequence.at(device_sequence.size() - 3)->id() &&
                    dest_device->id() == device_sequence.at(device_sequence.size() - 2)->id()) {
                    dest_device_edm_type =
                        tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDeviceUpstream;
                }
            }

            auto edm_axis = tt::tt_fabric::FabricEriscDatamoverAxis::Short;
            // change to long axis variantion, and using more buffer slots.
            if (device_sequence.size() >=
                tt::tt_fabric::FabricEriscDatamoverConfig::MESH_LONG_AXIS_OPTIMIZATION_THRESHOLD) {
                edm_axis = tt::tt_fabric::FabricEriscDatamoverAxis::Long;
            }
            // if ring topology set extra buffer on dateline edms.
            auto src_edm_options = tt::tt_fabric::FabricEriscDatamoverOptions{
                .edm_type = src_device_edm_type,
                .edm_axis = edm_axis,
                .edm_buffer_config = edm_buffer_config,
            };
            auto dest_edm_options = tt::tt_fabric::FabricEriscDatamoverOptions{
                .edm_type = dest_device_edm_type,
                .edm_axis = edm_axis,
                .edm_buffer_config = edm_buffer_config,
            };
            const auto src_curr_edm_config =
                tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size, topology, src_edm_options);
            const auto dest_curr_edm_config =
                tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size, topology, dest_edm_options);

            edm_builders_forward_direction[src_device->id()].reserve(local_link_cores.size());
            edm_builders_backward_direction[dest_device->id()].reserve(local_link_cores.size());
            for (size_t l = 0; l < this->num_links; l++) {
                log_trace(
                    tt::LogOp,
                    "Building forward direction EDM on chip {} on link {}",
                    src_device->id(),
                    edm_builders_forward_direction[src_device->id()].size());
                log_debug(
                    tt::LogOp,
                    "src_device {}, dest_device {}, is_dateline {}",
                    src_device->id(),
                    dest_device->id(),
                    dateline);
                edm_builders_forward_direction[src_device->id()].push_back(
                    tt::tt_fabric::FabricEriscDatamoverBuilder::build(
                        src_device,
                        *src_program,
                        local_link_cores[l],
                        src_device->id(),
                        dest_device->id(),
                        src_curr_edm_config,
                        build_in_worker_connection_mode,
                        dateline));

                log_trace(
                    tt::LogOp,
                    "Building backward direction EDM on chip {} on link {}",
                    dest_device->id(),
                    edm_builders_backward_direction[dest_device->id()].size());
                edm_builders_backward_direction[dest_device->id()].push_back(
                    tt::tt_fabric::FabricEriscDatamoverBuilder::build(
                        dest_device,
                        *dest_program,
                        remote_link_cores[l],
                        dest_device->id(),
                        src_device->id(),
                        dest_curr_edm_config,
                        build_in_worker_connection_mode,
                        dateline));
            }
        };

    tt::tt_fabric::FabricEriscDatamoverBuilder* a_builder = nullptr;
    // Construct the builders
    for (size_t hop = 0; hop < device_sequence.size() - 1; hop++) {
        auto src_device = device_sequence[hop];
        auto dest_device = device_sequence[hop + 1];
        auto src_program = programs[hop];
        auto dest_program = programs[hop + 1];
        build_edm_directions(src_device, dest_device, src_program, dest_program);
        // Move out of loop?
        a_builder = &edm_builders_backward_direction[dest_device->id()].front();
        this->buffer_size_bytes = a_builder->channel_buffer_size;
    }
    if (topology == Topology::Ring) {
        auto src_device = device_sequence.back();
        auto dest_device = device_sequence.front();
        auto src_program = programs.back();
        auto dest_program = programs.front();
        build_edm_directions(src_device, dest_device, src_program, dest_program);

        a_builder = &edm_builders_backward_direction[dest_device->id()].front();
        this->buffer_size_bytes = a_builder->channel_buffer_size;
    }

    if (!build_in_worker_connection_mode) {
        // Establish local connections between EDMs on the same chips to establish the line fabric
        uint32_t start_bidirectional_device_index = 1;
        uint32_t end_bidirectional_device_index = device_sequence.size() - 1;
        if (topology == Topology::Ring) {
            start_bidirectional_device_index = 0;
            end_bidirectional_device_index = device_sequence.size();
        }

        uint32_t fwd_edm_start_index = 0;
        uint32_t fwd_edm_end_index = (topology == Topology::Ring) ? device_sequence.size() : device_sequence.size() - 1;
        uint32_t bwd_edm_start_index = (topology == Topology::Ring) ? 0 : 1;
        uint32_t bwd_edm_end_index = device_sequence.size();

        auto assign_noc_vc =
            [](auto& edm_builders_direction, size_t start_index, size_t end_index, const auto& device_sequence) {
                for (size_t i = start_index; i < end_index; i++) {
                    const size_t num_links = edm_builders_direction.at(device_sequence[i]->id()).size();
                    auto& direction_edm = edm_builders_direction.at(device_sequence[i]->id());

                    for (size_t l = 0; l < num_links; l++) {
                        auto& edm = direction_edm[l];
                        auto edm_noc_vc = edm.config.DEFAULT_NOC_VC + (l % edm.config.NUM_EDM_NOC_VCS);
                        edm.config.edm_noc_vc = edm_noc_vc;
                    }
                }
            };

        // Call assign_noc_vc for both forward and backward directions
        assign_noc_vc(edm_builders_forward_direction, fwd_edm_start_index, fwd_edm_end_index, device_sequence);
        assign_noc_vc(edm_builders_backward_direction, bwd_edm_start_index, bwd_edm_end_index, device_sequence);

        for (size_t i = start_bidirectional_device_index; i < end_bidirectional_device_index; i++) {
            const size_t num_links = edm_builders_forward_direction.at(device_sequence[i]->id()).size();
            auto& forward_direction_edm = edm_builders_forward_direction.at(device_sequence[i]->id());
            auto& backward_direction_edm = edm_builders_backward_direction.at(device_sequence[i]->id());

            for (size_t l = 0; l < num_links; l++) {
                auto& edm_fwd = forward_direction_edm[l];
                auto& edm_bwd = backward_direction_edm[l];
                // currently is_galaxy is only being passed in through the fabric unit test, once we switch to fabric
                // device init, will use proper cluster type to decide which machine it is. For the optimzation on noc
                // selection, we empirically optimize on 3/4 links for linear, and 4 links on ring, as less links caused
                // perf degradation, potentially caused by sw overhead of checking two nocs.
                bool enable_core_placement_opt = false;
                if (is_galaxy) {
                    if (topology == Topology::Ring) {
                        enable_core_placement_opt = (num_links > 3) && (edm_fwd.my_noc_y != edm_bwd.my_noc_y);
                    } else {
                        enable_core_placement_opt = (num_links > 2) && (edm_fwd.my_noc_y != edm_bwd.my_noc_y);
                    }
                }
                if (enable_core_placement_opt) {
                    if (edm_fwd.my_noc_x < edm_bwd.my_noc_x) {
                        log_info(
                            tt::LogOp,
                            "Fabric MeshId {} ChipId {} edm_fwd {} {} is connecting to edm_bwd {} {} on link {}",
                            *(edm_fwd.local_fabric_node_id.mesh_id),
                            edm_fwd.local_fabric_node_id.chip_id,
                            edm_fwd.my_noc_x,
                            edm_fwd.my_noc_y,
                            edm_bwd.my_noc_x,
                            edm_bwd.my_noc_y,
                            l);
                        for (uint32_t i = 0; i < edm_fwd.config.num_receiver_channels; i++) {
                            edm_fwd.config.receiver_channel_forwarding_noc_ids[i] = 0;
                            edm_bwd.config.receiver_channel_forwarding_noc_ids[i] = 1;
                        }
                        for (uint32_t i = 0; i < edm_fwd.config.num_receiver_channels; i++) {
                            edm_fwd.config.receiver_channel_local_write_noc_ids[i] = 1;
                            edm_bwd.config.receiver_channel_local_write_noc_ids[i] = 1;
                        }
                        for (uint32_t i = 0; i < edm_fwd.config.num_sender_channels; i++) {
                            edm_fwd.config.sender_channel_ack_noc_ids[i] = 1;
                            edm_bwd.config.sender_channel_ack_noc_ids[i] = 0;
                        }
                    } else if (edm_fwd.my_noc_x > edm_bwd.my_noc_x) {
                        log_info(
                            tt::LogOp,
                            "Fabric MeshId {} ChipId {} edm_fwd {} {} is connecting to edm_bwd {} {} on link {}",
                            *(edm_fwd.local_fabric_node_id.mesh_id),
                            edm_fwd.local_fabric_node_id.chip_id,
                            edm_fwd.my_noc_x,
                            edm_fwd.my_noc_y,
                            edm_bwd.my_noc_x,
                            edm_bwd.my_noc_y,
                            l);
                        for (uint32_t i = 0; i < edm_fwd.config.num_receiver_channels; i++) {
                            edm_fwd.config.receiver_channel_forwarding_noc_ids[i] = 1;
                            edm_bwd.config.receiver_channel_forwarding_noc_ids[i] = 0;
                        }
                        for (uint32_t i = 0; i < edm_fwd.config.num_receiver_channels; i++) {
                            edm_fwd.config.receiver_channel_local_write_noc_ids[i] = 1;
                            edm_bwd.config.receiver_channel_local_write_noc_ids[i] = 1;
                        }
                        for (uint32_t i = 0; i < edm_fwd.config.num_sender_channels; i++) {
                            edm_fwd.config.sender_channel_ack_noc_ids[i] = 0;
                            edm_bwd.config.sender_channel_ack_noc_ids[i] = 1;
                        }
                    }
                }
            }
        }

        for (size_t i = start_bidirectional_device_index; i < end_bidirectional_device_index; i++) {
            const size_t num_links = edm_builders_forward_direction.at(device_sequence[i]->id()).size();
            auto& forward_direction_edm = edm_builders_forward_direction.at(device_sequence[i]->id());
            auto& backward_direction_edm = edm_builders_backward_direction.at(device_sequence[i]->id());

            for (size_t l = 0; l < num_links; l++) {
                forward_direction_edm.at(l).connect_to_downstream_edm(backward_direction_edm.at(l));
                backward_direction_edm.at(l).connect_to_downstream_edm(forward_direction_edm.at(l));
            }
        }
    }
}

// Invocable per chip if we want to collectively build the fabric by building this separately per chip
// (and implicitly building the fabric that way)
EdmLineFabricOpInterface::EdmLineFabricOpInterface(
    tt::tt_metal::IDevice* local_device,
    std::optional<tt::tt_metal::IDevice*> forward_device,
    std::optional<tt::tt_metal::IDevice*> backward_device,
    tt::tt_metal::Program* program,
    std::optional<size_t> desired_num_links,
    bool build_in_worker_connection_mode,
    Topology topology) :
    device_sequence({local_device}), programs({program}) {
    static constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes +
        sizeof(tt::tt_fabric::PacketHeader);
    const auto config = tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size, topology);

    log_trace(tt::LogOp, "device id={}", local_device->id());
    log_trace(tt::LogOp, "EDM Fabric Factory ctor on device: {}", local_device->id());
    if (forward_device.has_value()) {
        log_trace(tt::LogOp, "\tConnect[FORWARD]: {} -> {}", local_device->id(), forward_device.value()->id());
    }
    if (backward_device.has_value()) {
        log_trace(tt::LogOp, "\tConnect[BACKWARD]: {} -> {}", local_device->id(), backward_device.value()->id());
    }

    // Construct the builders
    std::array<std::pair<tt::tt_metal::IDevice*, std::optional<tt::tt_metal::IDevice*>>, 2> device_pairs = {
        std::pair<tt::tt_metal::IDevice*, std::optional<tt::tt_metal::IDevice*>>{local_device, forward_device},
        std::pair<tt::tt_metal::IDevice*, std::optional<tt::tt_metal::IDevice*>>{local_device, backward_device}};

    static_assert(EdmLineFabricOpInterface::Direction::FORWARD < 2);
    static_assert(EdmLineFabricOpInterface::Direction::BACKWARD < 2);
    std::array<std::unordered_map<size_t, std::vector<tt::tt_fabric::FabricEriscDatamoverBuilder>>*, 2>
        edm_builders_maps;
    edm_builders_maps[EdmLineFabricOpInterface::Direction::FORWARD] = &this->edm_builders_forward_direction;
    edm_builders_maps[EdmLineFabricOpInterface::Direction::BACKWARD] = &this->edm_builders_backward_direction;

    std::optional<size_t> counted_num_links = std::nullopt;
    std::optional<size_t> obtained_channel_buffer_size = std::nullopt;
    const size_t max_num_links = desired_num_links.value_or(std::numeric_limits<std::size_t>::max());
    for (size_t i = 0; i < device_pairs.size(); i++) {
        if (!device_pairs[i].second.has_value()) {
            continue;
        }
        log_trace(
            tt::LogOp,
            "Device {} is connected to {} at index {}",
            local_device->id(),
            device_pairs[i].second.value()->id(),
            i);
        auto& edm_builders = *edm_builders_maps[i];

        tt::tt_metal::IDevice* remote_device = device_pairs[i].second.value();
        const auto connected_sockets = local_device->get_ethernet_sockets(remote_device->id());
        // re-order the connected_sockets based on virtual coords
        auto reordered_connected_sockets = reorder_connected_sockets(local_device, connected_sockets);

        TT_FATAL(edm_builders.size() == 0, "EDM builders already exist for this device");
        edm_builders.clear();
        for (const auto& core : reordered_connected_sockets) {
            if (!local_device->is_active_ethernet_core(core, true)) {
                continue;
            }
            if (edm_builders[local_device->id()].size() >= max_num_links) {
                break;
            }
            log_trace(
                tt::LogOp,
                "DEBUG: build EDM: device: {}, &program: {}: core-logi(x={},y={})",
                local_device->id(),
                (void*)program,
                core.x,
                core.y);
            edm_builders[local_device->id()].push_back(tt::tt_fabric::FabricEriscDatamoverBuilder::build(
                local_device,
                *program,
                core,
                device_pairs[i].first->id(),
                device_pairs[i].second.value()->id(),
                config,
                build_in_worker_connection_mode));
        }
        if (!counted_num_links.has_value()) {
            TT_FATAL(!obtained_channel_buffer_size.has_value(), "No channel buffer size was counted");
            counted_num_links = edm_builders[local_device->id()].size();
            obtained_channel_buffer_size = edm_builders[local_device->id()].front().channel_buffer_size;
        }
    }
    TT_FATAL(counted_num_links.has_value(), "No links were counted");
    this->num_links = counted_num_links.value();

    TT_FATAL(obtained_channel_buffer_size.has_value(), "No channel buffer size was counted");
    this->buffer_size_bytes = obtained_channel_buffer_size.value();

    if (!build_in_worker_connection_mode) {
        // Establish local connections between EDMs on the same chips to establish the line fabric
        if (forward_device.has_value() && backward_device.has_value()) {
            auto& forward_direction_edm = edm_builders_forward_direction.at(local_device->id());
            auto& backward_direction_edm = edm_builders_backward_direction.at(local_device->id());

            for (size_t l = 0; l < this->num_links; l++) {
                forward_direction_edm.at(l).connect_to_downstream_edm(backward_direction_edm.at(l));
                backward_direction_edm.at(l).connect_to_downstream_edm(forward_direction_edm.at(l));
            }
        }
    }
}

tt::tt_fabric::SenderWorkerAdapterSpec EdmLineFabricOpInterface::uniquely_connect_worker(
    tt::tt_metal::IDevice* device, Direction direction) {
    TT_FATAL(
        (direction == FORWARD)
            ? edm_builders_forward_direction.find(device->id()) != edm_builders_forward_direction.end()
            : edm_builders_backward_direction.find(device->id()) != edm_builders_backward_direction.end(),
        "Device {} not found in edm builders",
        device->id());
    auto& edm_builders = (direction == FORWARD) ? edm_builders_forward_direction.at(device->id())
                                                : edm_builders_backward_direction.at(device->id());
    auto& link_count_map =
        (direction == FORWARD) ? next_forward_direction_edm_available : next_backward_direction_edm_available;
    log_trace(tt::LogOp, "EDM conecting in {} direction", direction == FORWARD ? "FORWARD" : "BACKWARD");
    const auto next_link = link_count_map[device->id()];
    link_count_map[device->id()] = (next_link + 1) % edm_builders.size();

    TT_FATAL(edm_builders.size() > 0, "No EDM builders found for device {}", device->id());
    TT_FATAL(
        next_link < edm_builders.size(), "Next link index {} is out of bounds for device {}", next_link, device->id());
    return edm_builders.at(next_link).build_connection_to_worker_channel();
}

EdmLineFabricOpInterface EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
    const std::vector<tt::tt_metal::IDevice*>& device_sequence,
    const std::vector<tt::tt_metal::Program*>& program_sequence,
    std::optional<size_t> desired_num_links,
    Topology topology) {
    return EdmLineFabricOpInterface(device_sequence, program_sequence, desired_num_links, true, topology);
}

EdmLineFabricOpInterface EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
    tt::tt_metal::IDevice* local_device,
    tt::tt_metal::IDevice* forward_device,
    tt::tt_metal::IDevice* backward_device,
    tt::tt_metal::Program* program,
    std::optional<size_t> desired_num_links,
    Topology topology) {
    return EdmLineFabricOpInterface(
        local_device,
        forward_device == nullptr ? std::nullopt : std::optional<tt::tt_metal::IDevice*>(forward_device),
        backward_device == nullptr ? std::nullopt : std::optional<tt::tt_metal::IDevice*>(backward_device),
        program,
        desired_num_links,
        true,
        topology);
}

void EdmLineFabricOpInterface::build_kernels() const {
    auto generate_kernels_in_direction =
        [this](tt::tt_metal::IDevice* device, tt::tt_metal::Program* program, Direction direction) {
            auto& edm_builders =
                direction == FORWARD ? edm_builders_forward_direction : edm_builders_backward_direction;
            if (edm_builders.find(device->id()) != edm_builders.end()) {
                for (auto& edm_builder : edm_builders.at(device->id())) {
                    for (uint32_t risc_id = 0; risc_id < edm_builder.get_configured_risc_count(); risc_id++) {
                        log_trace(
                            tt::LogOp,
                            "Building EDM kernel on device {}, logical-core (y={},x={}), noc_core (y={},x={}), risc_id "
                            "{}",
                            device->id(),
                            edm_builder.my_eth_core_logical.y,
                            edm_builder.my_eth_core_logical.x,
                            device->ethernet_core_from_logical_core(edm_builder.my_eth_core_logical).y,
                            device->ethernet_core_from_logical_core(edm_builder.my_eth_core_logical).x,
                            risc_id);
                        auto local_edm_kernel = ttnn::ccl::generate_edm_kernel(
                            *program,
                            device,
                            edm_builder,
                            edm_builder.my_eth_core_logical,
                            static_cast<tt::tt_metal::DataMovementProcessor>(risc_id),
                            tt::tt_metal::NOC::NOC_0);
                    }
                }
            }
        };

    TT_ASSERT(device_sequence.size() == programs.size());
    for (size_t i = 0; i < device_sequence.size(); i++) {
        tt::tt_metal::Program* program = programs[i];
        tt::tt_metal::IDevice* device = device_sequence[i];
        generate_kernels_in_direction(device, program, Direction::FORWARD);
        generate_kernels_in_direction(device, program, Direction::BACKWARD);
    }
}

std::vector<tt::tt_fabric::edm_termination_info_t>
EdmLineFabricOpInterface::generate_local_chip_fabric_termination_infos(tt::tt_metal::IDevice* device) const {
    auto generate_termination_info =
        [](const tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder) -> tt::tt_fabric::edm_termination_info_t {
        return tt::tt_fabric::edm_termination_info_t{
            0, edm_builder.my_noc_x, edm_builder.my_noc_y, edm_builder.config.termination_signal_address};
    };
    std::vector<tt::tt_fabric::edm_termination_info_t> edm_termination_infos;
    edm_termination_infos.reserve(this->num_links * 2);
    if (edm_builders_backward_direction.find(device->id()) != edm_builders_backward_direction.end()) {
        std::ranges::transform(
            edm_builders_backward_direction.at(device->id()),
            std::back_inserter(edm_termination_infos),
            generate_termination_info);
    }
    if (edm_builders_forward_direction.find(device->id()) != edm_builders_forward_direction.end()) {
        std::ranges::transform(
            edm_builders_forward_direction.at(device->id()),
            std::back_inserter(edm_termination_infos),
            generate_termination_info);
    }
    return edm_termination_infos;
}

std::vector<tt::tt_fabric::edm_termination_info_t>
EdmLineFabricOpInterface::generate_ordered_termination_info_farthest_to_nearest() const {
    static constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes +
        sizeof(tt::tt_fabric::PacketHeader);
    static const auto config = tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size);
    TT_ASSERT(device_sequence.size() > 0);
    const size_t num_hops = device_sequence.size() - 1;
    TT_ASSERT(num_hops > 0);
    std::vector<tt::tt_fabric::edm_termination_info_t> edm_termination_infos;
    edm_termination_infos.reserve(num_hops * 2 * this->num_links);
    for (int i = num_hops - 1; i >= 0; i--) {
        log_trace(tt::LogOp, "Generating termination info for hop {}", i);
        TT_ASSERT(i + 1 != 0);
        TT_ASSERT(i + 1 < device_sequence.size());
        TT_ASSERT(
            edm_builders_backward_direction.find(device_sequence[i + 1]->id()) != edm_builders_backward_direction.end(),
            "Device {} at index {} not found in `edm_builders_backward_direction` but it was expected there",
            i + 1,
            device_sequence[i + 1]->id());
        TT_ASSERT(
            edm_builders_forward_direction.find(device_sequence[i]->id()) != edm_builders_forward_direction.end(),
            "Device {} at index {} not found in `edm_builders_forward_direction` but it was expected there",
            i,
            device_sequence[i]->id());
        auto& farther_edms = edm_builders_backward_direction.at(device_sequence[i + 1]->id());
        auto& nearer_edms = edm_builders_forward_direction.at(device_sequence[i]->id());

        TT_ASSERT(farther_edms.size() <= this->num_links);
        TT_ASSERT(nearer_edms.size() <= this->num_links);
        for (size_t l = 0; l < this->num_links; l++) {
            auto& farther_edm = farther_edms.at(l);
            const std::size_t distance_receiver = i + 1;
            edm_termination_infos.push_back(
                {distance_receiver, farther_edm.my_noc_x, farther_edm.my_noc_y, config.termination_signal_address});
        }
        for (size_t l = 0; l < this->num_links; l++) {
            auto& nearer_edm = nearer_edms.at(l);
            const std::size_t distance_sender = i;
            edm_termination_infos.push_back(
                {distance_sender, nearer_edm.my_noc_x, nearer_edm.my_noc_y, config.termination_signal_address});
        }
    }
    log_trace(tt::LogOp, "Done Generating termination infos");
    return edm_termination_infos;
}

void EdmLineFabricOpInterface::teardown_from_host(tt::tt_fabric::TerminationSignal termination_signal) const {
    for (tt::tt_metal::IDevice* d : this->device_sequence) {
        if (edm_builders_forward_direction.find(d->id()) != edm_builders_forward_direction.end()) {
            for (auto& edm_builder : edm_builders_forward_direction.at(d->id())) {
                edm_builder.teardown_from_host(d, termination_signal);
            }
        }
        if (edm_builders_backward_direction.find(d->id()) != edm_builders_backward_direction.end()) {
            for (auto& edm_builder : edm_builders_backward_direction.at(d->id())) {
                edm_builder.teardown_from_host(d, termination_signal);
            }
        }
    }
}

void EdmLineFabricOpInterface::set_firmware_context_switch_interval(size_t interval) {
    for (auto& edm_builder : edm_builders_forward_direction) {
        for (auto& builder : edm_builder.second) {
            builder.set_firmware_context_switch_interval(interval);
        }
    }
    for (auto& edm_builder : edm_builders_backward_direction) {
        for (auto& builder : edm_builder.second) {
            builder.set_firmware_context_switch_interval(interval);
        }
    }
}

void initialize_edm_fabric(
    distributed::MeshDevice* mesh_device,
    bool wrap_fabric_around_mesh,
    std::optional<size_t> context_switch_interval_override,
    Topology topology) {
    if (wrap_fabric_around_mesh) {
        auto devices = mesh_device->get_view().get_ring_devices();
        std::vector<tt::tt_metal::Program*> program_ptrs;
        std::vector<tt::tt_metal::Program> programs(devices.size());
        program_ptrs.reserve(devices.size());

        std::transform(
            programs.begin(), programs.end(), std::back_inserter(program_ptrs), [](tt::tt_metal::Program& p) {
                return &p;
            });
        EdmLineFabricOpInterface fabric_device_builders =
            EdmLineFabricOpInterface(devices, program_ptrs, std::nullopt, false, topology);
        if (context_switch_interval_override.has_value()) {
            fabric_device_builders.set_firmware_context_switch_interval(context_switch_interval_override.value());
        }
        fabric_device_builders.build_kernels();

        for (size_t i = 0; i < devices.size(); i++) {
            auto* device = devices[i];
            auto* program_ptr = program_ptrs[i];
            tt::tt_metal::detail::CompileProgram(device, *program_ptr);
            tt::tt_metal::EnqueueProgram(device->command_queue(), *program_ptr, false);
        }
    } else {
        std::vector<EdmLineFabricOpInterface> row_fabric_lines;
        row_fabric_lines.reserve(mesh_device->get_view().get_row_views().size());
        std::vector<EdmLineFabricOpInterface> col_fabric_lines;
        col_fabric_lines.reserve(mesh_device->get_view().get_column_views().size());

        size_t num_rows = mesh_device->get_view().get_row_views().size();
        size_t num_cols = mesh_device->get_view().get_column_views().size();
        std::vector<std::vector<tt::tt_metal::Program>> programs(num_rows);
        for (size_t r = 0; r < num_rows; r++) {
            programs[r].resize(num_cols);
        }

        for (size_t i = 0; i < num_rows; i++) {
            std::vector<tt::tt_metal::Program*> program_ptrs;
            program_ptrs.reserve(num_cols);
            std::transform(
                programs[i].begin(), programs[i].end(), std::back_inserter(program_ptrs), [](tt::tt_metal::Program& p) {
                    return &p;
                });
            row_fabric_lines.push_back(EdmLineFabricOpInterface(
                mesh_device->get_view().get_row_views()[i], program_ptrs, std::nullopt, false, topology));
            if (context_switch_interval_override.has_value()) {
                row_fabric_lines.back().set_firmware_context_switch_interval(context_switch_interval_override.value());
            }
        }

        for (size_t i = 0; i < num_cols; i++) {
            std::vector<tt::tt_metal::Program*> program_ptrs;
            program_ptrs.reserve(num_rows);
            for (size_t r = 0; r < num_rows; r++) {
                program_ptrs.push_back(&programs[r][i]);
            }
            col_fabric_lines.push_back(EdmLineFabricOpInterface(
                mesh_device->get_view().get_column_views()[i], program_ptrs, std::nullopt, false, topology));
            if (context_switch_interval_override.has_value()) {
                col_fabric_lines.back().set_firmware_context_switch_interval(context_switch_interval_override.value());
            }
        }

        std::for_each(row_fabric_lines.begin(), row_fabric_lines.end(), [](auto& line) { line.build_kernels(); });
        std::for_each(col_fabric_lines.begin(), col_fabric_lines.end(), [](auto& line) { line.build_kernels(); });

        for (size_t r = 0; r < num_rows; r++) {
            for (size_t c = 0; c < num_cols; c++) {
                log_info(tt::LogOp, "Compile EDM program");
                tt::tt_metal::IDevice* device = mesh_device->get_device(r, c);
                auto& program = programs.at(r).at(c);
                tt::tt_metal::detail::CompileProgram(device, program);
                tt::tt_metal::EnqueueProgram(device->command_queue(), program, false);
            }
        }
    }
}

void teardown_edm_fabric(distributed::MeshDevice* mesh_device, bool wrap_fabric_around_mesh, Topology topology) {
    auto teardown = [topology](const std::vector<IDevice*>& line_view) {
        std::vector<tt::tt_metal::Program> programs(line_view.size());
        std::vector<tt::tt_metal::Program*> program_ptrs;
        program_ptrs.reserve(programs.size());
        std::transform(
            programs.begin(), programs.end(), std::back_inserter(program_ptrs), [](Program& p) { return &p; });
        EdmLineFabricOpInterface edm_fabric(line_view, program_ptrs, std::nullopt, false, topology);
        edm_fabric.teardown_from_host(tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
    };
    if (wrap_fabric_around_mesh) {
        auto devices = mesh_device->get_view().get_ring_devices();
        teardown(devices);
    } else {
        for (const auto& row_view : mesh_device->get_view().get_row_views()) {
            teardown(row_view);
        }
        for (const auto& col_view : mesh_device->get_view().get_column_views()) {
            teardown(col_view);
        }
    }
}

}  // namespace ttnn::ccl
