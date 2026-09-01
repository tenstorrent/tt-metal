// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_generic_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/ccl/common/types/fabric_directions.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <algorithm>
#include <cstddef>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace ttnn::experimental::prim {

namespace {
// The active route-buffer size is specialized into the device kernel. Bound host-generated custom routes by the
// smallest supported Fabric2D tier; the kernel additionally checks its exact compile-time capacity before injection.
// TODO: Replace this conservative bound when an existing Fabric API exposes the active route-buffer capacity.
constexpr uint32_t minimum_fabric2d_route_capacity = 19;

struct CustomFabric2DRoute {
    uint32_t num_commands = 0;
    tt::tt_fabric::eth_chan_directions initial_direction = tt::tt_fabric::eth_chan_directions::COUNT;
    std::vector<uint32_t> packed_commands;
};

uint8_t fabric_direction_command(tt::tt_fabric::eth_chan_directions direction) {
    using MeshRoutingFields = tt::tt_fabric::RoutingFieldsConstants::Mesh;
    switch (direction) {
        case tt::tt_fabric::eth_chan_directions::EAST: return MeshRoutingFields::FORWARD_EAST;
        case tt::tt_fabric::eth_chan_directions::WEST: return MeshRoutingFields::FORWARD_WEST;
        case tt::tt_fabric::eth_chan_directions::NORTH: return MeshRoutingFields::FORWARD_NORTH;
        case tt::tt_fabric::eth_chan_directions::SOUTH: return MeshRoutingFields::FORWARD_SOUTH;
        default:
            TT_THROW("All-to-all custom route does not support fabric direction {}", static_cast<uint32_t>(direction));
    }
}

uint8_t fabric_terminal_command(tt::tt_fabric::eth_chan_directions incoming_direction) {
    switch (incoming_direction) {
        case tt::tt_fabric::eth_chan_directions::EAST:
            return fabric_direction_command(tt::tt_fabric::eth_chan_directions::WEST);
        case tt::tt_fabric::eth_chan_directions::WEST:
            return fabric_direction_command(tt::tt_fabric::eth_chan_directions::EAST);
        case tt::tt_fabric::eth_chan_directions::NORTH:
            return fabric_direction_command(tt::tt_fabric::eth_chan_directions::SOUTH);
        case tt::tt_fabric::eth_chan_directions::SOUTH:
            return fabric_direction_command(tt::tt_fabric::eth_chan_directions::NORTH);
        default:
            TT_THROW(
                "All-to-all custom route does not support fabric direction {}",
                static_cast<uint32_t>(incoming_direction));
    }
}

void set_packed_route_command(CustomFabric2DRoute& route, uint32_t command_index, uint8_t command) {
    constexpr uint32_t commands_per_word = 8;
    TT_FATAL(command_index / commands_per_word < route.packed_commands.size(), "Custom route command is out of range");
    route.packed_commands[command_index / commands_per_word] |= static_cast<uint32_t>(command)
                                                                << ((command_index % commands_per_word) * 4);
}

ttnn::Shape get_tiled_shape(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& shape = input_tensor.padded_shape();
    ttsl::SmallVector<uint32_t> tiled_shape;
    tiled_shape.reserve(shape.rank());
    for (int i = 0; i < shape.rank(); i++) {
        uint32_t dim = 0;
        if (i == shape.rank() - 1) {
            dim = shape[i] / tile_shape[1];
        } else if (i == shape.rank() - 2) {
            dim = shape[i] / tile_shape[0];
        } else {
            dim = shape[i];
        }
        tiled_shape.push_back(dim);
    }
    return ttnn::Shape(tiled_shape);
}

constexpr uint32_t banks_owned_by_link(uint32_t num_dram_banks, uint32_t num_links, uint32_t link) {
    if (link >= num_dram_banks) {
        return 0;
    }
    return 1 + (num_dram_banks - 1 - link) / num_links;
}

static_assert(
    banks_owned_by_link(8, 3, 0) == 3 && banks_owned_by_link(8, 3, 1) == 3 && banks_owned_by_link(8, 3, 2) == 2,
    "Uneven 8-bank/3-link ownership must distribute as 3, 3, 2");

constexpr uint32_t direction_schedule_cores_per_link(uint32_t workers_per_direction, uint32_t num_direction_groups) {
    const uint32_t mux_cores = workers_per_direction > 1 ? num_direction_groups : 0;
    return num_direction_groups * workers_per_direction + mux_cores + 1;  // remote groups, muxes, and local copy
}

constexpr uint32_t num_cardinal_fabric_directions = ttnn::operations::ccl::common::num_fabric_directions;

struct AllToAllStreamSchedule {
    std::vector<std::vector<int32_t>> device_offsets;
    std::vector<std::vector<std::vector<uint32_t>>> block_starts;
    std::vector<std::vector<std::vector<uint32_t>>> block_ends;
    std::vector<std::vector<std::vector<uint32_t>>> block_strides;
    std::vector<std::vector<std::vector<uint32_t>>> completion_flags;
    uint32_t semaphore_sent = 0;
};

AllToAllStreamSchedule build_stream_schedule(
    const AllToAllAsyncGenericParams& operation_attributes,
    const Tensor& input_tensor,
    const MeshCoordinate& mesh_coordinate,
    const tt::tt_fabric::FabricNodeId& sender_device_fabric_node_id,
    ttnn::ccl::Topology effective_topology,
    uint32_t device_index,
    bool is_ring,
    bool is_fabric_2d,
    bool split_antipode_across_arcs,
    bool use_direction_owned_schedule,
    bool use_bank_owned_schedule,
    uint32_t workers_per_direction,
    size_t num_senders_per_link,
    size_t local_sender_stream,
    uint32_t num_direction_groups,
    const std::vector<uint32_t>& direction_group_to_physical_direction,
    const std::array<int32_t, num_cardinal_fabric_directions>& physical_direction_to_group,
    uint32_t num_dram_banks,
    uint32_t blocks_per_core,
    uint32_t num_blocks,
    uint32_t block_stride) {
    AllToAllStreamSchedule schedule{
        .device_offsets = std::vector<std::vector<int32_t>>(num_senders_per_link),
        .block_starts = std::vector<std::vector<std::vector<uint32_t>>>(num_senders_per_link),
        .block_ends = std::vector<std::vector<std::vector<uint32_t>>>(num_senders_per_link),
        .block_strides = std::vector<std::vector<std::vector<uint32_t>>>(num_senders_per_link),
        .completion_flags = std::vector<std::vector<std::vector<uint32_t>>>(num_senders_per_link),
    };
    for (size_t stream = 0; stream < num_senders_per_link; ++stream) {
        schedule.block_starts[stream].resize(operation_attributes.num_links);
        schedule.block_ends[stream].resize(operation_attributes.num_links);
        schedule.block_strides[stream].resize(operation_attributes.num_links);
        schedule.completion_flags[stream].resize(operation_attributes.num_links);
    }

    auto* device = input_tensor.device();
    std::vector<std::vector<uint32_t>> destination_groups(num_senders_per_link);
    std::vector<int32_t> ring_ordered_offsets;
    std::vector<uint32_t> direction_target_indices(num_direction_groups, 0);
    auto direction_group_for_offset = [&](int32_t device_offset) -> uint32_t {
        if (!is_fabric_2d) {
            return device_offset < 0 ? 1 : 0;
        }
        const bool antipodal =
            split_antipode_across_arcs && std::abs(device_offset) * 2 == operation_attributes.num_devices;
        const int32_t route_offset = antipodal ? (device_offset > 0 ? 1 : -1) : device_offset;
        const auto connection_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, mesh_coordinate, route_offset, effective_topology, operation_attributes.cluster_axis);
        TT_FATAL(connection_coord.has_value(), "No all-to-all target at device offset {}", device_offset);
        const auto connection_node = device->get_fabric_node_id(*connection_coord);
        const auto direction =
            tt::tt_fabric::get_eth_forwarding_direction(sender_device_fabric_node_id, connection_node);
        TT_FATAL(
            direction.has_value(),
            "No Fabric2D forwarding direction from all-to-all source {} to target {}",
            sender_device_fabric_node_id,
            connection_node);
        const uint32_t direction_index = static_cast<uint32_t>(*direction);
        TT_FATAL(
            direction_index < physical_direction_to_group.size() && physical_direction_to_group[direction_index] >= 0,
            "Fabric2D direction {} is absent from the collective all-to-all schedule",
            direction_index);
        return static_cast<uint32_t>(physical_direction_to_group[direction_index]);
    };
    auto sender_stream_for_offset = [&](int32_t device_offset) {
        if (device_offset == 0 && num_senders_per_link > 1) {
            return local_sender_stream;
        }
        if (use_direction_owned_schedule) {
            const uint32_t direction_group = direction_group_for_offset(device_offset);
            return static_cast<size_t>(direction_group) * workers_per_direction +
                   (direction_target_indices[direction_group]++ % workers_per_direction);
        }
        return size_t{0};
    };
    if (is_ring) {
        // Visit global destinations in the same order on every source to avoid a cyclic fabric schedule.
        // Use the shortest signed Ring offset; preserve the raw sign for an even-size antipode.
        for (uint32_t target_device = 0; target_device < operation_attributes.num_devices; ++target_device) {
            int32_t device_offset = static_cast<int32_t>(target_device) - static_cast<int32_t>(device_index);
            const int32_t half_ring = static_cast<int32_t>(operation_attributes.num_devices / 2);
            if (device_offset < -half_ring) {
                device_offset += operation_attributes.num_devices;
            } else if (device_offset > half_ring) {
                device_offset -= operation_attributes.num_devices;
            }
            if (use_direction_owned_schedule && use_bank_owned_schedule) {
                ring_ordered_offsets.push_back(device_offset);
            } else {
                const size_t sender_stream = sender_stream_for_offset(device_offset);
                schedule.device_offsets[sender_stream].push_back(device_offset);
                destination_groups[sender_stream].push_back(target_device);
            }
        }
    } else {
        // Keep a common global order on every source, but alternate destinations from opposite ends of the line.
        // This gives the independent positive/negative fabric connections a chance to drain concurrently instead
        // of issuing every destination on one side before turning to the other side.
        for (uint32_t step = 0; step < operation_attributes.num_devices; ++step) {
            const uint32_t target_device = step % 2 == 0 ? step / 2 : operation_attributes.num_devices - 1 - step / 2;
            const int32_t device_offset = static_cast<int32_t>(target_device) - static_cast<int32_t>(device_index);
            const size_t sender_stream = sender_stream_for_offset(device_offset);
            schedule.device_offsets[sender_stream].push_back(device_offset);
            destination_groups[sender_stream].push_back(step / 2);
        }
    }

    std::vector<std::vector<uint32_t>> bank_indices(num_senders_per_link);
    if (use_bank_owned_schedule) {
        const uint32_t max_banks_per_link = banks_owned_by_link(num_dram_banks, operation_attributes.num_links, 0);
        if (is_ring && use_direction_owned_schedule) {
            // A TP4 Ring has only one or two remote destinations per direction. Assigning a whole destination to one
            // worker would leave most mux clients idle, so stripe each destination's DRAM banks across all workers in
            // that direction. Each bank still owns a full contiguous packet stream; opposite directions retain
            // opposite bank order to reduce receiver-side bank contention.
            for (uint32_t target_index = 0; target_index < ring_ordered_offsets.size(); ++target_index) {
                const int32_t device_offset = ring_ordered_offsets[target_index];
                const bool local = device_offset == 0;
                const bool antipodal = !local && split_antipode_across_arcs &&
                                       std::abs(device_offset) * 2 == operation_attributes.num_devices;
                const int32_t half_ring = static_cast<int32_t>(operation_attributes.num_devices / 2);
                for (uint32_t bank_phase = 0; bank_phase < max_banks_per_link; ++bank_phase) {
                    // Split an even Ring's antipodal destination evenly across both arcs. Source parity swaps which
                    // arc owns bank 0 so adjacent sources do not start on the same direction. Other negative routes
                    // retain reverse bank order to avoid matching the positive direction's receiver-bank phase.
                    const int32_t routed_offset =
                        antipodal ? (((bank_phase + device_index) % 2 == 0) ? half_ring : -half_ring) : device_offset;
                    const uint32_t direction_base =
                        local ? 0 : direction_group_for_offset(routed_offset) * workers_per_direction;
                    const uint32_t bank_in_link =
                        !antipodal && routed_offset < 0 ? max_banks_per_link - 1 - bank_phase : bank_phase;
                    const size_t sender_stream =
                        local ? local_sender_stream
                              : direction_base + ((bank_phase + target_index) % workers_per_direction);
                    schedule.device_offsets[sender_stream].push_back(routed_offset);
                    bank_indices[sender_stream].push_back(bank_in_link);
                }
            }
        } else {
            for (uint32_t stream = 0; stream < num_senders_per_link; ++stream) {
                auto& stream_offsets = schedule.device_offsets[stream];
                const auto ordered_targets = stream_offsets;
                const auto& ordered_groups = destination_groups[stream];
                stream_offsets.clear();
                stream_offsets.reserve(ordered_targets.size() * max_banks_per_link);
                bank_indices[stream].reserve(ordered_targets.size() * max_banks_per_link);
                // Alternate the two globally paired destinations after each bank slice. This preserves the common
                // destination order while reducing the interval during which only one fabric direction is issued.
                const uint32_t num_destination_groups =
                    is_ring ? operation_attributes.num_devices : (operation_attributes.num_devices + 1) / 2;
                for (uint32_t group = 0; group < num_destination_groups; ++group) {
                    for (uint32_t bank_phase = 0; bank_phase < max_banks_per_link; ++bank_phase) {
                        bool negative_direction_stream = false;
                        if (use_direction_owned_schedule && stream < local_sender_stream) {
                            const uint32_t direction_group = stream / workers_per_direction;
                            const uint32_t direction = direction_group_to_physical_direction[direction_group];
                            negative_direction_stream = is_fabric_2d
                                                            ? direction == tt::tt_fabric::eth_chan_directions::WEST ||
                                                                  direction == tt::tt_fabric::eth_chan_directions::NORTH
                                                            : direction_group == 1;
                        }
                        // Walk opposite directions in opposite bank orders. This preserves each same-bank contiguous
                        // packet and avoids making both directions contend for the same bank at the same time.
                        const uint32_t bank_in_link =
                            negative_direction_stream ? max_banks_per_link - 1 - bank_phase : bank_phase;
                        for (uint32_t target_index = 0; target_index < ordered_targets.size(); ++target_index) {
                            if (ordered_groups[target_index] == group) {
                                stream_offsets.push_back(ordered_targets[target_index]);
                                bank_indices[stream].push_back(bank_in_link);
                            }
                        }
                    }
                }
            }
        }
    }

    for (uint32_t link = 0; link < operation_attributes.num_links; ++link) {
        uint32_t current_start_block = link * blocks_per_core;
        uint32_t current_end_block = (link + 1) * blocks_per_core;
        if (link == operation_attributes.num_links - 1) {
            current_end_block = num_blocks;
        }
        for (size_t stream = 0; stream < num_senders_per_link; ++stream) {
            for (size_t schedule_index = 0; schedule_index < schedule.device_offsets[stream].size(); ++schedule_index) {
                if (use_bank_owned_schedule) {
                    const uint32_t bank_in_link = bank_indices[stream][schedule_index];
                    const uint32_t bank_start = bank_in_link * operation_attributes.num_links + link;
                    // The final bank phase can be absent on some links when bank count is not divisible by link
                    // count. Keep the common stream schedule and represent that link's missing bank as empty work.
                    const bool link_owns_bank =
                        bank_in_link < banks_owned_by_link(num_dram_banks, operation_attributes.num_links, link);
                    schedule.block_starts[stream][link].push_back(link_owns_bank ? bank_start : num_blocks);
                    schedule.block_ends[stream][link].push_back(num_blocks);
                } else {
                    schedule.block_starts[stream][link].push_back(current_start_block);
                    schedule.block_ends[stream][link].push_back(current_end_block);
                }
                schedule.completion_flags[stream][link].push_back(false);
                schedule.block_strides[stream][link].push_back(block_stride);
            }
        }
    }

    // Direction-owned schedules signal at the end of each contiguous non-empty destination run. Bank-phased
    // schedules can alternate two destinations within a phase and signal once per destination and link. Both rules
    // skip absent bank phases and generic empty work ranges.
    // Each device must send the same number of completion signals that its transpose peer schedule sends back;
    // the final barrier relies on this symmetric per-pair completion-count invariant.
    for (size_t stream = 0; stream < num_senders_per_link; ++stream) {
        for (uint32_t link = 0; link < operation_attributes.num_links; ++link) {
            std::unordered_set<int32_t> completed_offsets;
            bool has_next_nonempty_offset = false;
            int32_t next_nonempty_offset = 0;
            for (size_t schedule_index = schedule.device_offsets[stream].size(); schedule_index-- > 0;) {
                if (schedule.block_starts[stream][link][schedule_index] >=
                    schedule.block_ends[stream][link][schedule_index]) {
                    continue;
                }
                const int32_t device_offset = schedule.device_offsets[stream][schedule_index];
                const bool target_complete = use_bank_owned_schedule && !use_direction_owned_schedule
                                                 ? completed_offsets.insert(device_offset).second
                                                 : (!has_next_nonempty_offset || device_offset != next_nonempty_offset);
                if (target_complete) {
                    schedule.completion_flags[stream][link][schedule_index] = true;
                    ++schedule.semaphore_sent;
                }
                has_next_nonempty_offset = true;
                next_nonempty_offset = device_offset;
            }
        }
    }
    return schedule;
}

}  // namespace

AllToAllAsyncGenericProgram::cached_mesh_workload_t AllToAllAsyncGenericProgram::create_mesh_workload(
    const AllToAllAsyncGenericParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllToAllAsyncGenericInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto sub_device_id = operation_attributes.sub_device_id;
    auto subdevice = sub_device_id.has_value() ? *sub_device_id : mesh_device->get_sub_device_ids().at(0);
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice);
    auto subdevices = {subdevice};

    auto init_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    auto final_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args,
            tensor_return_value,
            init_barrier_semaphore,
            final_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<AllToAllAsyncGenericProgram::shared_variables_t>
AllToAllAsyncGenericProgram::create_at(
    const AllToAllAsyncGenericParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const AllToAllAsyncGenericInputs& tensor_args,
    Tensor& tensor_return_value,
    const tt::tt_metal::GlobalSemaphore& init_barrier_semaphore,
    const tt::tt_metal::GlobalSemaphore& final_barrier_semaphore) {
    log_debug(tt::LogOp, "DEBUG: create_at is called");

    uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, operation_attributes.cluster_axis);
    const uint32_t cluster_axis = operation_attributes.cluster_axis.value_or(0);
    const auto sender_device_fabric_node_id = tensor_args.input_tensor.device()->get_fabric_node_id(mesh_coordinate);

    const auto fabric_config = tt::tt_fabric::GetFabricConfig();
    const bool is_fabric_2d = tt::tt_fabric::is_2d_fabric_config(fabric_config);
    const uint32_t fabric2d_route_capacity = is_fabric_2d ? minimum_fabric2d_route_capacity : 0;
    // FABRIC_2D_TORUS_X/Y wraps only one mesh axis even though get_fabric_topology() reports Torus for both.
    // Resolve wrapping for the collective's axis so a Ring on the other axis cannot open a nonexistent hop.
    const bool fabric_has_wrap_links = operation_attributes.axis_topology == tt::tt_fabric::Topology::Ring;
    const bool operation_uses_wrap_links =
        operation_attributes.topology == ttnn::ccl::Topology::Ring && fabric_has_wrap_links;
    const auto effective_topology =
        operation_attributes.topology == ttnn::ccl::Topology::Ring && !operation_uses_wrap_links
            ? ttnn::ccl::Topology::Linear
            : operation_attributes.topology;
    const auto connection_topology =
        operation_uses_wrap_links ? tt::tt_fabric::Topology::Ring : tt::tt_fabric::Topology::Linear;
    const std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, 1, effective_topology, operation_attributes.cluster_axis);
    const std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, -1, effective_topology, operation_attributes.cluster_axis);
    const auto fabric_directions = ttnn::operations::ccl::common::get_neighbors(
                                       tensor_args.input_tensor.device()->get_view(),
                                       mesh_coordinate,
                                       connection_topology,
                                       operation_attributes.cluster_axis)
                                       .second;

    TT_FATAL(device_index < operation_attributes.num_devices, "DEBUG: device_index: {}", device_index);

    tt::tt_metal::Program program{};
    MeshDevice* device = tensor_args.input_tensor.device();
    // Mesh workloads build programs for global coordinates and discard remote programs at dispatch. Inspect the view's
    // MaybeRemote entry rather than calling the deprecated physical-device APIs, which throw for remote coordinates.
    // A remote coordinate only needs a representative device to complete construction; its translated NOC coordinates
    // are never submitted by this rank. Local coordinates use their own chip for harvesting-aware NOC translation.
    const auto coordinate_index = mesh_coordinate.to_linear_index(device->shape());
    const auto& maybe_coordinate_device = *(device->get_view().begin() + static_cast<std::ptrdiff_t>(coordinate_index));
    tt::tt_metal::IDevice* coordinate_device = maybe_coordinate_device.when(
        [](tt::tt_metal::IDevice* local_device) { return local_device; },
        [device]() { return static_cast<tt::tt_metal::IDevice*>(device); });

    std::vector<Tensor> input_tensors = {tensor_args.input_tensor};
    std::vector<Tensor> output_tensors = {tensor_return_value};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, effective_topology);

    const auto input_shape = get_tiled_shape(tensor_args.input_tensor);
    uint32_t src_in_dims = 1;
    for (uint32_t i = operation_attributes.out_dim + 1; i < input_shape.size(); ++i) {
        src_in_dims *= input_shape[i];
    }
    const auto output_shape = get_tiled_shape(tensor_return_value);
    uint32_t dst_out_dims = 1;
    uint32_t dst_in_dims = 1;
    for (uint32_t i = 0; i < operation_attributes.in_dim; ++i) {
        dst_out_dims *= output_shape[i];
    }
    const uint32_t reader_has_extra_half_tile =
        operation_attributes.out_dim == input_shape.size() - 2 &&
        tensor_return_value.logical_shape()[operation_attributes.out_dim] % 32 == 16;
    const uint32_t writer_has_extra_half_tile =
        operation_attributes.in_dim == input_shape.size() - 2 &&
        tensor_args.input_tensor.logical_shape()[operation_attributes.in_dim] % 32 == 16;
    for (uint32_t i = operation_attributes.in_dim + 1; i < output_shape.size(); ++i) {
        dst_in_dims *= output_shape[i];
    }
    const uint32_t concat_num_half_tiles =
        output_shape[operation_attributes.in_dim] * 2 / operation_attributes.num_devices;
    const uint32_t concat_num_tiles = (concat_num_half_tiles + 1) / 2;
    const uint32_t num_blocks = dst_out_dims * dst_in_dims * concat_num_tiles;

    const bool is_ring = effective_topology == ttnn::ccl::Topology::Ring;
    const auto sub_device_id = operation_attributes.sub_device_id.value_or(device->get_sub_device_ids().at(0));
    const auto available_worker_cores =
        device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id).num_cores();
    const uint32_t page_size = op_config.get_page_size();
    const uint32_t max_payload_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    const uint32_t payload_capacity_pages = max_payload_size / page_size;
    const uint32_t num_dram_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    // With more than four pages, assign each stream whole destination DRAM banks. Output pages separated by the
    // bank count are physically contiguous within a bank, so the receiver can coalesce a full fabric payload even
    // though a scatter header can encode only four independent destination runs. Source pages are individually
    // address-generated, so this schedule is equally valid for DRAM and L1 inputs; that also lets the L1-input
    // perf proxy isolate reader DRAM work without silently switching to a different worker/fabric program. Keep
    // the generic scatter schedule for small payloads and non-DRAM outputs. Link l owns banks
    // l, l + num_links, ...; this remains valid when bank count is not divisible by link count.
    const bool use_bank_owned_schedule =
        payload_capacity_pages > 4 && tensor_return_value.buffer()->buffer_type() == BufferType::DRAM &&
        tensor_return_value.buffer()->buffer_layout() == TensorMemoryLayout::INTERLEAVED && num_dram_banks > 0 &&
        num_blocks >= num_dram_banks;
    const uint32_t number_pages_per_packet =
        use_bank_owned_schedule ? payload_capacity_pages : std::min<uint32_t>(4, payload_capacity_pages);
    const uint32_t block_stride = use_bank_owned_schedule ? num_dram_banks : 1;

    // Direction ownership, worker parallelism, and bank ownership are independent choices. Large messages use up to
    // three workers per physical egress; two or one are selected when a restricted subdevice cannot fit the preferred
    // schedule. Multiple workers sharing an egress use one mux, while a one-worker egress connects directly.
    constexpr uint32_t preferred_workers_per_direction = 3;
    constexpr uint32_t logical_num_directions = 2;

    // Build one collective-wide, compact ordering of the physical first-hop directions used by Fabric2D. Every chip
    // uses the same stream-group indices, even when a direction is inactive on that chip, so transpose peers retain
    // symmetric completion counts. A folded logical axis can use three or four physical egresses across the collective;
    // grouping by the actual first hop lets those paths retain the parallel mux schedule instead of falling back to the
    // serial routed schedule. Fabric1D uses fixed positive/negative logical groups.
    std::array<bool, num_cardinal_fabric_directions> collective_physical_directions{};
    if (is_fabric_2d) {
        auto record_physical_direction = [&](const MeshCoordinate& source_coord, int32_t offset) {
            const auto source_node = device->get_fabric_node_id(source_coord);
            const auto connection_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
                tensor_args.input_tensor, source_coord, offset, effective_topology, operation_attributes.cluster_axis);
            TT_FATAL(connection_coord.has_value(), "No all-to-all target at device offset {}", offset);
            const auto connection_node = device->get_fabric_node_id(*connection_coord);
            const auto direction = tt::tt_fabric::get_eth_forwarding_direction(source_node, connection_node);
            TT_FATAL(
                direction.has_value(),
                "No Fabric2D forwarding direction from all-to-all source {} to target {}",
                source_node,
                connection_node);
            const uint32_t direction_index = static_cast<uint32_t>(*direction);
            TT_FATAL(
                direction_index < num_cardinal_fabric_directions,
                "All-to-all does not support Fabric2D egress direction {}",
                direction_index);
            collective_physical_directions[direction_index] = true;
        };

        for (uint32_t source_device = 0; source_device < operation_attributes.num_devices; ++source_device) {
            MeshCoordinate source_coord = mesh_coordinate;
            source_coord[cluster_axis] = source_device;
            for (uint32_t target_device = 0; target_device < operation_attributes.num_devices; ++target_device) {
                int32_t device_offset = static_cast<int32_t>(target_device) - static_cast<int32_t>(source_device);
                if (is_ring) {
                    const int32_t half_ring = static_cast<int32_t>(operation_attributes.num_devices / 2);
                    if (device_offset < -half_ring) {
                        device_offset += operation_attributes.num_devices;
                    } else if (device_offset > half_ring) {
                        device_offset -= operation_attributes.num_devices;
                    }
                }
                if (device_offset == 0) {
                    continue;
                }
                record_physical_direction(source_coord, device_offset);
            }
        }
    }

    std::vector<uint32_t> direction_group_to_physical_direction;
    std::array<int32_t, num_cardinal_fabric_directions> physical_direction_to_group{};
    physical_direction_to_group.fill(-1);
    if (is_fabric_2d) {
        for (uint32_t direction = 0; direction < num_cardinal_fabric_directions; ++direction) {
            if (collective_physical_directions[direction]) {
                physical_direction_to_group[direction] =
                    static_cast<int32_t>(direction_group_to_physical_direction.size());
                direction_group_to_physical_direction.push_back(direction);
            }
        }
    } else {
        direction_group_to_physical_direction = {0, 1};
    }
    const uint32_t num_direction_groups = static_cast<uint32_t>(direction_group_to_physical_direction.size());
    TT_FATAL(num_direction_groups > 0, "All-to-all collective has no remote direction groups");
    auto get_direct_ring_hop_directions =
        [&](const MeshCoordinate& source_coord,
            int32_t device_offset) -> std::optional<std::vector<tt::tt_fabric::eth_chan_directions>> {
        const uint32_t num_hops = std::abs(device_offset);
        std::vector<tt::tt_fabric::eth_chan_directions> hop_directions;
        hop_directions.reserve(num_hops);
        auto previous_coord = source_coord;
        for (uint32_t hop = 1; hop <= num_hops; ++hop) {
            const int32_t hop_offset = device_offset > 0 ? static_cast<int32_t>(hop) : -static_cast<int32_t>(hop);
            const auto next_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
                tensor_args.input_tensor,
                source_coord,
                hop_offset,
                effective_topology,
                operation_attributes.cluster_axis);
            TT_FATAL(next_coord.has_value(), "No ring coordinate at offset {}", hop_offset);
            const auto previous_node = device->get_fabric_node_id(previous_coord);
            const auto next_node = device->get_fabric_node_id(*next_coord);
            // Host boundaries do not affect this proof because every rank has the global physical direction map.
            // Inter-mesh routers can replace a custom route at the boundary, so retain canonical routing for those
            // arcs.
            if (previous_node.mesh_id != next_node.mesh_id) {
                return std::nullopt;
            }
            const auto directions = tt::tt_fabric::get_neighbor_eth_directions(previous_node, next_node);
            if (directions.empty()) {
                return std::nullopt;
            }
            hop_directions.push_back(directions.front());
            previous_coord = *next_coord;
        }
        return hop_directions;
    };

    auto custom_ring_route_is_representable = [&](const std::optional<std::vector<tt::tt_fabric::eth_chan_directions>>&
                                                      maybe_hop_directions) {
        if (!maybe_hop_directions.has_value() || maybe_hop_directions->empty() ||
            maybe_hop_directions->size() > fabric2d_route_capacity) {
            return false;
        }
        const auto& hop_directions = *maybe_hop_directions;
        auto is_spine_direction = [](tt::tt_fabric::eth_chan_directions direction) {
            return direction == tt::tt_fabric::eth_chan_directions::NORTH ||
                   direction == tt::tt_fabric::eth_chan_directions::SOUTH;
        };
        auto is_branch_direction = [](tt::tt_fabric::eth_chan_directions direction) {
            return direction == tt::tt_fabric::eth_chan_directions::EAST ||
                   direction == tt::tt_fabric::eth_chan_directions::WEST;
        };
        // The mesh header has only one branch offset per E/W direction. A custom route with zeroed branch offsets is
        // therefore safe only when a spine router never hands the packet to an E/W branch router. Canonical routing
        // remains the correctness fallback for folded arcs that need that transition (including multi-turn snakes).
        for (uint32_t hop = 1; hop < hop_directions.size(); ++hop) {
            if (is_spine_direction(hop_directions[hop - 1]) && is_branch_direction(hop_directions[hop])) {
                return false;
            }
        }
        return true;
    };

    bool split_antipode_across_arcs = is_ring && !is_fabric_2d;
    if (is_ring && is_fabric_2d && operation_attributes.num_devices > 2 && operation_attributes.num_devices % 2 == 0) {
        const int32_t half_ring = static_cast<int32_t>(operation_attributes.num_devices / 2);
        split_antipode_across_arcs = true;
        for (uint32_t source_device = 0; source_device < operation_attributes.num_devices; ++source_device) {
            MeshCoordinate source_coord = mesh_coordinate;
            source_coord[cluster_axis] = source_device;
            const auto positive_hops = get_direct_ring_hop_directions(source_coord, half_ring);
            const auto negative_hops = get_direct_ring_hop_directions(source_coord, -half_ring);
            const bool source_can_split = custom_ring_route_is_representable(positive_hops) &&
                                          custom_ring_route_is_representable(negative_hops) &&
                                          positive_hops->front() != negative_hops->front();
            if (!source_can_split) {
                split_antipode_across_arcs = false;
                break;
            }
        }
    }

    auto build_custom_ring_route = [&](int32_t device_offset) {
        CustomFabric2DRoute route;
        const auto maybe_hop_directions = get_direct_ring_hop_directions(mesh_coordinate, device_offset);
        TT_FATAL(maybe_hop_directions.has_value(), "All-to-all custom Fabric2D route requires direct ring edges");
        const auto& hop_directions = *maybe_hop_directions;
        const uint32_t num_hops = hop_directions.size();
        TT_FATAL(
            custom_ring_route_is_representable(maybe_hop_directions),
            "All-to-all custom Fabric2D route with {} hops is not representable by the active {}-command header",
            num_hops,
            fabric2d_route_capacity);
        route.num_commands = num_hops;
        route.packed_commands.resize((num_hops + 7) / 8, 0);
        route.initial_direction = hop_directions.front();

        // Injection selects hop 0's egress. Each intermediate router consumes the next hop's direction, and the
        // destination consumes the direction opposite its ingress to drain locally.
        for (uint32_t command = 0; command + 1 < num_hops; ++command) {
            set_packed_route_command(route, command, fabric_direction_command(hop_directions[command + 1]));
        }
        set_packed_route_command(route, num_hops - 1, fabric_terminal_command(hop_directions.back()));
        return route;
    };

    const uint32_t max_useful_workers_per_direction =
        is_ring ? preferred_workers_per_direction
                : std::min(preferred_workers_per_direction, operation_attributes.num_devices - 1);
    TT_FATAL(max_useful_workers_per_direction > 0, "Direction-owned all-to-all requires at least two devices");
    // Normalize the message into full-payload packet equivalents per direction-worker lane. A lane is the pair of
    // same-index positive/negative workers: destination placement decides which member receives the work, while the
    // total message determines whether adding that lane can amortize its worker and mux startup. Linear requires 16
    // packet equivalents per lane. Ring requires four because the compact schedule serializes both physical directions
    // through one remote stream. Unlike a byte cutoff, this scales with fabric payload capacity, tensor page size,
    // link count, and the number of useful worker lanes.
    constexpr uint32_t linear_min_packets_per_lane = 16;
    constexpr uint32_t ring_min_packets_per_lane = 4;
    const uint32_t min_packets_per_lane = is_ring ? ring_min_packets_per_lane : linear_min_packets_per_lane;
    const uint64_t packets_per_lane =
        tensor_return_value.buffer()->num_pages() / (static_cast<uint64_t>(operation_attributes.num_links) *
                                                     max_useful_workers_per_direction * number_pages_per_packet);
    // The thresholds apply to two direction groups. Folded axes can require three or four physical egresses, each with
    // its own workers and mux, so scale the startup cost proportionally.
    const bool parallel_workers_are_worthwhile =
        packets_per_lane * logical_num_directions >= static_cast<uint64_t>(min_packets_per_lane) * num_direction_groups;
    uint32_t workers_per_direction = 0;
    if (parallel_workers_are_worthwhile) {
        // Qualify the preferred three-worker schedule before adapting to core capacity. Two workers are a
        // restricted-subdevice fallback for large messages, not a lower size tier.
        for (uint32_t candidate = max_useful_workers_per_direction; candidate > 0; --candidate) {
            if (direction_schedule_cores_per_link(candidate, num_direction_groups) * operation_attributes.num_links <=
                available_worker_cores) {
                workers_per_direction = candidate;
                break;
            }
        }
    }
    // A single direct worker per direction is safe for generic scatter order. Bank-owned batches can cyclically block
    // independent direct directions, so use the compact schedule when a restricted subdevice cannot fit a mux tier.
    // Fabric2D antipodes are safe once streams are grouped by their concrete first hop.
    if (workers_per_direction == 1 && use_bank_owned_schedule) {
        workers_per_direction = 0;
    }
    const bool use_direction_owned_schedule = workers_per_direction > 0;
    // Explicit antipode routes belong to the direction-owned schedule. The compact schedule uses canonical Fabric2D
    // routing, which also keeps its per-target runtime record below the Tensix RTA limit.
    if (is_fabric_2d && !use_direction_owned_schedule) {
        split_antipode_across_arcs = false;
    }
    const uint32_t custom_fabric2d_route_words =
        is_fabric_2d && split_antipode_across_arcs ? (operation_attributes.num_devices / 2 + 7) / 8 : 0;
    const bool use_worker_mux = workers_per_direction > 1;
    const uint32_t mux_cores_per_direction = workers_per_direction + 1;
    const uint32_t direction_senders_per_link = num_direction_groups * workers_per_direction + 1;
    const uint32_t direction_total_cores_per_link =
        direction_schedule_cores_per_link(workers_per_direction, num_direction_groups);
    const size_t num_senders_per_link = use_direction_owned_schedule
                                            ? direction_senders_per_link
                                            : (available_worker_cores >= 2 * operation_attributes.num_links ? 2 : 1);
    const uint32_t total_cores_per_link =
        use_direction_owned_schedule ? direction_total_cores_per_link : num_senders_per_link;
    const auto [all_worker_core_range, all_worker_cores] = ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links, total_cores_per_link, device, operation_attributes.sub_device_id);
    (void)all_worker_core_range;
    TT_FATAL(
        all_worker_cores.size() == static_cast<size_t>(operation_attributes.num_links) * total_cores_per_link,
        "All-to-all needs {} worker cores ({} links x {} cores/link), but only {} were selected",
        operation_attributes.num_links * total_cores_per_link,
        operation_attributes.num_links,
        total_cores_per_link,
        all_worker_cores.size());

    std::vector<CoreCoord> sender_worker_cores;
    sender_worker_cores.reserve(operation_attributes.num_links * num_senders_per_link);
    std::vector<std::vector<CoreCoord>> mux_cores(
        use_worker_mux ? operation_attributes.num_links : 0,
        std::vector<CoreCoord>(use_worker_mux ? num_direction_groups : 0));
    std::set<CoreRange> sender_worker_core_set;
    for (uint32_t link = 0; link < operation_attributes.num_links; ++link) {
        const uint32_t link_base = link * total_cores_per_link;
        if (use_worker_mux) {
            for (uint32_t direction_group = 0; direction_group < num_direction_groups; ++direction_group) {
                const uint32_t direction_base = link_base + direction_group * mux_cores_per_direction;
                mux_cores[link][direction_group] = all_worker_cores[direction_base];
                for (uint32_t worker = 0; worker < workers_per_direction; ++worker) {
                    sender_worker_cores.push_back(all_worker_cores[direction_base + 1 + worker]);
                }
            }
            sender_worker_cores.push_back(all_worker_cores[link_base + direction_total_cores_per_link - 1]);
        } else if (use_direction_owned_schedule) {
            for (uint32_t worker = 0; worker < num_direction_groups * workers_per_direction; ++worker) {
                sender_worker_cores.push_back(all_worker_cores[link_base + worker]);
            }
            sender_worker_cores.push_back(all_worker_cores[link_base + direction_total_cores_per_link - 1]);
        } else {
            for (uint32_t stream = 0; stream < num_senders_per_link; ++stream) {
                sender_worker_cores.push_back(all_worker_cores[link_base + stream]);
            }
        }
        for (uint32_t stream = 0; stream < num_senders_per_link; ++stream) {
            sender_worker_core_set.emplace(sender_worker_cores[link * num_senders_per_link + stream]);
        }
    }
    const CoreRangeSet sender_worker_core_range(sender_worker_core_set);

    // Create CB
    // Three packet slots provide reader/writer overlap without coupling pipeline depth to payload/page geometry.
    // In particular, a one-page payload still needs multiple slots to avoid fully serializing both kernels.
    constexpr uint32_t cb_depth = 3;
    const uint32_t cb_size = cb_depth * number_pages_per_packet * page_size;
    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());

    auto cb_src0_config = tt::tt_metal::CircularBufferConfig(cb_size, {{tt::CB::c_in0, data_format}})
                              .set_page_size(tt::CB::c_in0, number_pages_per_packet * page_size);

    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // Create CB for fabric
    const auto reserved_packet_header_CB_index = tt::CB::c_in4;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    const uint32_t num_packet_headers_storable = 4;
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    const uint32_t num_cores_per_blocks = operation_attributes.num_links;
    const uint32_t blocks_per_core = num_blocks / num_cores_per_blocks;
    const size_t local_sender_stream = num_senders_per_link - 1;

    auto sender_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    sender_reader_kernel_config.compile_args = {
        tt::CB::c_in0,                              // cb0_id
        page_size,                                  // tensor0_page_size
        device_index,                               // device_index
        operation_attributes.num_devices,           // num_devices
        input_shape[operation_attributes.out_dim],  // split_dim_size
        src_in_dims,                                // inner_dims_size
        input_shape[input_shape.size() - 1],        // last_dim_sizes
        reader_has_extra_half_tile,                 // has_reader_tail
        writer_has_extra_half_tile,                 // has_writer_tail
        concat_num_tiles,                           // concat_num_tiles
        dst_in_dims,                                // dst_inner_dims_size
        number_pages_per_packet                     // max_pages_per_packet
    };

    tt::tt_metal::TensorAccessorArgs(tensor_args.input_tensor.buffer())
        .append_to(sender_reader_kernel_config.compile_args);

    auto sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/"
        "all_to_all_sender_reader.cpp",
        sender_worker_core_range,
        sender_reader_kernel_config);

    const auto stream_schedule = build_stream_schedule(
        operation_attributes,
        tensor_args.input_tensor,
        mesh_coordinate,
        sender_device_fabric_node_id,
        effective_topology,
        device_index,
        is_ring,
        is_fabric_2d,
        split_antipode_across_arcs,
        use_direction_owned_schedule,
        use_bank_owned_schedule,
        workers_per_direction,
        num_senders_per_link,
        local_sender_stream,
        num_direction_groups,
        direction_group_to_physical_direction,
        physical_direction_to_group,
        num_dram_banks,
        blocks_per_core,
        num_blocks,
        block_stride);
    const auto& device_offsets = stream_schedule.device_offsets;
    const auto& block_starts = stream_schedule.block_starts;
    const auto& block_ends = stream_schedule.block_ends;
    const auto& block_strides = stream_schedule.block_strides;
    const auto& completion_flags = stream_schedule.completion_flags;
    const uint32_t semaphore_sent = stream_schedule.semaphore_sent;
    std::vector<CoreRangeSet> sender_stream_core_ranges(num_senders_per_link);
    for (uint32_t core_id = 0; core_id < sender_worker_cores.size(); ++core_id) {
        const auto& core = sender_worker_cores[core_id];
        const size_t stream = core_id % num_senders_per_link;
        sender_stream_core_ranges[stream] =
            sender_stream_core_ranges[stream].merge(CoreRangeSet(CoreRange(core, core)));
    }

    const uint32_t all_fabric_direction_mask =
        ttnn::operations::ccl::common::fabric_directions_to_mask(fabric_directions);
    std::vector<uint32_t> sender_stream_direction_masks(num_senders_per_link, 0);
    if (use_direction_owned_schedule) {
        // Direction indices follow {East, West, North, South}. Axis 1 is horizontal and axis 0 is vertical.
        // A 1D connection only needs distinct non-zero masks; its low-latency header routes by signed hop count.
        for (uint32_t direction_group = 0; direction_group < num_direction_groups; ++direction_group) {
            const uint32_t direction = direction_group_to_physical_direction[direction_group];
            bool has_connection = direction_group == 0 ? forward_coord.has_value() : backward_coord.has_value();
            if (is_fabric_2d) {
                has_connection = false;
                for (uint32_t worker = 0; worker < workers_per_direction && !has_connection; ++worker) {
                    const auto& offsets = device_offsets[direction_group * workers_per_direction + worker];
                    has_connection = std::any_of(
                        offsets.begin(), offsets.end(), [](int32_t device_offset) { return device_offset != 0; });
                }
            }
            if (!has_connection) {
                continue;
            }
            for (uint32_t worker = 0; worker < workers_per_direction; ++worker) {
                sender_stream_direction_masks[direction_group * workers_per_direction + worker] = 1U << direction;
            }
        }
    } else {
        sender_stream_direction_masks[0] = all_fabric_direction_mask;
    }

    constexpr uint8_t num_mux_buffers_per_channel = 2;
    const uint32_t mux_config_clients = std::max(1u, workers_per_direction);
    tt::tt_fabric::FabricMuxV2Config mux_config(
        /*num_channels=*/static_cast<uint8_t>(mux_config_clients),
        /*num_buffers_per_channel=*/num_mux_buffers_per_channel,
        /*channel_buffer_size_bytes=*/tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(),
        /*base_l1_address=*/device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    if (use_worker_mux) {
        TT_FATAL(
            mux_config.get_memory_map_end_address() <= device->l1_size_per_core(),
            "All-to-all fabric mux requires L1 through address {:#x}, but each Tensix core has only {:#x} bytes",
            mux_config.get_memory_map_end_address(),
            device->l1_size_per_core());
    }

    struct Fabric2DConnectionInfo {
        std::optional<tt::tt_fabric::FabricNodeId> representative_node;
        std::vector<uint32_t> common_links;
    };
    auto add_fabric2d_connection_target = [&](std::array<Fabric2DConnectionInfo, num_cardinal_fabric_directions>& infos,
                                              int32_t device_offset) {
        if (device_offset == 0) {
            return;
        }
        const bool antipodal =
            split_antipode_across_arcs && std::abs(device_offset) * 2 == operation_attributes.num_devices;
        const int32_t route_offset = antipodal ? (device_offset > 0 ? 1 : -1) : device_offset;
        const auto connection_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor,
            mesh_coordinate,
            route_offset,
            effective_topology,
            operation_attributes.cluster_axis);
        TT_FATAL(connection_coord.has_value(), "No all-to-all target at device offset {}", device_offset);

        const auto connection_node = device->get_fabric_node_id(*connection_coord);
        const auto direction =
            tt::tt_fabric::get_eth_forwarding_direction(sender_device_fabric_node_id, connection_node);
        TT_FATAL(
            direction.has_value(),
            "No Fabric2D forwarding direction from all-to-all source {} to target {}",
            sender_device_fabric_node_id,
            connection_node);
        auto& info = infos[static_cast<uint32_t>(*direction)];
        const auto valid_links =
            tt::tt_fabric::get_forwarding_link_indices(sender_device_fabric_node_id, connection_node);
        TT_FATAL(
            !valid_links.empty(),
            "No Fabric2D forwarding links from all-to-all source {} to target {}",
            sender_device_fabric_node_id,
            connection_node);
        if (!info.representative_node.has_value()) {
            info.representative_node = connection_node;
            info.common_links = valid_links;
            return;
        }
        std::erase_if(info.common_links, [&](uint32_t link) {
            return std::find(valid_links.begin(), valid_links.end(), link) == valid_links.end();
        });
        TT_FATAL(
            !info.common_links.empty(),
            "Fabric2D all-to-all targets in direction {} have no common forwarding link",
            static_cast<uint32_t>(*direction));
    };

    if (use_worker_mux) {
        for (uint32_t link = 0; link < operation_attributes.num_links; ++link) {
            for (uint32_t direction_group = 0; direction_group < num_direction_groups; ++direction_group) {
                const uint32_t representative_stream = direction_group * workers_per_direction;
                if (sender_stream_direction_masks[representative_stream] == 0) {
                    continue;
                }
                const uint32_t physical_direction = direction_group_to_physical_direction[direction_group];
                std::optional<tt::tt_fabric::FabricNodeId> connection_node;
                std::vector<uint32_t> common_links;
                if (is_fabric_2d) {
                    std::array<Fabric2DConnectionInfo, num_cardinal_fabric_directions> connection_infos;
                    for (uint32_t worker = 0; worker < workers_per_direction; ++worker) {
                        const auto& offsets = device_offsets[direction_group * workers_per_direction + worker];
                        for (const int32_t device_offset : offsets) {
                            add_fabric2d_connection_target(connection_infos, device_offset);
                        }
                    }
                    connection_node = connection_infos[physical_direction].representative_node;
                    common_links = std::move(connection_infos[physical_direction].common_links);
                } else {
                    const auto& neighbor_coord = direction_group == 0 ? forward_coord : backward_coord;
                    if (neighbor_coord.has_value()) {
                        connection_node = device->get_fabric_node_id(*neighbor_coord);
                    }
                }
                TT_FATAL(connection_node.has_value(), "Active all-to-all mux direction has no connection node");
                uint32_t connection_link = link;
                // Fabric1D wrap links are not represented in the routing table, so use the configured link index.
                if (is_fabric_2d) {
                    TT_FATAL(
                        link < common_links.size(),
                        "All-to-all link {} is unavailable for Fabric2D mux direction {} ({} link(s) available)",
                        link,
                        physical_direction,
                        common_links.size());
                    connection_link = common_links[link];
                }
                tt::tt_fabric::add_fabric_mux_v2_to_program(
                    program,
                    mux_config,
                    mux_cores[link][direction_group],
                    sender_device_fabric_node_id,
                    *connection_node,
                    connection_link);
            }
        }
    }

    const CoreCoord drain_sync_logical_core = sender_worker_cores[0];
    const auto drain_sync_core = is_fabric_2d
                                     ? coordinate_device->worker_core_from_logical_core(drain_sync_logical_core)
                                     : device->worker_core_from_logical_core(drain_sync_logical_core);
    size_t drain_candidate_index = 0;
    if (is_fabric_2d) {
        const auto candidate_it = std::find(
            operation_attributes.drain_logical_core_candidates.begin(),
            operation_attributes.drain_logical_core_candidates.end(),
            drain_sync_logical_core);
        TT_FATAL(
            candidate_it != operation_attributes.drain_logical_core_candidates.end(),
            "All-to-all drain logical core {} is not one of the exchanged candidates",
            drain_sync_logical_core);
        drain_candidate_index = std::distance(operation_attributes.drain_logical_core_candidates.begin(), candidate_it);
        TT_FATAL(
            operation_attributes.drain_virtual_cores.size() ==
                device->shape().mesh_size() * operation_attributes.drain_logical_core_candidates.size(),
            "All-to-all drain-core map has {} entries for {} nodes and {} candidates",
            operation_attributes.drain_virtual_cores.size(),
            device->shape().mesh_size(),
            operation_attributes.drain_logical_core_candidates.size());
    }
    auto get_target_drain_core = [&](const MeshCoordinate& target_coord) {
        const size_t target_index =
            target_coord.to_linear_index(device->shape()) * operation_attributes.drain_logical_core_candidates.size() +
            drain_candidate_index;
        TT_FATAL(
            target_index < operation_attributes.drain_virtual_cores.size(),
            "Missing all-to-all drain core for mesh coordinate {}",
            target_coord);
        return operation_attributes.drain_virtual_cores[target_index];
    };
    bool use_multicast_initialization = is_fabric_2d && operation_attributes.axis_is_straight;
    if (use_multicast_initialization) {
        // Multicast is valid only on a physically straight axis and when every destination maps the drain semaphore
        // to the same harvested worker. Bent axes and heterogeneous harvesting use destination-specific unicasts.
        for (uint32_t target_device = 0;
             target_device < operation_attributes.num_devices && use_multicast_initialization;
             ++target_device) {
            MeshCoordinate target_coord = mesh_coordinate;
            target_coord[cluster_axis] = target_device;
            use_multicast_initialization = get_target_drain_core(target_coord) == drain_sync_core;
        }
    }

    std::vector<tt::tt_metal::KernelHandle> sender_writer_kernel_ids;
    sender_writer_kernel_ids.reserve(num_senders_per_link);
    for (size_t stream = 0; stream < num_senders_per_link; ++stream) {
        auto sender_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
        sender_writer_kernel_config.compile_args = {
            tt::CB::c_in0,                               // cb0_id
            device_index,                                // device_index
            operation_attributes.num_devices,            // num_devices
            output_shape[operation_attributes.in_dim],   // concat_dim_size
            dst_in_dims,                                 // inner_dims_size
            writer_has_extra_half_tile,                  // has_writer_tail
            page_size,                                   // intermediate_page_size
            reserved_packet_header_CB_index,             // reserved_packet_header_cb_id
            semaphore_sent,                              // semaphore_expected_value
            concat_num_tiles,                            // concat_num_tiles
            (concat_num_half_tiles * device_index) / 2,  // full_block_offset
            static_cast<uint32_t>(effective_topology),   // topology
            cluster_axis,                                // replicate_axis
            sender_device_fabric_node_id.chip_id,        // source_chip_id
            *sender_device_fabric_node_id.mesh_id,       // source_mesh_id
            is_fabric_2d,                                // is_fabric_2d
            sender_stream_direction_masks[stream],       // fabric_direction_mask
            number_pages_per_packet,                     // max_pages_per_packet
            custom_fabric2d_route_words,                 // custom_fabric2d_route_words
            use_multicast_initialization                 // use_multicast_initialization
        };

        tt::tt_metal::TensorAccessorArgs(tensor_return_value.buffer())
            .append_to(sender_writer_kernel_config.compile_args);
        const bool stream_uses_mux = use_worker_mux && stream < num_direction_groups * workers_per_direction &&
                                     sender_stream_direction_masks[stream] != 0;
        sender_writer_kernel_config.compile_args.push_back(stream_uses_mux);
        sender_writer_kernel_config.compile_args.push_back(workers_per_direction);

        sender_writer_kernel_ids.push_back(tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/"
            "all_to_all_sender_writer.cpp",
            sender_stream_core_ranges[stream],
            sender_writer_kernel_config));
    }

    CoreRange sender_box = sender_worker_core_range.bounding_box();
    // Swap start and end coord
    // MeshDevice translates through a representative chip, which is insufficient when Galaxy chips have different
    // Tensix harvesting maps. These addresses are consumed by the program at mesh_coordinate, so translate them on
    // that chip specifically.
    const uint32_t mcast_dest_noc_start_x = coordinate_device->worker_core_from_logical_core(sender_box.end_coord).x;
    const uint32_t mcast_dest_noc_end_x = coordinate_device->worker_core_from_logical_core(sender_box.start_coord).x;
    const uint32_t mcast_dest_noc_start_y = coordinate_device->worker_core_from_logical_core(sender_box.end_coord).y;
    const uint32_t mcast_dest_noc_end_y = coordinate_device->worker_core_from_logical_core(sender_box.start_coord).y;
    const uint32_t mcast_size = sender_box.size();

    CustomFabric2DRoute positive_antipode_route;
    CustomFabric2DRoute negative_antipode_route;
    if (is_fabric_2d && split_antipode_across_arcs) {
        const int32_t half_ring = static_cast<int32_t>(operation_attributes.num_devices / 2);
        positive_antipode_route = build_custom_ring_route(half_ring);
        negative_antipode_route = build_custom_ring_route(-half_ring);
    }
    const CustomFabric2DRoute no_custom_route;

    for (uint32_t core_id = 0; core_id < sender_worker_cores.size(); ++core_id) {
        const auto& core = sender_worker_cores[core_id];
        const size_t sender_stream = core_id % num_senders_per_link;
        const uint32_t link = core_id / num_senders_per_link;
        const auto& stream_device_offsets = device_offsets[sender_stream];
        std::vector<uint32_t> sender_reader_rt_args = {
            tensor_args.input_tensor.buffer()->address(),
            stream_device_offsets.size(),
        };
        for (uint32_t i = 0; i < stream_device_offsets.size(); ++i) {
            sender_reader_rt_args.push_back(stream_device_offsets[i]);
            sender_reader_rt_args.push_back(block_starts[sender_stream][link][i]);
            sender_reader_rt_args.push_back(block_ends[sender_stream][link][i]);
            sender_reader_rt_args.push_back(block_strides[sender_stream][link][i]);
        }
        tt::tt_metal::SetRuntimeArgs(program, sender_reader_kernel_id, {core}, sender_reader_rt_args);

        std::vector<uint32_t> sender_writer_rt_args = {
            tensor_return_value.buffer()->address(),
            init_barrier_semaphore.address(),
            final_barrier_semaphore.address(),
            sender_stream,
            link,
            mcast_dest_noc_start_x,
            mcast_dest_noc_start_y,
            mcast_dest_noc_end_x,
            mcast_dest_noc_end_y,
            mcast_size,
            drain_sync_core.x,
            drain_sync_core.y,
            stream_device_offsets.size(),
        };

        for (uint32_t i = 0; i < stream_device_offsets.size(); ++i) {
            sender_writer_rt_args.push_back(stream_device_offsets[i]);
            sender_writer_rt_args.push_back(block_starts[sender_stream][link][i]);
            sender_writer_rt_args.push_back(block_ends[sender_stream][link][i]);
            sender_writer_rt_args.push_back(block_strides[sender_stream][link][i]);
            sender_writer_rt_args.push_back(completion_flags[sender_stream][link][i]);

            const int32_t device_offset = stream_device_offsets[i];
            const auto target_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
                tensor_args.input_tensor,
                mesh_coordinate,
                device_offset,
                effective_topology,
                operation_attributes.cluster_axis);
            TT_FATAL(target_coord.has_value(), "No all-to-all target at device offset {}", device_offset);

            if (is_fabric_2d) {
                const auto target_node_id = device->get_fabric_node_id(target_coord.value());
                sender_writer_rt_args.push_back(*target_node_id.mesh_id);
                sender_writer_rt_args.push_back(target_node_id.chip_id);
                const auto target_drain_sync_core = get_target_drain_core(*target_coord);
                sender_writer_rt_args.push_back(target_drain_sync_core.x);
                sender_writer_rt_args.push_back(target_drain_sync_core.y);
            } else {
                // The low-latency 1D header uses the second route field as a hop count.
                sender_writer_rt_args.push_back(0);
                sender_writer_rt_args.push_back(std::abs(device_offset));
            }

            const CustomFabric2DRoute* custom_route = &no_custom_route;
            if (is_fabric_2d && split_antipode_across_arcs &&
                std::abs(device_offset) * 2 == operation_attributes.num_devices) {
                custom_route = device_offset > 0 ? &positive_antipode_route : &negative_antipode_route;
            }
            if (custom_fabric2d_route_words > 0) {
                TT_FATAL(
                    custom_route->packed_commands.empty() ||
                        custom_route->packed_commands.size() == custom_fabric2d_route_words,
                    "All-to-all custom route ABI expected {} packed words, got {}",
                    custom_fabric2d_route_words,
                    custom_route->packed_commands.size());
                sender_writer_rt_args.push_back(custom_route->num_commands);
                sender_writer_rt_args.push_back(static_cast<uint32_t>(custom_route->initial_direction));
                if (custom_route->packed_commands.empty()) {
                    sender_writer_rt_args.insert(sender_writer_rt_args.end(), custom_fabric2d_route_words, 0);
                } else {
                    sender_writer_rt_args.insert(
                        sender_writer_rt_args.end(),
                        custom_route->packed_commands.begin(),
                        custom_route->packed_commands.end());
                }
            }
        }
        const bool is_remote_sender = sender_stream != local_sender_stream || num_senders_per_link == 1;
        const bool stream_uses_mux = use_worker_mux && sender_stream < num_direction_groups * workers_per_direction &&
                                     sender_stream_direction_masks[sender_stream] != 0;
        if (stream_uses_mux) {
            const uint32_t direction = sender_stream / workers_per_direction;
            const uint32_t worker = sender_stream % workers_per_direction;
            const auto mux_virtual_core = coordinate_device->worker_core_from_logical_core(mux_cores[link][direction]);
            const auto flow_control_sem_id = tt::tt_metal::CreateSemaphore(program, core, 0);
            const auto teardown_sem_id = tt::tt_metal::CreateSemaphore(program, core, 0);
            mux_config.append_client_connection_rt_args(
                mux_virtual_core,
                static_cast<uint8_t>(worker),
                {.flow_control_sem_id = flow_control_sem_id, .teardown_sem_id = teardown_sem_id},
                sender_writer_rt_args);
        } else if (is_fabric_2d) {
            // Open one routed connection for every physical first-hop direction used by this stream. Logical axes can
            // fold through the physical mesh, so neither the logical sign nor a single link index identifies all valid
            // sender planes. The manager tags each connection by physical direction for the kernel's route lookup.
            std::array<Fabric2DConnectionInfo, num_cardinal_fabric_directions> connection_infos;
            if (is_remote_sender) {
                for (const int32_t device_offset : device_offsets[sender_stream]) {
                    add_fabric2d_connection_target(connection_infos, device_offset);
                }
            }
            std::vector<tt::tt_fabric::FabricNodeId> connection_nodes;
            std::vector<uint32_t> connection_links;
            const uint32_t requested_link = link;
            for (const auto& info : connection_infos) {
                if (!info.representative_node.has_value()) {
                    continue;
                }
                TT_FATAL(
                    requested_link < info.common_links.size(),
                    "All-to-all link {} is unavailable for Fabric2D direction with {} common link(s)",
                    requested_link,
                    info.common_links.size());
                connection_nodes.push_back(*info.representative_node);
                connection_links.push_back(info.common_links[requested_link]);
            }
            sender_writer_rt_args.push_back(static_cast<uint32_t>(connection_nodes.size()));
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                sender_device_fabric_node_id,
                connection_nodes,
                connection_links,
                program,
                sender_writer_kernel_ids[sender_stream],
                core,
                sender_writer_rt_args,
                tt::tt_fabric::FabricApiType::Linear);
        } else {
            const bool owns_positive_direction = !use_direction_owned_schedule || sender_stream < workers_per_direction;
            const bool owns_negative_direction =
                !use_direction_owned_schedule || (sender_stream >= workers_per_direction &&
                                                  sender_stream < logical_num_directions * workers_per_direction);
            const bool with_forward = is_remote_sender && owns_positive_direction && forward_coord.has_value();
            const bool with_backward = is_remote_sender && owns_negative_direction && backward_coord.has_value();
            sender_writer_rt_args.push_back(with_forward);

            if (with_forward) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    sender_device_fabric_node_id,
                    device->get_fabric_node_id(forward_coord.value()),
                    link,
                    program,
                    {core},
                    sender_writer_rt_args);
            }

            sender_writer_rt_args.push_back(with_backward);
            if (with_backward) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    sender_device_fabric_node_id,
                    device->get_fabric_node_id(backward_coord.value()),
                    link,
                    program,
                    {core},
                    sender_writer_rt_args);
            }
        }
        tt::tt_metal::SetRuntimeArgs(program, sender_writer_kernel_ids[sender_stream], {core}, sender_writer_rt_args);
    }

    return {
        std::move(program),
        {.sender_reader_kernel_id = sender_reader_kernel_id,
         .sender_writer_kernel_ids = std::move(sender_writer_kernel_ids),
         .sender_worker_cores = sender_worker_cores,
         .num_senders_per_link = num_senders_per_link,
         .init_barrier_semaphore = init_barrier_semaphore,
         .final_barrier_semaphore = final_barrier_semaphore}};
}

void AllToAllAsyncGenericProgram::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllToAllAsyncGenericParams& /*operation_attributes*/,
    const AllToAllAsyncGenericInputs& tensor_args,
    Tensor& tensor_return_value) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        const auto& coord = coordinate_range.start_coord();
        TT_FATAL(
            coord == coordinate_range.end_coord(),
            "Expected single coordinate per program but got range of {} to {}",
            coord,
            coordinate_range.end_coord());
        auto& shared_variables = cached_workload.shared_variables.at(coordinate_range);

        auto& sender_reader_runtime_args = GetRuntimeArgs(program, shared_variables.sender_reader_kernel_id);
        for (size_t core_id = 0; core_id < shared_variables.sender_worker_cores.size(); ++core_id) {
            const auto& core = shared_variables.sender_worker_cores[core_id];
            const auto writer_kernel_id =
                shared_variables.sender_writer_kernel_ids[core_id % shared_variables.num_senders_per_link];
            auto& sender_writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);
            auto& worker_sender_reader_runtime_args = sender_reader_runtime_args[core.x][core.y];
            auto& worker_sender_writer_runtime_args = sender_writer_runtime_args[core.x][core.y];
            worker_sender_reader_runtime_args[0] = tensor_args.input_tensor.buffer()->address();
            worker_sender_writer_runtime_args[0] = tensor_return_value.buffer()->address();
            worker_sender_writer_runtime_args[1] = shared_variables.init_barrier_semaphore.address();
            worker_sender_writer_runtime_args[2] = shared_variables.final_barrier_semaphore.address();
        }
    }
}

}  // namespace ttnn::experimental::prim
