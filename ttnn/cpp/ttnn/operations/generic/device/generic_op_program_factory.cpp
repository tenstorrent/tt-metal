// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "generic_op_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::generic {
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

std::vector<FabricConnectionDescriptor> generate_fabric_connections_from_topology(
    const FabricTopologyDescriptor& topo_desc,
    const Tensor& reference_tensor,
    const std::vector<ttnn::MeshCoordinate>& all_coords) {
    std::vector<FabricConnectionDescriptor> connections;

    for (const auto& coord : all_coords) {
        auto forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            reference_tensor, coord, 1, topo_desc.topology, topo_desc.cluster_axis);

        if (forward_coord.has_value()) {
            connections.push_back(FabricConnectionDescriptor{
                .src_coord = coord,
                .dst_coord = forward_coord.value(),
                .kernel_index = topo_desc.kernel_index,
                .worker_core = topo_desc.worker_core,
                .link_idx = 0,
                .core_type = topo_desc.core_type});
        }

        if (topo_desc.topology != tt::tt_fabric::Topology::NeighborExchange) {
            auto backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
                reference_tensor, coord, -1, topo_desc.topology, topo_desc.cluster_axis);

            if (backward_coord.has_value()) {
                connections.push_back(FabricConnectionDescriptor{
                    .src_coord = coord,
                    .dst_coord = backward_coord.value(),
                    .kernel_index = topo_desc.kernel_index,
                    .worker_core = topo_desc.worker_core,
                    .link_idx = 0,
                    .core_type = topo_desc.core_type});
            }
        }
    }

    return connections;
}

// Apply a single fabric connection to a program
void apply_fabric_connection(
    Program& program, const FabricConnectionDescriptor& connection, distributed::MeshDevice* mesh_device) {
    auto src_fabric_node_id = mesh_device->get_fabric_node_id(connection.src_coord);
    auto dst_fabric_node_id = mesh_device->get_fabric_node_id(connection.dst_coord);

    auto link_indices = tt::tt_fabric::get_forwarding_link_indices(src_fabric_node_id, dst_fabric_node_id);
    uint32_t selected_link =
        (link_indices.empty() || connection.link_idx < link_indices.size()) ? connection.link_idx : link_indices[0];

    // Copy existing runtime args and append fabric connection args
    auto& kernel_rt_args = GetRuntimeArgs(program, connection.kernel_index, connection.worker_core);
    std::vector<uint32_t> args_vec(kernel_rt_args.data(), kernel_rt_args.data() + kernel_rt_args.size());

    tt::tt_fabric::append_fabric_connection_rt_args(
        src_fabric_node_id,
        dst_fabric_node_id,
        selected_link,
        program,
        connection.worker_core,
        args_vec,
        connection.core_type);

    SetRuntimeArgs(program, connection.kernel_index, connection.worker_core, args_vec);
}

// Find the ProgramDescriptor that covers the given coordinate
const ProgramDescriptor* find_program_descriptor_for_coord(
    const std::unordered_map<ttnn::MeshCoordinateRange, ProgramDescriptor>& mesh_programs,
    const ttnn::MeshCoordinate& coord) {
    for (const auto& [range, desc] : mesh_programs) {
        if (range.contains(coord)) {
            return &desc;
        }
    }
    return nullptr;
}

}  // namespace

GenericOpDeviceOperation::GenericMeshProgram::cached_mesh_workload_t
GenericOpDeviceOperation::GenericMeshProgram::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, mesh_shared_variables_t> mesh_shared_variables;

    const bool has_fabric_topology = operation_attributes.fabric_topology.has_value();
    const bool has_fabric_connections = !operation_attributes.fabric_connections.empty();
    const bool use_fabric = has_fabric_topology || has_fabric_connections;

    if (!use_fabric) {
        for (const auto& [mesh_coord_range, program_descriptor] : operation_attributes.mesh_programs) {
            auto cached_program = create_at(program_descriptor, tensor_args, tensor_return_value);
            mesh_workload.add_program(mesh_coord_range, std::move(cached_program.program));
            mesh_shared_variables[mesh_coord_range] =
                mesh_shared_variables_t{std::move(cached_program.shared_variables)};
        }
        return cached_mesh_workload_t{std::move(mesh_workload), std::move(mesh_shared_variables)};
    }

    distributed::MeshDevice* mesh_device = tensor_args.io_tensors.front().device();
    std::vector<ttnn::MeshCoordinate> all_coords = tensor_coords.coords();

    std::vector<FabricConnectionDescriptor> fabric_connections;
    if (has_fabric_topology) {
        const auto& reference_tensor = tensor_args.io_tensors.front();
        fabric_connections = generate_fabric_connections_from_topology(
            operation_attributes.fabric_topology.value(), reference_tensor, all_coords);
    } else {
        fabric_connections = operation_attributes.fabric_connections;
    }

    std::unordered_map<ttnn::MeshCoordinate, Program> programs_by_coord;
    std::unordered_map<ttnn::MeshCoordinate, shared_variables_t> shared_vars_by_coord;
    for (const auto& [mesh_coord_range, program_descriptor] : operation_attributes.mesh_programs) {
        for (const auto& coord : mesh_coord_range) {
            auto cached_program = create_at(program_descriptor, tensor_args, tensor_return_value);
            programs_by_coord[coord] = std::move(cached_program.program);
            shared_vars_by_coord[coord] = std::move(cached_program.shared_variables);
        }
    }

    // Apply fabric connections to programs
    for (const auto& connection : fabric_connections) {
        auto it = programs_by_coord.find(connection.src_coord);
        if (it != programs_by_coord.end()) {
            apply_fabric_connection(it->second, connection, mesh_device);
        }
    }

    // Add programs to workload (stored by individual coordinate instead of ranges when fabric is used)
    for (auto& [coord, program] : programs_by_coord) {
        ttnn::MeshCoordinateRange single_coord_range(coord);
        mesh_workload.add_program(single_coord_range, std::move(program));
        mesh_shared_variables[single_coord_range] = mesh_shared_variables_t{std::move(shared_vars_by_coord[coord])};
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(mesh_shared_variables)};
}

GenericOpDeviceOperation::GenericMeshProgram::cached_program_t GenericOpDeviceOperation::GenericMeshProgram::create_at(
    const tt::tt_metal::ProgramDescriptor& program_descriptor,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    Program program{program_descriptor};
    shared_variables_t shared_vars;

    auto cbs = program.circular_buffers();
    shared_vars.cb_handles.reserve(cbs.size());
    for (const auto& cb : cbs) {
        shared_vars.cb_handles.push_back(static_cast<tt::tt_metal::CBHandle>(cb->id()));
    }
    shared_vars.num_kernel_handles = program_descriptor.kernels.size();

    return {std::move(program), std::move(shared_vars)};
}

void override_program_runtime_arguments(
    Program& program,
    GenericOpDeviceOperation::GenericMeshProgram::shared_variables_t& shared_vars,
    const ProgramDescriptor& program_descriptor) {
    // Update kernel runtime args.
    TT_ASSERT(
        shared_vars.num_kernel_handles == program_descriptor.kernels.size(),
        "Number of kernel handles mismatch: cached {} vs new program {}",
        shared_vars.num_kernel_handles,
        program_descriptor.kernels.size());
    for (size_t kernel_handle = 0; kernel_handle < shared_vars.num_kernel_handles; ++kernel_handle) {
        const auto& kernel_desc = program_descriptor.kernels[kernel_handle];

        for (const auto& [core_coord, runtime_arg] : kernel_desc.runtime_args) {
            if (!runtime_arg.empty()) {
                auto& cached_runtime_args = GetRuntimeArgs(program, kernel_handle, core_coord);
                TT_FATAL(
                    cached_runtime_args.size() == runtime_arg.size(),
                    "Runtime args size mismatch: cached {} vs new {}",
                    cached_runtime_args.size(),
                    runtime_arg.size());
                std::copy(runtime_arg.begin(), runtime_arg.end(), cached_runtime_args.data());
            }
        }
        if (!kernel_desc.common_runtime_args.empty()) {
            auto& cached_common_runtime_args = GetCommonRuntimeArgs(program, kernel_handle);
            TT_FATAL(
                cached_common_runtime_args.size() == kernel_desc.common_runtime_args.size(),
                "Common runtime args size mismatch: cached {} vs new {}",
                cached_common_runtime_args.size(),
                kernel_desc.common_runtime_args.size());
            std::copy(
                kernel_desc.common_runtime_args.begin(),
                kernel_desc.common_runtime_args.end(),
                cached_common_runtime_args.data());
        }
    }

    // Update circular buffer config.
    for (size_t cb_idx = 0; cb_idx < program_descriptor.cbs.size(); ++cb_idx) {
        const auto& cb_desc = program_descriptor.cbs[cb_idx];
        auto cb_handle = shared_vars.cb_handles[cb_idx];
        const CircularBufferConfig& cb_config = GetCircularBufferConfig(program, cb_handle);

        if (cb_config.total_size() != cb_desc.total_size) {
            UpdateCircularBufferTotalSize(program, cb_handle, cb_desc.total_size);
        }
        const auto& current_page_sizes = cb_config.page_sizes();
        for (const auto& format_desc : cb_desc.format_descriptors) {
            if (current_page_sizes[format_desc.buffer_index].has_value() &&
                current_page_sizes[format_desc.buffer_index].value() != format_desc.page_size) {
                UpdateCircularBufferPageSize(program, cb_handle, format_desc.buffer_index, format_desc.page_size);
            }
        }
        if (cb_desc.buffer != nullptr) {
            UpdateDynamicCircularBufferAddress(program, cb_handle, *cb_desc.buffer);
        }
        if (cb_desc.global_circular_buffer != nullptr) {
            experimental::UpdateDynamicCircularBufferAddress(program, cb_handle, *cb_desc.global_circular_buffer);
        }
    }
}

void GenericOpDeviceOperation::GenericMeshProgram::override_runtime_arguments(
    cached_mesh_workload_t& cached_mesh_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    const bool use_fabric =
        operation_attributes.fabric_topology.has_value() || !operation_attributes.fabric_connections.empty();

    for (auto& [mesh_coord_range, program] : cached_mesh_workload.workload.get_programs()) {
        auto& shared_vars = cached_mesh_workload.shared_variables.at(mesh_coord_range);
        auto& program_shared_vars = shared_vars.program_shared_variables;

        const ProgramDescriptor* program_desc = nullptr;
        if (!use_fabric) {
            auto it = operation_attributes.mesh_programs.find(mesh_coord_range);
            if (it != operation_attributes.mesh_programs.end()) {
                program_desc = &it->second;
            }
        } else {
            // Fabric used - programs stored by single coordinate (not ranges), find matching descriptor
            for (const auto& coord : mesh_coord_range) {
                program_desc = find_program_descriptor_for_coord(operation_attributes.mesh_programs, coord);
                break;
            }
        }

        if (program_desc) {
            override_program_runtime_arguments(program, program_shared_vars, *program_desc);
        }
    }
}
}  // namespace ttnn::operations::generic
