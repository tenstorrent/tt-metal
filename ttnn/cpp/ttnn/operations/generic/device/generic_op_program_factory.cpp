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

#include "generic_op_program_factory.hpp"

namespace ttnn::operations::generic::program {
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

GenericMeshProgramFactory::cached_mesh_workload_t GenericMeshProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, mesh_shared_variables_t> mesh_shared_variables;

    for (const auto& [mesh_coord_range, program_descriptor] : operation_attributes.mesh_programs) {
        auto cached_program = create_at(program_descriptor, tensor_args, tensor_return_value);
        mesh_workload.add_program(mesh_coord_range, std::move(cached_program.program));
        mesh_shared_variables[mesh_coord_range] = mesh_shared_variables_t{std::move(cached_program.shared_variables)};
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(mesh_shared_variables)};
}

GenericMeshProgramFactory::cached_program_t GenericMeshProgramFactory::create_at(
    const tt::tt_metal::ProgramDescriptor& program_descriptor,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
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
    GenericMeshProgramFactory::shared_variables_t& shared_vars,
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

void GenericMeshProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_mesh_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    auto& workload_programs = cached_mesh_workload.workload.get_programs();
    const auto& mesh_programs = operation_attributes.mesh_programs;

    TT_FATAL(
        workload_programs.size() == mesh_programs.size(),
        "Size mismatch between cached workload programs ({}) and operation mesh_programs ({})",
        workload_programs.size(),
        mesh_programs.size());

    for (const auto& [range, program_descriptor] : mesh_programs) {
        auto program_it = workload_programs.find(range);
        TT_FATAL(
            program_it != workload_programs.end(),
            "MeshCoordinateRange {} not found in cached workload programs",
            range);

        auto& shared_vars = cached_mesh_workload.shared_variables.at(range);
        override_program_runtime_arguments(
            program_it->second, shared_vars.program_shared_variables, program_descriptor);
    }
}
}  // namespace ttnn::operations::generic::program
