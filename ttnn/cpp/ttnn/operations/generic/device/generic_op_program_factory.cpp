// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/global_circular_buffer.hpp>

#include "generic_op_device_operation.hpp"

namespace ttnn::operations::generic {
using namespace tt::tt_metal;

GenericOpDeviceOperation::GenericProgram::cached_program_t GenericOpDeviceOperation::GenericProgram::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    Program program{operation_attributes};

    shared_variables_t shared_vars;

    auto cbs = program.circular_buffers();
    shared_vars.cb_handles.reserve(cbs.size());
    for (const auto& cb : cbs) {
        shared_vars.cb_handles.push_back(static_cast<tt::tt_metal::CBHandle>(cb->id()));
    }
    shared_vars.num_kernel_handles = operation_attributes.kernels.size();

    return {std::move(program), std::move(shared_vars)};
}

void GenericOpDeviceOperation::GenericProgram::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    // Update kernel runtime args.
    TT_ASSERT(
        shared_vars.num_kernel_handles == operation_attributes.kernels.size(),
        "Number of kernel handles mismatch: cached {} vs new program {}",
        shared_vars.num_kernel_handles,
        operation_attributes.kernels.size());
    for (size_t kernel_handle = 0; kernel_handle < shared_vars.num_kernel_handles; ++kernel_handle) {
        const auto& kernel_desc = operation_attributes.kernels[kernel_handle];

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
    for (size_t cb_idx = 0; cb_idx < operation_attributes.cbs.size(); ++cb_idx) {
        const auto& cb_desc = operation_attributes.cbs[cb_idx];
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

}  // namespace ttnn::operations::generic
