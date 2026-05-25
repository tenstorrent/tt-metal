// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_moe_gate_program_factory.hpp"

#include <tt_stl/assert.hpp>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>

#include "deepseek_moe_gate_program_descriptor_builder.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::program {

namespace {

using namespace tt::tt_metal;

void patch_program_from_descriptor(
    Program& program, DeepseekMoeGateSharedVariables& shared_vars, const ProgramDescriptor& program_descriptor) {
    TT_FATAL(
        shared_vars.num_kernel_handles == program_descriptor.kernels.size(),
        "Kernel handle count mismatch: cached {} vs new {}",
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
            ::tt::tt_metal::experimental::UpdateDynamicCircularBufferAddress(
                program, cb_handle, *cb_desc.global_circular_buffer);
        }
    }
}

}  // namespace

DeepseekMoeGateProgramFactory::cached_program_t DeepseekMoeGateProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    [[maybe_unused]] tensor_return_value_t& tensor_return_value) {
    ProgramDescriptor program_descriptor = build_moe_gate_program_descriptor(tensor_args, operation_attributes);
    Program program{program_descriptor};

    DeepseekMoeGateSharedVariables shared{};
    auto cbs = program.circular_buffers();
    shared.cb_handles.reserve(cbs.size());
    for (const auto& cb : cbs) {
        shared.cb_handles.push_back(static_cast<CBHandle>(cb->id()));
    }
    shared.num_kernel_handles = program_descriptor.kernels.size();

    return {std::move(program), std::move(shared)};
}

void DeepseekMoeGateProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    [[maybe_unused]] tensor_return_value_t& tensor_return_value) {
    ProgramDescriptor program_descriptor = build_moe_gate_program_descriptor(tensor_args, operation_attributes);
    patch_program_from_descriptor(cached_program.program, cached_program.shared_variables, program_descriptor);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::program
