// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generalized_moe_gate_program_factory.hpp"

#include <tt_stl/assert.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/work_split.hpp>

#include "generalized_moe_gate_program_descriptor_builder.hpp"

#include <algorithm>
#include <cstdint>

namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate::program {

namespace {

void update_tensor_cb(tt::tt_metal::Program& program, uint8_t cb_index, const Tensor& tensor) {
    auto* buffer = tensor.buffer();
    TT_FATAL(
        buffer != nullptr, "generalized_moe_gate tensor buffer is null for CB {}", static_cast<uint32_t>(cb_index));
    const auto cbs = program.circular_buffers();
    const auto cb_it = std::find_if(cbs.begin(), cbs.end(), [cb_index](const auto& cb) {
        return cb->globally_allocated() && cb->buffer_indices().contains(cb_index);
    });
    TT_FATAL(cb_it != cbs.end(), "Missing globally allocated circular buffer {}", static_cast<uint32_t>(cb_index));
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, (*cb_it)->id(), *buffer);
}

// An interleaved input has no tensor backed CB; kernel 0 is the reader, the first kernel of the descriptor.
void update_reader_input_address(tt::tt_metal::Program& program, const tensor_args_t& tensor_args) {
    constexpr tt::tt_metal::KernelHandle reader_kernel = 0;
    const uint32_t address = tensor_args.input_tensor.buffer()->address();
    const auto& grid = tensor_args.bias_tensor.shard_spec().value().grid;
    for (const auto& core : tt::tt_metal::corerange_to_cores(grid, std::nullopt, true)) {
        tt::tt_metal::GetRuntimeArgs(program, reader_kernel, core)[0] = address;
    }
}

}  // namespace

tt::tt_metal::ProgramDescriptor GeneralizedMoeGateProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t&) {
    return build_moe_gate_program_descriptor(tensor_args, operation_attributes);
}

void GeneralizedMoeGateProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t&,
    const std::optional<ttnn::MeshCoordinate>&) {
    if (tensor_args.input_tensor.is_sharded()) {
        update_tensor_cb(program, kInputCb, tensor_args.input_tensor);
    } else {
        update_reader_input_address(program, tensor_args);
    }
    update_tensor_cb(program, kBiasCb, tensor_args.bias_tensor);
    update_tensor_cb(program, kOutputCb, tensor_args.output_tensor);
    update_tensor_cb(program, kInputIndicesCb, tensor_args.input_indices_tensor);
    update_tensor_cb(program, kOutputIndicesCb, tensor_args.output_indices_tensor);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate::program
