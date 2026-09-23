// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_moe_gate_program_factory.hpp"

#include <tt_stl/assert.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>

#include "deepseek_moe_gate_program_descriptor_builder.hpp"

#include <algorithm>
#include <cstdint>

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::program {

namespace {

constexpr uint8_t kInputCb = 0;
constexpr uint8_t kBiasCb = 1;
constexpr uint8_t kOutputCb = 2;
constexpr uint8_t kInputIndicesCb = 3;
constexpr uint8_t kOutputIndicesCb = 4;

void update_tensor_cb(tt::tt_metal::Program& program, uint8_t cb_index, const Tensor& tensor) {
    auto* buffer = tensor.buffer();
    TT_FATAL(buffer != nullptr, "deepseek_moe_gate tensor buffer is null for CB {}", static_cast<uint32_t>(cb_index));
    const auto cbs = program.circular_buffers();
    const auto cb_it = std::find_if(cbs.begin(), cbs.end(), [cb_index](const auto& cb) {
        return cb->globally_allocated() && cb->buffer_indices().contains(cb_index);
    });
    TT_FATAL(cb_it != cbs.end(), "Missing globally allocated circular buffer {}", static_cast<uint32_t>(cb_index));
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, (*cb_it)->id(), *buffer);
}

}  // namespace

tt::tt_metal::ProgramDescriptor DeepseekMoeGateProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t&) {
    return build_moe_gate_program_descriptor(tensor_args, operation_attributes);
}

void DeepseekMoeGateProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t&,
    const std::optional<ttnn::MeshCoordinate>&) {
    update_tensor_cb(program, kInputCb, tensor_args.input_tensor);
    update_tensor_cb(program, kBiasCb, tensor_args.bias_tensor);
    update_tensor_cb(program, kOutputCb, tensor_args.output_tensor);
    update_tensor_cb(program, kInputIndicesCb, tensor_args.input_indices_tensor);
    update_tensor_cb(program, kOutputIndicesCb, tensor_args.output_indices_tensor);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::program
