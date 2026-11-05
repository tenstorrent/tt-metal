// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_moe_gate_program_factory.hpp"

#include <tt_stl/assert.hpp>

#include "deepseek_moe_gate_program_descriptor_builder.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::program {

namespace {

// The tensor-backed dataflow buffer bound to tensor_name re-resolves its L1 address from this argument.
void add_tensor_backed_dfb_arg(
    tt::tt_metal::experimental::ProgramRunArgs& run_args,
    const tt::tt_metal::experimental::TensorParamName& tensor_name,
    const Tensor& tensor) {
    auto* buffer = tensor.buffer();
    TT_FATAL(buffer != nullptr, "deepseek_moe_gate tensor buffer is null for {}", tensor_name.get());
    run_args.tensor_args.emplace(tensor_name, tensor.mesh_tensor());
}

}  // namespace

ttnn::device_operation::ProgramArtifacts DeepseekMoeGateProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t&) {
    return build_moe_gate_program_artifacts(tensor_args, operation_attributes);
}

tt::tt_metal::experimental::ProgramRunArgs DeepseekMoeGateProgramFactory::override_runtime_arguments(
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t&,
    const std::optional<ttnn::MeshCoordinate>&) {
    tt::tt_metal::experimental::ProgramRunArgs run_args;
    add_tensor_backed_dfb_arg(run_args, kInputTensor, tensor_args.input_tensor);
    add_tensor_backed_dfb_arg(run_args, kBiasTensor, tensor_args.bias_tensor);
    add_tensor_backed_dfb_arg(run_args, kOutputTensor, tensor_args.output_tensor);
    add_tensor_backed_dfb_arg(run_args, kInputIndicesTensor, tensor_args.input_indices_tensor);
    add_tensor_backed_dfb_arg(run_args, kOutputIndicesTensor, tensor_args.output_indices_tensor);
    return run_args;
}

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::program
