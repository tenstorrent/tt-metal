// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gate_op.hpp"
#include "gate_program_factory.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::deepseek_b1::gate {

using namespace tt::constants;

void GateDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 3, "Gate operation requires 3 input tensors (a, b, expert_bias)");

    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& input_tensor_expert_bias = input_tensors.at(2);

    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Input A must be on device");
    TT_FATAL(input_tensor_b.storage_type() == StorageType::DEVICE, "Input B must be on device");
    TT_FATAL(input_tensor_expert_bias.storage_type() == StorageType::DEVICE, "Expert bias must be on device");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Input A must be tilized");
    TT_FATAL(input_tensor_b.layout() == Layout::TILE, "Input B must be tilized");
    TT_FATAL(input_tensor_expert_bias.layout() == Layout::TILE, "Expert bias must be tilized");

    // TODO: Add more comprehensive validation when implementing the full gate operation
}

std::vector<ttnn::TensorSpec> GateDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    // Use the matmul helper function to compute output shape
    const auto output_shape = compute_matmul_output_shape(input_tensor_a, input_tensor_b);

    auto dtype = output_dtype.value_or(input_tensor_a.dtype());
    auto mem_config = output_mem_config.value_or(input_tensor_a.memory_config());
    auto tile = input_tensor_a.tensor_spec().tile();

    return {TensorSpec(
        output_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(Layout::TILE, tile), mem_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks GateDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& input_tensor_expert_bias = input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);

    // Call our local program factory with simplified signature
    return deepseek_b1_gate(
        input_tensor_a,
        input_tensor_b,
        input_tensor_expert_bias,
        output_tensor,
        program_config.compute_with_storage_grid_size,
        compute_kernel_config.value_or(DeviceComputeKernelConfig{}));
}

}  // namespace ttnn::operations::experimental::deepseek_b1::gate
