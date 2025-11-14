// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gate_op.hpp"
#include "gate_program_factory.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::deepseek_b1::gate {

using namespace tt::constants;

void GateDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& input_tensor_expert_bias = input_tensors.at(2);

    const auto& ashape = input_tensor_a.padded_shape();
    const auto& bshape = input_tensor_b.padded_shape();
    const auto& expert_bias_shape = input_tensor_expert_bias.padded_shape();

    auto in0_tile = input_tensor_a.tensor_spec().tile();
    auto in1_tile = input_tensor_b.tensor_spec().tile();
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();

    // Buffer size checks
    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_b.dtype());
    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    tt::tt_metal::Buffer* in0_buffer = input_tensor_a.buffer();
    tt::tt_metal::Buffer* in1_buffer = input_tensor_b.buffer();
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
