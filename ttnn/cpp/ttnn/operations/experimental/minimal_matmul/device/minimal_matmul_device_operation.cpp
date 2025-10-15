// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_device_operation.hpp"
#include <array>
#include <cstdint>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "minimal_matmul_program_factory.hpp"

#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::minimal_matmul {

void MinimalMatmulOp::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {}

std::vector<TensorSpec> MinimalMatmulOp::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& in0_input_tensor = input_tensors.at(0);
    const auto& in1_input_tensor = input_tensors.at(1);
    const auto& in0_input_tensor_shape = in0_input_tensor.logical_shape();
    const auto& in1_input_tensor_shape = in1_input_tensor.logical_shape();
    uint32_t M = in0_input_tensor_shape[0];
    uint32_t N = in1_input_tensor_shape[1];

    ttnn::Shape output_shape({M, N});

    const auto& memory_config = this->output_mem_config.value_or(in0_input_tensor.memory_config());
    auto dtype = in0_input_tensor.dtype();

    return {TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks MinimalMatmulOp::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& act_tensor = input_tensors.at(0);
    const auto& weight_tensor = input_tensors.at(1);
    const auto& bias_tensor = optional_input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    return detail::minimal_matmul_factory(
        act_tensor,
        weight_tensor,
        bias_tensor,
        this->fused_activation,
        this->config,
        output_tensor,
        compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::minimal_matmul
