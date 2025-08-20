// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "maxpool3d_device_operation.hpp"
#include <array>
#include <cstdint>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "maxpool3d_program_factory.hpp"

#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::maxpool3d {

namespace detail {
std::tuple<uint32_t, uint32_t, uint32_t> compute_output_dims(
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    const std::array<uint32_t, 3>& padding,
    const std::array<uint32_t, 3>& stride,
    const std::array<uint32_t, 3>& kernel_size) {
    uint32_t T_out = (T_in + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
    uint32_t H_out = (H_in + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;
    uint32_t W_out = (W_in + 2 * padding[2] - kernel_size[2]) / stride[2] + 1;
    return {T_out, H_out, W_out};
}
}  // namespace detail

void MaxPool3dOp::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    TT_FATAL(
        input_tensor.logical_shape().size() == 5,
        "Input tensor must have 5 dimensions [N, T, H, W, C]. got {}",
        input_tensor.logical_shape().size());
    TT_FATAL(
        input_tensor.logical_shape()[0] == 1,
        "Input tensor must have batch size 1. got {}",
        input_tensor.logical_shape()[0]);
    // Check row-major layout
    TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "Input tensor must be row-major.");

    // Input must be interleaved, bfloat16
    TT_FATAL(!input_tensor.memory_config().is_sharded(), "Input tensor must be interleaved.");
    TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "Input tensor must be bfloat16.");

    // MaxPool3D doesn't use optional tensors (no bias), but check if provided anyway
    if (optional_input_tensors.at(0).has_value()) {
        TT_FATAL(false, "MaxPool3D does not support optional tensors (bias)");
    }

    // Validate padding mode
    TT_FATAL(
        config.padding_mode == "zeros" || config.padding_mode == "replicate",
        "Padding mode must be zeros or replicate. got {}",
        config.padding_mode);

    // Add grid size validation
    const auto& device_grid = input_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(
        config.compute_with_storage_grid_size.x <= device_grid.x &&
            config.compute_with_storage_grid_size.y <= device_grid.y,
        "Requested grid size ({}, {}) exceeds device grid size ({}, {})",
        config.compute_with_storage_grid_size.x,
        config.compute_with_storage_grid_size.y,
        device_grid.x,
        device_grid.y);
}

std::vector<TensorSpec> MaxPool3dOp::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input_tensor_shape = input_tensor.logical_shape();
    uint32_t N = input_tensor_shape[0];
    uint32_t T_in = input_tensor_shape[1];
    uint32_t H_in = input_tensor_shape[2];
    uint32_t W_in = input_tensor_shape[3];
    uint32_t C = input_tensor_shape[4];

    auto [T_out, H_out, W_out] =
        detail::compute_output_dims(T_in, H_in, W_in, config.padding, config.stride, config.kernel_size);

    ttnn::Shape output_shape({N, T_out, H_out, W_out, C});

    const auto& memory_config = input_tensor.memory_config();
    auto dtype = input_tensor.dtype();

    return {TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), memory_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks MaxPool3dOp::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    return detail::maxpool3d_factory(input_tensor, config, output_tensor, compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::maxpool3d
