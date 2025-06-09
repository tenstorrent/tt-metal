// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "conv3d_program_factory.hpp"

#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::conv3d {

namespace detail {
std::tuple<uint32_t, uint32_t, uint32_t> compute_output_dims(
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    const std::array<uint32_t, 3>& padding,
    const std::array<uint32_t, 3>& kernel_size) {
    uint32_t T_out = T_in + 2 * padding[0] - (kernel_size[0] - 1);
    uint32_t H_out = H_in + 2 * padding[1] - (kernel_size[1] - 1);
    uint32_t W_out = W_in + 2 * padding[2] - (kernel_size[2] - 1);
    return {T_out, H_out, W_out};
}
}  // namespace detail

void Conv3dOp::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);

    TT_FATAL(
        input_tensor_a.logical_shape().size() == 5,
        "Activation tensor must have 5 dimensions. got {}",
        input_tensor_a.logical_shape().size());
    TT_FATAL(
        input_tensor_a.logical_shape()[0] == 1,
        "Activation tensor must have batch size 1. got {}",
        input_tensor_a.logical_shape()[0]);
    // check row-major
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Activation tensor must be row-major.");

    for (const auto& tensor : input_tensors) {
        // input and weight must both be interleaved, bfloat16
        TT_FATAL(!tensor.memory_config().is_sharded(), "Activation tensor must be interleaved.");
        TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "Activation tensor must be bfloat16.");
    }

    const auto& weight_tensor = input_tensors.at(1);
    TT_FATAL(weight_tensor.layout() == Layout::TILE, "Weight tensor must be tile.");

    if (optional_input_tensors.at(0).has_value()) {
        const auto& bias_tensor = optional_input_tensors.at(0).value();
        TT_FATAL(!bias_tensor.memory_config().is_sharded(), "Bias tensor must be interleaved.");
        TT_FATAL(bias_tensor.layout() == Layout::TILE, "Bias tensor must be tiled.");
        TT_FATAL(
            bias_tensor.dtype() == DataType::BFLOAT16, "Bias tensor must be bfloat16. got {}", bias_tensor.dtype());
        TT_FATAL(
            bias_tensor.logical_shape().size() == 2,
            "Bias tensor must have 2 dimensions. got {}",
            bias_tensor.logical_shape().size());
    }

    // Add assertions for strides and groups
    TT_FATAL(
        config.stride[0] == 1 && config.stride[1] == 1 && config.stride[2] == 1,
        "Strides must be (1,1,1). got ({}, {}, {})",
        config.stride[0],
        config.stride[1],
        config.stride[2]);
    TT_FATAL(config.groups == 1, "Groups must be 1. got {}", config.groups);
    // assert padding on T is zero
    TT_FATAL(
        config.padding[0] == 0,
        "Padding must be (0,x,x). got ({}, {}, {})",
        config.padding[0],
        config.padding[1],
        config.padding[2]);
    TT_FATAL(
        config.padding_mode == "zeros" || config.padding_mode == "replicate",
        "Padding mode must be zeros or replicate. got {}",
        config.padding_mode);

    if (config.C_out_block > 0) {
        TT_FATAL(
            config.output_channels % config.C_out_block == 0 && config.C_out_block % tt::constants::TILE_WIDTH == 0,
            "C_out_block must be a multiple of {} and divide evenly into output channels. Got C_out_block={} and "
            "output_channels={}.",
            tt::constants::TILE_WIDTH,
            config.C_out_block,
            config.output_channels);
    }

    TT_FATAL(
        config.output_channels % tt::constants::TILE_WIDTH == 0,
        "Output channels must be a multiple of {}.",
        tt::constants::TILE_WIDTH);

    // Validate weight shape and config arguments
    const auto patch_size =
        config.kernel_size[0] * config.kernel_size[1] * config.kernel_size[2] * input_tensor_a.logical_shape()[4];
    TT_FATAL(
        weight_tensor.logical_shape()[0] == patch_size,
        "Weight patch size must match input patch size. got {} vs {}",
        weight_tensor.logical_shape()[0],
        patch_size);
    TT_FATAL(
        weight_tensor.logical_shape()[1] == config.output_channels,
        "Weight output channels must match input output channels. got {} vs {}",
        weight_tensor.logical_shape()[1],
        config.output_channels);
    if (optional_input_tensors.at(0).has_value()) {
        const auto& bias_tensor = optional_input_tensors.at(0).value();
        TT_FATAL(
            bias_tensor.logical_shape()[1] == config.output_channels,
            "Bias must match output channels. got {} vs {}",
            bias_tensor.logical_shape()[1],
            config.output_channels);
    }

    // Add grid size validation
    const auto& device_grid = input_tensor_a.device()->compute_with_storage_grid_size();
    TT_FATAL(
        config.compute_with_storage_grid_size.x <= device_grid.x &&
            config.compute_with_storage_grid_size.y <= device_grid.y,
        "Requested grid size ({}, {}) exceeds device grid size ({}, {})",
        config.compute_with_storage_grid_size.x,
        config.compute_with_storage_grid_size.y,
        device_grid.x,
        device_grid.y);

    uint32_t C_in = input_tensor_a.logical_shape()[4];
    const uint32_t l1_alignment = hal::get_l1_alignment();
    if (config.C_in_block > 0) {
        TT_FATAL(
            C_in % config.C_in_block == 0,
            "Input channels ({}) must be divisible by C_in_block ({})",
            C_in,
            config.C_in_block);
        TT_FATAL(
            config.C_in_block % l1_alignment == 0,
            "C_in_block ({}) must be a multiple of {}",
            config.C_in_block,
            l1_alignment);
    }

    // Verify number of C_in_blocks is <= the number of cores
    uint32_t C_in_block = (config.C_in_block > 0) ? config.C_in_block : C_in;
    uint32_t C_in_blocks = C_in / C_in_block;
    uint32_t total_cores = config.compute_with_storage_grid_size.x * config.compute_with_storage_grid_size.y;
    TT_FATAL(
        C_in_blocks <= total_cores,
        "Number of C_in blocks ({}) must be <= the number of cores ({})",
        C_in_blocks,
        total_cores);
}

std::vector<TensorSpec> Conv3dOp::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_a_shape = input_tensor_a.logical_shape();
    uint32_t N = input_tensor_a_shape[0];
    uint32_t T_in = input_tensor_a_shape[1];
    uint32_t H_in = input_tensor_a_shape[2];
    uint32_t W_in = input_tensor_a_shape[3];
    uint32_t C_in = input_tensor_a_shape[4];
    uint32_t C_out = config.output_channels;

    auto [T_out, H_out, W_out] = detail::compute_output_dims(T_in, H_in, W_in, config.padding, config.kernel_size);

    ttnn::Shape output_shape({N, T_out, H_out, W_out, C_out});

    auto memory_config = input_tensor_a.memory_config();
    auto dtype = input_tensor_a.dtype();

    return {TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), memory_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks Conv3dOp::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& act_tensor = input_tensors.at(0);
    const auto& weight_tensor = input_tensors.at(1);
    const auto& bias_tensor = optional_input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    return detail::conv3d_factory(act_tensor, weight_tensor, bias_tensor, config, output_tensor, compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::conv3d
