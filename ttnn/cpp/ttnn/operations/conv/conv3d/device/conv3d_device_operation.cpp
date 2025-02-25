// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "conv3d_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::conv {
namespace conv3d {

void Conv3dOp::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    // const auto& input_tensor_b = input_tensors.at(1);
    // TT_FATAL(input_tensor_b.memory_config().is_interleaved(), "Weights tensor must be interleaved.");
    TT_FATAL(input_tensor_a.shape().size() == 5, "Activation tensor must have 5 dimensions.");
    TT_FATAL(input_tensor_a.shape()[0] == 1, "Activation tensor must have batch size 1.");
    // check row-major
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Activation tensor must be row-major.");

    for (const auto& tensor : input_tensors) {
        // input and weight must both be interleaved, bfloat16
        TT_FATAL(!tensor.memory_config().is_sharded(), "Activation tensor must be interleaved.");
        TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "Activation tensor must be bfloat16.");
    }

    // Add assertions for strides and groups
    TT_FATAL(config.stride[0] == 1 && config.stride[1] == 1 && config.stride[2] == 1, "Strides must be (1,1,1).");
    TT_FATAL(config.groups == 1, "Groups must be 1.");
    // assert padding is 0
    TT_FATAL(config.padding[0] == 0, "Padding must be (0,x,x).");
    // TT_FATAL(config.padding[0] == 0 && config.padding[1] == 0 && config.padding[2] == 0, "Padding must be (0,0,0).");
    TT_FATAL(
        config.padding_mode == "zeros" || config.padding_mode == "replicate",
        "Padding mode must be zeros or replicate.");

    // TODO: Use tile_width instead of 32
    if (config.C_out_block > 0) {
        TT_FATAL(
            config.output_channels % config.C_out_block == 0 && config.C_out_block % 32 == 0,
            "C_out_block must be a multiple of 32 and divide evenly into output channels. Got C_out_block={} and "
            "output_channels={}.",
            config.C_out_block,
            config.output_channels);
    }

    TT_FATAL(config.output_channels % 32 == 0, "Output channels must be a multiple of 32.");

    if (optional_input_tensors.at(0).has_value()) {
        const auto& bias_tensor = optional_input_tensors.at(0).value();
        TT_FATAL(!bias_tensor.memory_config().is_sharded(), "Bias tensor must be interleaved.");
        TT_FATAL(bias_tensor.get_layout() == Layout::TILE, "Bias tensor must be tile-major.");
        TT_FATAL(bias_tensor.dtype() == DataType::BFLOAT16, "Bias tensor must be bfloat16.");
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

    // Validate C_in_block if specified
    uint32_t C_in = input_tensor_a.shape()[4];
    if (config.C_in_block > 0) {
        TT_FATAL(
            C_in % config.C_in_block == 0,
            "Input channels ({}) must be divisible by C_in_block ({})",
            C_in,
            config.C_in_block);
        TT_FATAL(config.C_in_block % 32 == 0, "C_in_block ({}) must be a multiple of 32", config.C_in_block);
    }
}

std::vector<TensorSpec> Conv3dOp::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    // Compute vol2col output shape
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_a_shape = input_tensor_a.shape();
    uint32_t N = input_tensor_a_shape[0];
    uint32_t T_in = input_tensor_a_shape[1];
    uint32_t H_in = input_tensor_a_shape[2];
    uint32_t W_in = input_tensor_a_shape[3];
    uint32_t C_in = input_tensor_a_shape[4];
    uint32_t C_out = config.output_channels;

    auto [T_out, H_out, W_out] = detail::compute_output_dims(T_in, H_in, W_in, config.padding, config.kernel_size);

    // uint32_t num_patches = N * T_out * H_out * W_out;
    // uint32_t patch_size = config.kernel_size[0] * config.kernel_size[1] * config.kernel_size[2] * C_in;

    // ttnn::SimpleShape output_shape({num_patches, patch_size});
    ttnn::SimpleShape output_shape({N, T_out, H_out, W_out, C_out});

    auto memory_config = input_tensor_a.memory_config();
    auto dtype = input_tensor_a.dtype();

    return {TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            dtype, PageConfig(Layout::ROW_MAJOR), memory_config, output_shape, output_shape))};
}

operation::Hash Conv3dOp::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    return operation::hash_operation<Conv3dOp>(
        input_tensors, optional_input_tensors, config, output_mem_config, compute_kernel_config);
}

operation::ProgramWithCallbacks Conv3dOp::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& act_tensor = input_tensors.at(0);
    const auto& weight_tensor = input_tensors.at(1);
    const auto& bias_tensor = optional_input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    return detail::conv3d_factory(act_tensor, weight_tensor, bias_tensor, config, output_tensor, compute_kernel_config);
}

}  // namespace conv3d
}  // namespace ttnn::operations::conv
