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

void Conv3dOp::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    // const auto& input_tensor_b = input_tensors.at(1);
    TT_FATAL(!input_tensor_a.memory_config().is_sharded(), "Activation tensor must be interleaved.");
    // TT_FATAL(input_tensor_b.memory_config().is_interleaved(), "Weights tensor must be interleaved.");
    TT_FATAL(input_tensor_a.shape().size() == 5, "Activation tensor must have 5 dimensions.");
    // check row-major
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Activation tensor must be row-major.");

    TT_FATAL(input_tensor_a.dtype() == DataType::BFLOAT16, "Activation tensor must be bfloat16.");

    // Add assertions for strides and groups
    TT_FATAL(stride[0] == 1 && stride[1] == 1 && stride[2] == 1, "Strides must be (1,1,1).");
    TT_FATAL(groups == 1, "Groups must be 1.");
    // assert padding is 0
    TT_FATAL(padding[0] == 0, "Padding must be (0,x,x).");
    // TT_FATAL(padding[0] == 0 && padding[1] == 0 && padding[2] == 0, "Padding must be (0,0,0).");
    TT_FATAL(padding_mode == "zeros", "Padding mode must be zeros.");
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

    uint32_t T_out = T_in + 2 * padding[0] - (kernel_size[0] - 1);
    uint32_t H_out = H_in + 2 * padding[1] - (kernel_size[1] - 1);
    uint32_t W_out = W_in + 2 * padding[2] - (kernel_size[2] - 1);
    uint32_t C_out = output_channels;

    uint32_t num_patches = N * T_out * H_out * W_out;
    uint32_t patch_size = kernel_size[0] * kernel_size[1] * kernel_size[2] * C_in;

    ttnn::SimpleShape output_shape({num_patches, patch_size});

    auto memory_config = input_tensor_a.memory_config();
    auto dtype = input_tensor_a.dtype();

    return {TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            dtype, PageConfig(Layout::ROW_MAJOR), memory_config, output_shape, output_shape))};
}

operation::ProgramWithCallbacks Conv3dOp::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    // TODO: Implement actual conv3d program
    return detail::conv3d_factory(
        input_tensors.at(0), output_channels, kernel_size, stride, padding, padding_mode, groups, output_tensors.at(0));
}

}  // namespace conv3d
}  // namespace ttnn::operations::conv
