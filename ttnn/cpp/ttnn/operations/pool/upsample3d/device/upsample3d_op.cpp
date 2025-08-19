// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample3d_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::upsample3d {

void UpSample3D::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Only 1 input tensor supported");
    const auto& input_tensor = input_tensors.at(0);
    const auto& input_shape = input_tensor.logical_shape();

    TT_FATAL(input_shape.rank() == 5, "Input tensor must be 5D (N, D, H, W, C)");
    TT_FATAL(
        scale_factor_d_ > 0 && scale_factor_h_ > 0 && scale_factor_w_ > 0, "Scale factors must be positive integers");
    TT_FATAL(mode_ == "nearest", "Only 'nearest' mode is supported");
}

std::vector<TensorSpec> UpSample3D::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input_shape = input_tensor.logical_shape();

    // Calculate output shape: [N, D*scale_d, H*scale_h, W*scale_w, C]
    auto output_shape = input_shape;
    output_shape[1] = input_shape[1] * scale_factor_d_;  // D dimension
    output_shape[2] = input_shape[2] * scale_factor_h_;  // H dimension
    output_shape[3] = input_shape[3] * scale_factor_w_;  // W dimension
    // C dimension (output_shape[4]) stays the same

    // Use similar layout to existing upsample operation
    Layout output_layout = Layout::ROW_MAJOR;
    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(output_layout), output_mem_config_))};
}

tt::tt_metal::operation::ProgramWithCallbacks UpSample3D::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    return upsample3d_multi_core_interleaved(
        input_tensor, output_tensor, scale_factor_d_, scale_factor_h_, scale_factor_w_);
}

}  // namespace ttnn::operations::upsample3d
