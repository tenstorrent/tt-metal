// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample3d_device_operation.hpp"
#include "upsample3D_program_factory.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "tt_stl/assert.hpp"

namespace ttnn::operations::upsample {

// Validation
void UpSample3DDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Expected 1 input tensor");

    const auto& input = input_tensors[0];

    TT_FATAL(input.logical_shape().rank() == 5, "Input must be 5D tensor, got rank {}", input.logical_shape().rank());

    TT_FATAL(
        input.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Only ROW_MAJOR layout is supported, got {}",
        input.layout());

    TT_FATAL(
        input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported");

    TT_FATAL(input.is_allocated(), "Input tensor must be allocated on device");

    TT_FATAL(
        input.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "Input must be on device, got storage type {}",
        input.storage_type());

    TT_FATAL(scale_factor_d_ > 0, "Scale factor D must be positive");
    TT_FATAL(scale_factor_h_ > 0, "Scale factor H must be positive");
    TT_FATAL(scale_factor_w_ > 0, "Scale factor W must be positive");
}

// Compute output specifications
std::vector<TensorSpec> UpSample3DDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input = input_tensors[0];
    const auto& input_shape = input.padded_shape();

    // Calculate output shape: (N, D*scale_d, H*scale_h, W*scale_w, C)
    auto output_shape = input_shape;
    output_shape[1] = input_shape[1] * scale_factor_d_;  // D * scale_d
    output_shape[2] = input_shape[2] * scale_factor_h_;  // H * scale_h
    output_shape[3] = input_shape[3] * scale_factor_w_;  // W * scale_w

    // Create output tensor spec
    return {TensorSpec(
        output_shape, TensorLayout(input.dtype(), PageConfig(tt::tt_metal::Layout::ROW_MAJOR), output_mem_config_))};
}

// Create program
tt::tt_metal::operation::ProgramWithCallbacks UpSample3DDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input = input_tensors[0];
    auto& output = output_tensors[0];

    return upsample3d_multi_core_interleaved(input, output, scale_factor_d_, scale_factor_h_, scale_factor_w_);
}

}  // namespace ttnn::operations::upsample
