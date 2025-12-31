// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d.hpp"
#include "device/conv3d_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental {

Tensor conv3d(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<Tensor>& bias_tensor,
    const operations::experimental::conv3d::Conv3dConfig& config,
    tt::tt_metal::DataType dtype_,
    uint32_t output_channels_,
    const std::array<uint32_t, 3>& kernel_size_,
    const std::array<uint32_t, 3>& stride_,
    const std::array<uint32_t, 3>& padding_,
    const std::array<uint32_t, 3>& dilation_,
    const std::string& padding_mode_,
    uint32_t groups_,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = operations::experimental::conv3d::Conv3dDeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    auto operation_attributes = OperationType::operation_attributes_t{
        .config = config,
        .output_mem_config = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        .compute_kernel_config = kernel_config_val,
        .dtype = dtype_,
        .output_channels = output_channels_,
        .kernel_size = kernel_size_,
        .stride = stride_,
        .padding = padding_,
        .dilation = dilation_,
        .padding_mode = padding_mode_,
        .groups = groups_};
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor, .weight_tensor = weight_tensor, .bias_tensor = bias_tensor};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
