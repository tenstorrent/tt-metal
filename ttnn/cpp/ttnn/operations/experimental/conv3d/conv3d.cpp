// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d.hpp"
#include "device/conv3d_device_operation.hpp"
#include "prepare_conv3d_weights.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include <tt-logger/tt-logger.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::conv3d {

ttnn::Tensor ExecuteConv3d::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    const Conv3dConfig& config,
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
    ttnn::Tensor weight_tensor_prepared = weight_tensor;

    log_info(tt::LogTest, "weight_tensor.logical_shape(): {}", weight_tensor.logical_shape());

    Conv3dWeightsBiasPrepConfig weight_prep_config;
    weight_prep_config.groups = groups_;

    weight_tensor_prepared = prepare_conv_weights(weight_tensor, weight_prep_config, input_tensor.device());

    log_info(tt::LogTest, "weight_tensor.storage_type(): {}", weight_tensor.storage_type());
    log_info(tt::LogTest, "weight_tensor_prepared.storage_type(): {}", weight_tensor_prepared.storage_type());

    log_info(tt::LogTest, "weight_tensor.layout(): {}", weight_tensor.layout());
    log_info(tt::LogTest, "weight_tensor_prepared.layout(): {}", weight_tensor_prepared.layout());

    log_info(tt::LogTest, "weight_tensor_prepared.logical_shape(): {}", weight_tensor_prepared.logical_shape());

    return ttnn::prim::conv3d(
        input_tensor,
        weight_tensor_prepared,
        bias_tensor,
        config,
        dtype_,
        output_channels_,
        kernel_size_,
        stride_,
        padding_,
        dilation_,
        padding_mode_,
        groups_,
        memory_config,
        compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::conv3d
