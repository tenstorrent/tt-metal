// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d.hpp"
#include "device/conv3d_device_operation.hpp"
#include "ttnn/operations/experimental/conv3d/prepare_conv3d_weights.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::conv3d {

static Tensor prepare_and_check_weight_tensor(
    const Tensor& weight_tensor,
    uint32_t groups_,
    const ttnn::experimental::prim::Conv3dConfig& config,
    ttnn::MeshDevice* device) {
    Tensor prepared_weight_tensor = weight_tensor;
    switch (prepared_weight_tensor.logical_shape().rank()) {
        case 5:
            TT_FATAL(prepared_weight_tensor.device() == nullptr, "Unprepared weight tensor must be on host");
            prepared_weight_tensor = ttnn::operations::experimental::conv3d::prepare_weights(
                prepared_weight_tensor, groups_, config.C_in_block, device);
            break;
        case 2: break;
        default: TT_THROW("Unsupported weight tensor rank: {}", prepared_weight_tensor.logical_shape().rank());
    }

    if (prepared_weight_tensor.layout() != Layout::TILE) {
        prepared_weight_tensor = ttnn::to_layout(prepared_weight_tensor, ttnn::Layout::TILE);
    }

    return prepared_weight_tensor;
}

ttnn::Tensor ExecuteConv3d::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    ttnn::MeshDevice* device,
    const std::optional<ttnn::Tensor>& bias_tensor,
    const ttnn::experimental::prim::Conv3dConfig& config,
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
    Tensor prepared_weight_tensor = prepare_and_check_weight_tensor(weight_tensor, groups_, config, device);
    return ttnn::prim::conv3d(
        input_tensor,
        prepared_weight_tensor,
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
