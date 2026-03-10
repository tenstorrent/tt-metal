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

namespace ttnn::experimental {

static Tensor prepare_and_check_weight_tensor(
    const Tensor& weight_tensor,
    uint32_t groups_,
    const ttnn::experimental::prim::Conv3dConfig& config,
    std::optional<ttnn::MeshDevice*> device) {
    Tensor prepared_weight_tensor = weight_tensor;
    switch (prepared_weight_tensor.logical_shape().rank()) {
        case 5:
            TT_FATAL(prepared_weight_tensor.device() == nullptr, "Unprepared weight tensor must be on host");
            TT_FATAL(device.has_value(), "Device must be provided when weight tensor is unprepared (rank 5)");
            prepared_weight_tensor = ttnn::operations::experimental::conv3d::prepare_conv3d_weights(
                prepared_weight_tensor, groups_, config.C_in_block, config.alignment, device.value());
            break;
        case 2: break;
        default: TT_THROW("Unsupported weight tensor rank: {}", prepared_weight_tensor.logical_shape().rank());
    }

    if (prepared_weight_tensor.layout() != Layout::TILE) {
        prepared_weight_tensor = ttnn::to_layout(prepared_weight_tensor, ttnn::Layout::TILE);
    }

    return prepared_weight_tensor;
}

ttnn::Tensor conv3d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    std::optional<ttnn::MeshDevice*> device,
    const std::optional<ttnn::Tensor>& bias_tensor,
    const std::optional<ttnn::experimental::prim::Conv3dConfig>& config_opt,
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
    // If no config provided, use conservative default blocking:
    // minimal spatial blocks (1,1,1), smallest valid C_out_block (32), full C_in (no reduction)
    auto config = config_opt.value_or(ttnn::experimental::prim::Conv3dConfig(
        tt::tt_metal::DataType::BFLOAT16,                        // weights_dtype
        tt::tt_metal::Layout::ROW_MAJOR,                         // output_layout
        1,                                                       // T_out_block
        1,                                                       // W_out_block
        1,                                                       // H_out_block
        32,                                                      // C_out_block (one tile width)
        0,                                                       // C_in_block (0 = full C_in)
        dilation_,                                               // dilation (match the op's dilation)
        32,                                                      // alignment
        input_tensor.device()->compute_with_storage_grid_size()  // use full device grid
        ));

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

}  // namespace ttnn::experimental
