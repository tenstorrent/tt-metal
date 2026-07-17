// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d.hpp"
#include "device/conv3d_device_operation.hpp"
#include "ttnn/operations/experimental/conv3d/prepare_conv3d_weights.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/common/constants.hpp"
#include <numeric>
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
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t logical_h_mask,
    uint32_t logical_w_mask,
    const std::optional<ttnn::Tensor>& pad_offset_tensor,
    uint32_t output_pad_h,
    uint32_t output_pad_w) {
    // Default blocking: minimal spatial blocks, smallest valid C_in_block. The same
    // default is used by prepare_conv3d_weights, so the prepared weight's K-row
    // blocking always matches the conv compute -- no near-zero-PCC mismatch (#47316) --
    // and the minimal block keeps large kernels within L1 (#42146). This holds for both
    // a rank-5 (prepared here) and a rank-2 (pre-prepared with the same default) weight.
    const uint32_t default_c_in_block = ttnn::operations::experimental::conv3d::default_c_in_block(
        kernel_size_[0] * kernel_size_[1] * kernel_size_[2]);

    auto config = config_opt.value_or(ttnn::experimental::prim::Conv3dConfig(
        tt::tt_metal::DataType::BFLOAT16,                        // weights_dtype
        tt::tt_metal::Layout::ROW_MAJOR,                         // output_layout
        1,                                                       // T_out_block
        1,                                                       // W_out_block
        1,                                                       // H_out_block
        tt::constants::TILE_WIDTH,                               // C_out_block (one tile width)
        default_c_in_block,                                      // C_in_block (match weight blocking)
        dilation_,                                               // dilation (match the op's dilation)
        32,                                                      // alignment
        input_tensor.device()->compute_with_storage_grid_size()  // use full device grid
        ));

    // An explicitly-provided config may still carry C_in_block == 0 ("auto"). Resolve it to the
    // same default so prepare_conv3d_weights and the conv compute use one identical, non-zero
    // block -- otherwise the internal prepare and the device op disagree on the K-row blocking
    // (near-zero PCC, #47316). This mirrors prepare_conv3d_weights' own 0 handling.
    if (config.C_in_block == 0) {
        config.C_in_block = default_c_in_block;
    }

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
        compute_kernel_config,
        logical_h_mask,
        logical_w_mask,
        pad_offset_tensor,
        output_pad_h,
        output_pad_w);
}

}  // namespace ttnn::experimental
