// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample3d.hpp"
#include "device/upsample3d_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_stl/assert.hpp"

namespace ttnn::operations::upsample {

ttnn::Tensor ExecuteUpSample3D::invoke(
    const ttnn::Tensor& input_tensor,
    std::variant<int, std::array<int, 3>> scale_factor,
    const std::optional<MemoryConfig>& output_mem_config) {
    // Input validation
    TT_FATAL(
        input_tensor.logical_shape().rank() == 5,
        "Input tensor must be 5D (N, D, H, W, C), got rank {}",
        input_tensor.logical_shape().rank());

    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Only ROW_MAJOR layout is supported for 3D upsample, got {}",
        input_tensor.layout());

    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Input tensor must be on device");

    // Parse and validate scale factors
    uint32_t scale_d, scale_h, scale_w;
    if (std::holds_alternative<int>(scale_factor)) {
        int scale = std::get<int>(scale_factor);
        TT_FATAL(scale > 0, "Scale factor must be positive, got {}", scale);
        scale_d = scale_h = scale_w = scale;
    } else {
        auto scales = std::get<std::array<int, 3>>(scale_factor);
        TT_FATAL(
            scales[0] > 0 && scales[1] > 0 && scales[2] > 0,
            "All scale factors must be positive, got ({}, {}, {})",
            scales[0],
            scales[1],
            scales[2]);
        scale_d = scales[0];
        scale_h = scales[1];
        scale_w = scales[2];
    }

    // Use memory config from parameter or input
    auto output_memory_config = output_mem_config.value_or(input_tensor.memory_config());

    // Call device operation using the proper TTNN operation framework
    return tt::tt_metal::operation::run(
               UpSample3DDeviceOperation{scale_d, scale_h, scale_w, output_memory_config}, {input_tensor})
        .front();
}

}  // namespace ttnn::operations::upsample
