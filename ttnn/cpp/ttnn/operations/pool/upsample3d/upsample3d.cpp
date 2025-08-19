// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample3d.hpp"
#include "device/upsample3d_op.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::upsample3d {

ttnn::Tensor ExecuteUpSample3D::invoke(
    const ttnn::Tensor& input_tensor,
    std::variant<int, tt::tt_metal::Array3D> scale_factor,
    const std::string& mode,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // Parameter validation
    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL(input_shape.rank() == 5, "Input tensor must be 5D (N, D, H, W, C)");

    // Parse scale factors
    int scale_d = 1;
    int scale_h = 1;
    int scale_w = 1;
    std::visit(
        [&scale_d, &scale_h, &scale_w](auto&& sf) {
            using T = std::decay_t<decltype(sf)>;
            if constexpr (std::is_same_v<T, int>) {
                scale_d = sf;
                scale_h = sf;
                scale_w = sf;
            } else if constexpr (std::is_same_v<T, tt::tt_metal::Array3D>) {
                scale_d = sf.at(0);
                scale_h = sf.at(1);
                scale_w = sf.at(2);
            } else {
                static_assert(sizeof(T) != 0, "Type check failed.");
            }
        },
        scale_factor);

    // Validate scale factors
    TT_FATAL(scale_d > 0 && scale_h > 0 && scale_w > 0, "Scale factors must be positive integers");

    // Validate mode
    TT_FATAL(mode == "nearest", "Only 'nearest' mode is supported for upsample3d");

    // Get memory config
    MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());

    // Get compute kernel config
    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(
        ttnn::init_device_compute_kernel_config(input_tensor.device()->arch(), std::nullopt, MathFidelity::HiFi4));

    // Run the device operation
    auto output_tensor =
        tt::tt_metal::operation::run(UpSample3D{scale_d, scale_h, scale_w, mode, mem_config, config}, {input_tensor})
            .front();

    return output_tensor;
}

}  // namespace ttnn::operations::upsample3d
