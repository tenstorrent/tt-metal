// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "maxpool3d.hpp"
#include "device/maxpool3d_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::maxpool3d {

ttnn::Tensor ExecuteMaxPool3d::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 3>& kernel_size,
    const std::array<uint32_t, 3>& stride,
    const std::array<uint32_t, 3>& padding,
    const std::string& padding_mode,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    // Calculate output dimensions for grid size estimation
    auto input_shape = input_tensor.logical_shape();
    uint32_t T_in = input_shape[1];
    uint32_t H_in = input_shape[2];
    uint32_t W_in = input_shape[3];

    uint32_t T_out = (T_in + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
    uint32_t H_out = (H_in + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;
    uint32_t W_out = (W_in + 2 * padding[2] - kernel_size[2]) / stride[2] + 1;

    // Calculate intelligent grid size based on output shape
    auto device_grid = input_tensor.device()->compute_with_storage_grid_size();
    uint32_t max_cores = device_grid.x * device_grid.y;
    uint32_t output_elements = T_out * H_out * W_out;

    // Use up to 8 cores for small outputs, scale up for larger outputs
    uint32_t target_cores = std::min(max_cores, std::max(1u, std::min(8u, output_elements)));

    // Try to make a reasonable grid shape
    CoreCoord grid_size = {1, 1};
    if (target_cores > 1) {
        // Try to make a roughly square grid
        uint32_t grid_x = static_cast<uint32_t>(std::sqrt(target_cores));
        uint32_t grid_y = (target_cores + grid_x - 1) / grid_x;

        // Ensure we don't exceed device limits
        grid_x = std::min(grid_x, static_cast<uint32_t>(device_grid.x));
        grid_y = std::min(grid_y, static_cast<uint32_t>(device_grid.y));

        grid_size = {grid_x, grid_y};
    }

    MaxPool3dConfig config(
        input_tensor.dtype(),
        tt::tt_metal::Layout::ROW_MAJOR,
        1,  // T_out_block
        1,  // W_out_block
        1,  // H_out_block
        kernel_size,
        stride,
        padding,
        padding_mode,
        grid_size);

    std::optional<ttnn::Tensor> bias_tensor = std::nullopt;  // MaxPool3D doesn't use bias

    return operation::run(
               MaxPool3dOp{
                   .config = config,
                   .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .compute_kernel_config = kernel_config_val},
               {input_tensor},
               {bias_tensor},
               {},
               queue_id)
        .at(0);
}

}  // namespace ttnn::operations::experimental::maxpool3d
