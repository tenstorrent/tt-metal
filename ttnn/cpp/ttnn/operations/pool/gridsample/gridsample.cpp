// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gridsample.hpp"
#include "device/gridsample_op.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::gridsample {

template <typename T>
float normalize_index(T val, int size, bool align_corners) {
    if (align_corners == true) {
        float computed = (val + 1) * 0.5 * (size - 1);
        return computed;
    } else {
        if (val < -1 || val > 1) {
            return std::numeric_limits<float>::max();
        }

        float computed = ((val + 1) * 0.5 * (size)) - 0.5;

        return computed;
    }
}

std::vector<float> normalize_grid(const ttnn::Tensor& grid, int height, int width, bool align_corners) {
    std::vector<float> normalized_grid = grid.to_vector<float>();

    for (int i = 0; i < normalized_grid.size() - 1; i += 2) {
        normalized_grid[i] = normalize_index<float>(normalized_grid[i], width, align_corners);
        normalized_grid[i + 1] = normalize_index<float>(normalized_grid[i + 1], height, align_corners);
    }

    return normalized_grid;
}

ttnn::Tensor ExecuteGridSample::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& grid,
    const std::string& mode,
    const bool align_corners,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    auto normalized_grid =
        normalize_grid(grid, input_tensor.get_logical_shape()[2], input_tensor.get_logical_shape()[3], align_corners);

    MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(
        ttnn::init_device_compute_kernel_config(input_tensor.device()->arch(), std::nullopt, MathFidelity::HiFi4));

    ttnn::Tensor reshaped_input = ttnn::reshape(input_tensor, ttnn::Shape({1, 1, 1, -1}));

    auto output_tensor =
        tt::tt_metal::operation::run(
            gridsample{normalized_grid, mode, align_corners, mem_config, config}, {input_tensor, grid, reshaped_input})
            .front();

    return output_tensor;
}

}  // namespace ttnn::operations::gridsample
