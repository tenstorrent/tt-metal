// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grid_sample.hpp"
#include "device/grid_sample_op.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::grid_sample {

using namespace tt;
using namespace tt::tt_metal;

ttnn::Tensor ExecuteGridSample::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& grid,
    const std::string& mode,
    const std::string& padding_mode,
    bool use_precomputed_grid,
    bool batch_output_channels,
    const std::optional<MemoryConfig>& memory_config) {
    // Use input memory config if not specified
    auto output_memory_config = memory_config.value_or(grid.memory_config());

    // Create the device operation
    auto output_tensors = operation::run(
        GridSample{
            .mode_ = mode,
            .padding_mode_ = padding_mode,
            .use_precomputed_grid_ = use_precomputed_grid,
            .batch_output_channels_ = batch_output_channels,
            .output_mem_config_ = output_memory_config},
        {input_tensor, grid});

    return output_tensors.at(0);
}

}  // namespace ttnn::operations::grid_sample
