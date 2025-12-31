// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grid_sample.hpp"
#include "ttnn/operations/pool/grid_sample/device/grid_sample_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::grid_sample {

ttnn::Tensor grid_sample(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& grid,
    const std::string& mode,
    const std::string& padding_mode,
    bool align_corners,
    bool use_precomputed_grid,
    bool batch_output_channels,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = pool::grid_sample::GridSampleOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .mode = mode,
            .padding_mode = padding_mode,
            .align_corners = align_corners,
            .use_precomputed_grid = use_precomputed_grid,
            .batch_output_channels = batch_output_channels,
            .output_mem_config = memory_config.value_or(grid.memory_config()),
        },
        OperationType::tensor_args_t{.input_tensor = input_tensor, .grid = grid});
}

}  // namespace ttnn::operations::grid_sample
