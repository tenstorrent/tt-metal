// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grid_sample.hpp"
// #include "device/grid_sample_op.hpp"
#include "ttnn/operations/pool/grid_sample/device/grid_sample_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::grid_sample {

using namespace tt;
using namespace tt::tt_metal;

ttnn::Tensor ExecuteGridSample::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& grid,
    const std::string& mode,
    const std::string& padding_mode,
    bool align_corners,
    bool use_precomputed_grid,
    bool batch_output_channels,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::grid_sample(
        input_tensor,
        grid,
        mode,
        padding_mode,
        align_corners,
        use_precomputed_grid,
        batch_output_channels,
        memory_config);
}

}  // namespace ttnn::operations::grid_sample
