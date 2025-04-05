// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>

#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {

namespace operations::pool {

TensorMemoryLayout determine_pool_config_for_auto_shard(
    uint32_t batch_size,
    uint32_t channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t input_height,
    uint32_t input_width,
    const CoreCoord& compute_grid_size,
    Layout input_tensor_layout,
    const std::array<uint32_t, 2>& kernel_size,
    const DeviceComputeKernelConfig& compute_config,
    const DataType& input_dtype);

}  // namespace operations::pool
}  // namespace ttnn
