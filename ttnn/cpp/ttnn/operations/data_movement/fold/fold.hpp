// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/types.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"

#include "device/fold_device_op.hpp"

namespace ttnn {
namespace operations::data_movement {

struct FoldOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor &input_tensor,
                               uint32_t stride_h,
                               uint32_t stride_w,
                               bool use_transpose_as_fold = false,
                               const std::optional<const tt::tt_metal::LegacyShape> &output_shape = std::nullopt,
                               uint32_t pad_c = 0,
                               uint32_t pad_h = 0,
                               uint32_t pad_w = 0,
                               const std::optional<CoreCoord> grid_size = std::nullopt,
                               const std::optional<MemoryConfig> override_memory_config = std::nullopt);
    static ttnn::Tensor invoke(uint8_t queue_id,
                               const ttnn::Tensor &input_tensor,
                               uint32_t stride_h,
                               uint32_t stride_w,
                               bool use_transpose_as_fold = false,
                               const std::optional<const tt::tt_metal::LegacyShape> &output_shape = std::nullopt,
                               uint32_t pad_c = 0,
                               uint32_t pad_h = 0,
                               uint32_t pad_w = 0,
                               const std::optional<CoreCoord> grid_size = std::nullopt,
                               const std::optional<MemoryConfig> override_memory_config = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto fold = register_operation_with_auto_launch_op<"ttnn::fold", operations::data_movement::FoldOperation>();

}  // namespace ttnn
