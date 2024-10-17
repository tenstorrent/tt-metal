// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/experimental/pool/avgpool/avg_pool.hpp"

namespace ttnn {
namespace operations::experimental::pool {

struct AveragePool2DOp {
    static Tensor invoke(
        const Tensor& input_tensor,
        uint32_t batch_size,
        uint32_t input_h,
        uint32_t input_w,
        uint32_t channels,
        const std::array<uint32_t, 2>& kernel_size,
        const std::array<uint32_t, 2>& stride,
        const std::array<uint32_t, 2>& padding,
        bool ceil_mode = false,
        bool count_include_pad = true,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<DataType>& output_dtype = std::nullopt,
        uint8_t queue_id = 0);
};

}  // namespace operations::experimental::pool

namespace experimental {

constexpr auto avg_pool2d = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::avg_pool2d",
    ttnn::operations::experimental::pool::AveragePool2DOp>();

}  // namespace experimental

}  // namespace ttnn
