// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"

#include "device/pool_op.hpp"

namespace ttnn {
namespace operations::pool {

struct MaxPool2DOp {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        uint32_t batch_size,
        uint32_t input_h,
        uint32_t input_w,
        uint32_t channels,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::array<uint32_t, 2> padding,
        std::array<uint32_t, 2> dilation,
        bool ceil_mode = false,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
        bool in_place_halo = false);
};
struct AvgPool2DOp {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        uint32_t batch_size,
        uint32_t input_h,
        uint32_t input_w,
        uint32_t channels,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::array<uint32_t, 2> padding,
        bool ceil_mode = false,
        bool count_include_pad = true,
        std::optional<int32_t> divisor_override = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
        bool in_place_halo = false);
};

}  // namespace operations::pool

constexpr auto max_pool2d = ttnn::register_operation<"ttnn::max_pool2d", operations::pool::MaxPool2DOp>();
constexpr auto avg_pool2d = ttnn::register_operation<"ttnn::avg_pool2d", operations::pool::AvgPool2DOp>();

}  // namespace ttnn
