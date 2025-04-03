// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>
#include <vector>

#include <tt-metalium/small_vector.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace operations::data_movement {

struct SplitOperation {
    static std::vector<ttnn::Tensor> invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<int64_t>& split_sizes,
        const int64_t dim,
        const std::optional<MemoryConfig>& memory_config_arg);

    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<int64_t>& split_sizes,
        const int64_t dim,
        const std::optional<MemoryConfig>& memory_config);

    static std::vector<ttnn::Tensor> invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const int64_t split_size,
        const int64_t dim,
        const std::optional<MemoryConfig>& memory_config_arg);

    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const int64_t split_size,
        const int64_t dim,
        const std::optional<MemoryConfig>& memory_config);
};

}  // namespace operations::data_movement

constexpr auto split = ttnn::register_operation<"ttnn::split", ttnn::operations::data_movement::SplitOperation>();

}  // namespace ttnn
