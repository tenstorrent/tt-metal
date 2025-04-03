// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>

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

struct ExecutePermute {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const SmallVector<int64_t>& dims,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<float>& pad_value = 0.0f);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const SmallVector<int64_t>& dims,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<float>& pad_value = 0.0f);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const SmallVector<int64_t>& dims,
        const std::optional<float>& pad_value = 0.0f);
};

}  // namespace operations::data_movement

constexpr auto permute = ttnn::register_operation<"ttnn::permute", ttnn::operations::data_movement::ExecutePermute>();

}  // namespace ttnn
