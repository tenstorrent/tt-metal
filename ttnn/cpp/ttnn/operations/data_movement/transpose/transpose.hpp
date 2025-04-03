// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>

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

struct ExecuteTranspose {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const int64_t& dim1,
        const int64_t& dim2,
        const std::optional<MemoryConfig>& memory_config_arg,
        const std::optional<float>& pad_value = 0.0f);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int64_t& dim1,
        const int64_t& dim2,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<float>& pad_value = 0.0f);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int64_t& dim1,
        const int64_t& dim2,
        const std::optional<float>& pad_value = 0.0f);
};

}  // namespace operations::data_movement

constexpr auto transpose =
    ttnn::register_operation<"ttnn::transpose", ttnn::operations::data_movement::ExecuteTranspose>();

}  // namespace ttnn
