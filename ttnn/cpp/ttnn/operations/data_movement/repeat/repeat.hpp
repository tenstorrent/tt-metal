// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>

#include <tt-metalium/small_vector.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {
namespace tt_metal {
class Shape;
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace operations::data_movement {

struct RepeatOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<uint32_t>& repetition_vector,
        const std::optional<MemoryConfig>& provided_output_mem_config,
        QueueId queue_id);

    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::Shape& repeat_dims);
};

}  // namespace operations::data_movement

constexpr auto repeat = ttnn::register_operation<"ttnn::repeat", ttnn::operations::data_movement::RepeatOperation>();

}  // namespace ttnn
