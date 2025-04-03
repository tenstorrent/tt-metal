// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>

#include <tt_stl/span.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
class Shape;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace operations::data_movement {

struct ReshapeOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& logical_shape,
        const ttnn::Shape& padded_shape,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt);

    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& logical_shape,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt);

    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        tt::stl::Span<const int32_t> shape_vector,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt);
};

}  // namespace operations::data_movement

// TODO: unify with ttnn::reshape in core.cpp
constexpr auto reshape_on_device =
    ttnn::register_operation<"ttnn::reshape_on_device", ttnn::operations::data_movement::ReshapeOperation>();

}  // namespace ttnn
