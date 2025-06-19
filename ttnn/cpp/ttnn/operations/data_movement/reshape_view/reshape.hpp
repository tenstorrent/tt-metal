// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"


namespace ttnn {
namespace operations::data_movement {

std::pair<ttnn::Shape, ttnn::Shape> shape_corrector(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape);
std::pair<ttnn::Shape, ttnn::Shape> tiling_reshape_corrector(
    const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape);
ttnn::Tensor PerformView(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    uint32_t tile_first_dim,
    uint32_t tile_second_dim);

struct ReshapeViewOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& logical_shape,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<PadValue>& pad_value = std::nullopt);
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& logical_shape,
        const ttnn::Shape& padded_shape,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<PadValue>& pad_value = std::nullopt);
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        tt::stl::Span<const int32_t> shape_vector,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<PadValue>& pad_value = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto reshape = ttnn::register_operation<"ttnn::reshape", ttnn::operations::data_movement::ReshapeViewOperation>();

}  // namespace ttnn
