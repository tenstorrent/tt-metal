// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"


namespace ttnn {
namespace operations::data_movement {
namespace detail {
ttnn::Tensor host_reshape(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape);
ttnn::Tensor convert_tensor_to_rm_reshape_convert_back_to_orig_layout(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig& memory_config,
    const QueueId queue_id,
    const PadValue& pad_value);
ttnn::Tensor fix_shape_and_perform_reshape_on_2D_RM(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig& memory_config,
    const QueueId queue_id);
ttnn::Tensor fix_shape_and_perform_reshape_on_3D_TILE(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig& memory_config,
    const QueueId queue_id,
    const PadValue& pad_value);
ttnn::Tensor perform_reshape_on_2D_RM(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const MemoryConfig& memory_config,
    const QueueId queue_id);
ttnn::Tensor convert_tile_to_rm(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig& memory_config,
    const QueueId queue_id,
    const PadValue& pad_value);
}

std::pair<ttnn::Shape, ttnn::Shape> shape_corrector(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape);
std::pair<ttnn::Shape, ttnn::Shape> tiling_reshape_corrector(
    const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape);
ttnn::Tensor PerformView(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim);

struct ReshapeViewOperation {
    static ttnn::Tensor invoke(
        const QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& logical_shape,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<PadValue>& pad_value = std::nullopt);
    static ttnn::Tensor invoke(
        const QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& logical_shape,
        const ttnn::Shape& padded_shape,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<PadValue>& pad_value = std::nullopt);
    static ttnn::Tensor invoke(
        const QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        tt::stl::Span<const int32_t> shape_vector,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<PadValue>& pad_value = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto reshape = ttnn::register_operation<"ttnn::reshape", ttnn::operations::data_movement::ReshapeViewOperation>();

}  // namespace ttnn
