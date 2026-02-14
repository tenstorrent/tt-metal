// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"


namespace ttnn {

enum class TileReshapeMapMode {
    CACHE,
    RECREATE,

};

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

}  // namespace operations::data_movement

// Free function declarations with default parameters
ttnn::Tensor reshape(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& logical_shape,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<PadValue>& pad_value = std::nullopt,
    TileReshapeMapMode reshape_map_mode = TileReshapeMapMode::CACHE,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

ttnn::Tensor reshape(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& logical_input_shape,
    const ttnn::Shape& padded_input_shape,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<PadValue>& pad_value = std::nullopt,
    TileReshapeMapMode reshape_map_mode = TileReshapeMapMode::CACHE,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

ttnn::Tensor reshape(
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const int32_t> shape_vector,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<PadValue>& pad_value = std::nullopt,
    TileReshapeMapMode reshape_map_mode = TileReshapeMapMode::CACHE,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

}  // namespace ttnn
