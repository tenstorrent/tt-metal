// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

ttnn::Tensor tilize_with_val_padding(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& output_padded_shape,
    tt::tt_metal::PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> output_dtype = std::nullopt,
    bool use_multicore = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const tt::tt_metal::Tile& tile = {});

ttnn::Tensor tilize_with_val_padding(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& output_padded_shape,
    tt::tt_metal::PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> output_dtype = std::nullopt,
    bool use_multicore = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const tt::tt_metal::Tile& tile = {});

ttnn::Tensor tilize_with_zero_padding(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> output_dtype = std::nullopt,
    bool use_multicore = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const tt::tt_metal::Tile& tile = {});

}  // namespace ttnn::operations::experimental::quasar
