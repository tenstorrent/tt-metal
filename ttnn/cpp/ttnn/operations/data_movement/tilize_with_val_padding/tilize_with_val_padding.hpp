// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/tilize_with_val_padding_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {

namespace operations::data_movement {

struct ExecuteTilizeWithValPadding {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<uint32_t>& output_padded_shape,
        tt::tt_metal::PadValue pad_value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& output_padded_shape,
        tt::tt_metal::PadValue pad_value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

struct ExecuteTilizeWithZeroPadding {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto tilize_with_val_padding = ttnn::
    register_operation<"ttnn::tilize_with_val_padding", ttnn::operations::data_movement::ExecuteTilizeWithValPadding>();

constexpr auto tilize_with_zero_padding = ttnn::register_operation<
    "ttnn::tilize_with_zero_padding",
    ttnn::operations::data_movement::ExecuteTilizeWithZeroPadding>();

}  // namespace ttnn
