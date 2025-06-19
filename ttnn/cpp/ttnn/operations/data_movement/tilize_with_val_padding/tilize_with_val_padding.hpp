// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/tilize_with_val_padding_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/common/queue_id.hpp"
#include "tilize_with_val_padding_common.hpp"

namespace ttnn {

namespace operations::data_movement {

struct ExecuteTilizeWithValPadding {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<uint32_t>& output_padded_shape,
        PadValue pad_value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<uint32_t>& output_padded_shape,
        PadValue pad_value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true);

    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& output_padded_shape,
        PadValue pad_value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& output_padded_shape,
        PadValue pad_value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true);
};

struct ExecuteTilizeWithZeroPadding {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true);
};

}  // namespace operations::data_movement

constexpr auto tilize_with_val_padding = ttnn::
    register_operation<"ttnn::tilize_with_val_padding", ttnn::operations::data_movement::ExecuteTilizeWithValPadding>();

constexpr auto tilize_with_zero_padding = ttnn::register_operation<
    "ttnn::tilize_with_zero_padding",
    ttnn::operations::data_movement::ExecuteTilizeWithZeroPadding>();

}  // namespace ttnn
