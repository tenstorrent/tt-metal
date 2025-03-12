// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <reflect>
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::quantization {

struct QuantOp {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const float scale,
        const int32_t zero_point,
        const std::optional<int32_t> axis,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const Tensor& scale,
        const int32_t zero_point,
        const std::optional<int32_t> axis,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

struct RequantOp {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const float in_scale,
        const int32_t in_zero_point,
        const float out_scale,
        const int32_t out_zero_point,
        const std::optional<int32_t> axis,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

struct DequantOp {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const float scale,
        const int32_t zero_point,
        const std::optional<int32_t> axis,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const Tensor& scale,
        const int32_t zero_point,
        const std::optional<int32_t> axis,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

}  // namespace ttnn::operations::quantization

namespace ttnn {
constexpr auto quantize =
    ttnn::register_operation_with_auto_launch_op<"ttnn::quantize", operations::quantization::QuantOp>();
constexpr auto requantize =
    ttnn::register_operation_with_auto_launch_op<"ttnn::requantize", operations::quantization::RequantOp>();
constexpr auto dequantize =
    ttnn::register_operation_with_auto_launch_op<"ttnn::dequantize", operations::quantization::DequantOp>();

}  // namespace ttnn
