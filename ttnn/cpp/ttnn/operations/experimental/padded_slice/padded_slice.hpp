// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace experimental {

struct PaddedSliceOperation {
    template <typename T>
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        tt::stl::Span<const T> begins,
        tt::stl::Span<const T> ends,
        tt::stl::Span<const T> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<float>& pad_value = std::nullopt);

    template <typename T>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        tt::stl::Span<const T> output_tensor_start,
        tt::stl::Span<const T> output_tensor_end,
        tt::stl::Span<const T> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<float>& pad_value = std::nullopt);

    template <typename T>
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<T>& begins,
        const ttnn::SmallVector<T>& ends,
        const ttnn::SmallVector<T>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<float>& pad_value = std::nullopt) {
        return invoke(
            queue_id,
            input_tensor,
            tt::stl::Span<const T>(begins),
            tt::stl::Span<const T>(ends),
            tt::stl::Span<const T>(step),
            memory_config_arg,
            optional_output_tensor,
            pad_value);
    }

    template <typename T>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<T>& begins,
        const ttnn::SmallVector<T>& ends,
        const ttnn::SmallVector<T>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<float>& pad_value = std::nullopt) {
        return invoke(
            input_tensor,
            tt::stl::Span<const T>(begins),
            tt::stl::Span<const T>(ends),
            tt::stl::Span<const T>(step),
            memory_config_arg,
            optional_output_tensor,
            pad_value);
    }

    template <typename T, std::size_t N>
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const std::array<T, N>& output_tensor_start,
        const std::array<T, N>& output_tensor_end,
        const std::array<T, N>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<float>& pad_value = std::nullopt);

    template <typename T, std::size_t N>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::array<T, N>& output_tensor_start,
        const std::array<T, N>& output_tensor_end,
        const std::array<T, N>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<float>& pad_value = std::nullopt);
};

}  // namespace experimental
}  // namespace operations
}  // namespace ttnn

namespace ttnn::experimental {
constexpr auto padded_slice =
    ttnn::register_operation<"ttnn::padded_slice", ttnn::operations::experimental::PaddedSliceOperation>();
}
