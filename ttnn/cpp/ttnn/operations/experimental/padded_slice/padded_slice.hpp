// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
        const MemoryConfig& memory_config_arg,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<float>& pad_value = std::nullopt);

    template <typename T>
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<T>& begins,
        const ttnn::SmallVector<T>& ends,
        const ttnn::SmallVector<T>& step,
        const MemoryConfig& memory_config_arg,
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
};

}  // namespace experimental
}  // namespace operations
}  // namespace ttnn

namespace ttnn::experimental {
constexpr auto padded_slice =
    ttnn::register_operation<"ttnn::padded_slice", ttnn::operations::experimental::PaddedSliceOperation>();
}
