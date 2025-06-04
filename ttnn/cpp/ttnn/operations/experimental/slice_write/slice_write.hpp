// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace experimental {

struct SliceWriteOperation {
    template <typename T, std::size_t N>
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& output_tensor,
        const std::array<T, N>& output_tensor_start,
        const std::array<T, N>& output_tensor_end,
        const std::array<T, N>& step);
};

}  // namespace experimental
}  // namespace operations
}  // namespace ttnn

namespace ttnn::experimental {
constexpr auto slice_write =
    ttnn::register_operation<"ttnn::slice_write", ttnn::operations::experimental::SliceWriteOperation>();
}
