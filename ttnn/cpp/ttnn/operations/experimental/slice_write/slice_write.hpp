// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace experimental {

struct SliceWriteOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& output_tensor,
        const ttnn::SmallVector<uint32_t>& output_tensor_start,
        const ttnn::SmallVector<uint32_t>& output_tensor_end,
        const ttnn::SmallVector<uint32_t>& step);
};

}  // namespace experimental
}  // namespace operations
}  // namespace ttnn

namespace ttnn::experimental {
constexpr auto slice_write =
    ttnn::register_operation<"ttnn::slice_write", ttnn::operations::experimental::SliceWriteOperation>();
}
