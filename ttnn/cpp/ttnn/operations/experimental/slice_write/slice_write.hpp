// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental {

struct SliceWriteOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        ttnn::Tensor& output_tensor,
        const ttnn::SmallVector<uint32_t>& begins,
        const ttnn::SmallVector<uint32_t>& ends,
        const ttnn::SmallVector<uint32_t>& step);
};

}  // namespace ttnn::operations::experimental

namespace ttnn::experimental {
constexpr auto slice_write =
    ttnn::register_operation<"ttnn::slice_write", ttnn::operations::experimental::SliceWriteOperation>();
}
