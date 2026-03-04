// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_view_nanobind.hpp"

#include <cstdint>
#include <nanobind/nanobind.h>

#include "batch_view.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::deepseek::detail {

namespace nb = nanobind;

void bind_batch_view(nb::module_& mod) {
    const auto* doc = R"doc(
        Creates a view of a single batch from a 3D tensor without copying data.

        This is a zero-copy operation that returns a tensor view pointing to the
        same underlying memory at the specified batch offset.

        Args:
            input_tensor: Input tensor with shape [b, M, N]. Must be:
                - On device (DRAM interleaved, not sharded)
                - 3D tensor
            batch_index: Index of the batch to select (0 <= batch_index < b)

        Returns:
            ttnn.Tensor: A view tensor with shape [M, N] pointing to the selected batch.

        Constraints:
            - For TILE layout: M * N must be divisible by 1024 (bfloat16) or 512 (float32)
            - For ROW_MAJOR layout: always valid

        Example:
            >>> input = ttnn.from_torch(torch.randn(4, 512, 1024), device=device, layout=ttnn.TILE_LAYOUT)
            >>> batch_0 = ttnn.experimental.deepseek.batch_view(input, 0)  # shape [512, 1024]
            >>> batch_1 = ttnn.experimental.deepseek.batch_view(input, 1)  # shape [512, 1024]
        )doc";

    ttnn::bind_function<"batch_view", "ttnn.experimental.deepseek.">(
        mod,
        doc,
        &ttnn::operations::experimental::deepseek::batch_view,
        nb::arg("input_tensor"),
        nb::arg("batch_index"));
}

}  // namespace ttnn::operations::experimental::deepseek::detail
