// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "topk_large_indices.hpp"

namespace ttnn::operations::experimental::topk_large_indices::detail {

void bind_topk_large_indices(nb::module_& mod) {
    ttnn::bind_function<"topk_large_indices", "ttnn.experimental.">(
        mod,
        R"doc(
        Experimental Top-K over the last dimension of a row-major BFLOAT16 tensor.
        This op is Blackhole-only.

        Returns a ROW_MAJOR UINT32 tensor containing sorted descending top-k indices.
        The output shape matches the input shape except that the last dimension is k.

        This op is intended for large row-major rows. Internally it snaps k to the
        nearest supported LLK size and streams each input row in LLK-sized windows.
        Input values equal to -inf produce the sentinel index 0xFFFFFFFF when they
        survive into the final top-k result.

        K constraints:
            * k must be in [16, 2048];
            * k must be a multiple of 16;
            * the internal LLK window is snapped to 512, 1024, or 2048 elements.

        Input tensor constraints:
            * the input tensor must be allocated on a Blackhole device;
            * rank must be >= 1;
            * all leading dimensions are flattened into independent rows;
            * the flattened leading-dimension row count must fit in uint32_t;
            * the last dimension is the input row length;
            * the flattened row count must be > 0;
            * the last dimension must be >= k and <= 1,073,741,824 elements.

        Args:
            input_tensor: device tensor with ROW_MAJOR layout and BFLOAT16 dtype.
            k: required number of indices to return.
        )doc",
        &ttnn::experimental::topk_large_indices,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("k"));
}

}  // namespace ttnn::operations::experimental::topk_large_indices::detail
