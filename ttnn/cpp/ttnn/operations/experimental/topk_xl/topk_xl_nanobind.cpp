// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_xl_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "topk_xl.hpp"

namespace ttnn::operations::experimental::topk_xl {

ttnn::Tensor topk_xl_func(const ttnn::Tensor& input_tensor, uint32_t k, bool largest, bool sorted) {
    auto [operation_attributes, tensor_args] = TopkXLDeviceOperation::invoke(input_tensor, k, largest, sorted);
    return ttnn::device_operation::launch<TopkXLDeviceOperation>(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::experimental::topk_xl

namespace ttnn::operations::experimental::topk_xl::detail {

void bind_topk_xl(nb::module_& mod) {
    ttnn::bind_function<"topk_xl", "ttnn.experimental.">(
        mod,
        R"doc(
        Experimental row-major Top-K XL for UINT32 row-major indices.

        Args:
            input_tensor: row-major BFLOAT16 tensor. Reduction is over the last dimension.
            k: number of top elements. Must be <= 2048 and a multiple of 16.
            largest: must be true.
            sorted: must be true.
        )doc",
        &ttnn::operations::experimental::topk_xl::topk_xl_func,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("k") = 1024,
        nb::arg("largest") = true,
        nb::arg("sorted") = true);
}

}  // namespace ttnn::operations::experimental::topk_xl::detail
