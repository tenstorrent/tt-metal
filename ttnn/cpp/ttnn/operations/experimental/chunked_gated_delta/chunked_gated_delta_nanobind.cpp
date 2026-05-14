// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunked_gated_delta_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/chunked_gated_delta/chunked_gated_delta.hpp"

namespace ttnn::operations::experimental::chunked_gated_delta::detail {

void bind_experimental_chunked_gated_delta_operation(nb::module_& mod) {
    ttnn::bind_function<"chunked_gated_delta">(
        mod,
        R"doc(
            Chunked gated delta (GDN) recurrence over precomputed tensors.

            Shapes are inferred from inputs (``total_num_heads = g_exp.shape[0]``):

            - ``g_exp``: ``[total_num_heads, seq_len, 1, 1]``
            - ``factor``: ``[total_num_heads, seq_len, dim_k, dim_k]``
            - ``bktv``: ``[total_num_heads, seq_len, dim_k, dim_v]``
            - ``state``: ``[total_num_heads, 1, dim_k, dim_v]``

            Returns:
                ttnn.Tensor: ``[total_num_heads, seq_len, dim_k, dim_v]`` (boilerplate stub).
        )doc",
        &ttnn::experimental::chunked_gated_delta,
        nb::arg("g_exp"),
        nb::arg("factor"),
        nb::arg("bktv"),
        nb::arg("state"));
}

}  // namespace ttnn::operations::experimental::chunked_gated_delta::detail
