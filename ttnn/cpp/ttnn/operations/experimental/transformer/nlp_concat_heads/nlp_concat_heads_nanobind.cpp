// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads.hpp"

namespace ttnn::operations::experimental::nlp_concat_heads::detail {

void bind_nlp_concat_heads(nb::module_& mod) {
    ttnn::bind_function<"nlp_concat_heads", "ttnn.experimental.">(
        mod,
        R"doc(
            Shuffles [B, num_heads, S, head_dim] tensor into tensor with shape [B, 1, S, num_heads * head_dim].

            Keyword Args:
                memory_config: Memory Config of the output tensor, if `None` then it gets set to input_tensor.memory_config(). Defaults to `None`.
                runtime_rows (bool): interleaved input and output only: one program serves every S, so a change in S is a program cache hit. Defaults to `False`.
        )doc",
        &ttnn::experimental::nlp_concat_heads,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("runtime_rows") = false);
}

}  // namespace ttnn::operations::experimental::nlp_concat_heads::detail
