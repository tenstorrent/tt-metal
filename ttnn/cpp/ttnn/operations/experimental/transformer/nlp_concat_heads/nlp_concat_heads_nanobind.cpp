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
            ``head_split=True`` (interleaved input and output, num_heads > 1, fewer tile rows than cores) splits the
            work per (tile row, head) instead of per tile row so short sequences use the whole grid; the output is the
            same, otherwise it is ignored. Defaults to False.
        )doc",
        &ttnn::experimental::nlp_concat_heads,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("head_split") = false);
}

}  // namespace ttnn::operations::experimental::nlp_concat_heads::detail
