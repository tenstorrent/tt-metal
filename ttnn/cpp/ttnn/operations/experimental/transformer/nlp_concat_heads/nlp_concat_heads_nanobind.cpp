// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "nlp_concat_heads.hpp"
#include "ttnn/types.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::nlp_concat_heads::detail {

void bind_nlp_concat_heads(nb::module_& mod) {
    mod.def(
        "nlp_concat_heads",
        &ttnn::experimental::nlp_concat_heads,
        R"doc(
            Shuffles [B, num_heads, S, head_dim] tensor into tensor with shape [B, 1, S, num_heads * head_dim].
        )doc",
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::nlp_concat_heads::detail
