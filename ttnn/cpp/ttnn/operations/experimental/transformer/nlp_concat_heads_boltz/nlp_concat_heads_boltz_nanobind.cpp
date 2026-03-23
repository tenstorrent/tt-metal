// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_boltz_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads_boltz/nlp_concat_heads_boltz.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_nlp_concat_heads_boltz(nb::module_& mod) {
    ttnn::bind_function<"nlp_concat_heads_boltz", "ttnn.experimental.">(
        mod,
        R"doc(
            Shuffles [num_heads, S, S, head_dim] tensor into tensor with shape [1, S, S, num_heads * head_dim].
        )doc",
        &ttnn::experimental::nlp_concat_heads_boltz,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer::detail
