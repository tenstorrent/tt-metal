// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_vit_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/nlp_create_qkv_heads_vit.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_nlp_create_qkv_heads_vit(nb::module_& mod) {
    ttnn::bind_function<"nlp_create_qkv_heads_vit", "ttnn.experimental.">(
        mod,
        R"doc(
            Shuffles [B, 1, S, 2304] fused qkv matrix into 3 heads with shapes [B, 12, S, 64], [B, 12, S, 64], and [B, 12, S, 64].
        )doc",
        &ttnn::experimental::nlp_create_qkv_heads_vit,
        nb::arg("input").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensors") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer::detail
