// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_nlp_create_qkv_heads(nb::module_& mod) {
    ttnn::bind_function<"nlp_create_qkv_heads", "ttnn.experimental.">(
        mod,
        R"doc(
             Shuffles [B, 1, S, 3 * head_dim * num_heads] fused qkv matrix into 3 Q, K, and V heads with shapes [B, num_heads, S, head_dim], [B, num_kv_heads, head_dim, S], and [B, num_kv_heads, S, head_dim]. If optional ``input_kv`` tensor is provided, K and V will be created from ``input_kv`` and ``input`` should have shape [B, 1, S, head_dim * num_heads] instead. ``num_kv_heads`` defaults to ``num_heads`` if not provided. An additional transpose along the last two dims is performed by default for K heads, but this can be skipped with ``transpose_k_heads=false``.
        )doc",
        &ttnn::experimental::nlp_create_qkv_heads,
        nb::arg("input").noconvert(),
        nb::arg("input_kv").noconvert() = nb::none(),
        nb::kw_only(),
        nb::arg("num_heads").noconvert(),
        nb::arg("num_kv_heads").noconvert() = nb::none(),
        nb::arg("transpose_k_heads").noconvert() = true,
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("output_tensors").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer::detail
