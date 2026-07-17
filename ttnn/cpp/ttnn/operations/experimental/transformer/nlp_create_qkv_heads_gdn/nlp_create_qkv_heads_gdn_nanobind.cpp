// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_gdn_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_gdn/nlp_create_qkv_heads_gdn.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_nlp_create_qkv_heads_gdn(nb::module_& mod) {
    ttnn::bind_function<"nlp_create_qkv_heads_gdn", "ttnn.experimental.">(
        mod,
        R"doc(
            GDN fork of nlp_create_qkv_heads. Shuffles a fused token-major
            [B, 1, S, (num_q_heads + num_k_heads + num_v_heads) * head_dim] input into head-major
            Q [B, num_q_heads, S, head_dim], K [B, num_k_heads, S, head_dim], V [B, num_v_heads, S, head_dim].
            Q/K/V may each have an independent head count (GDN uses num_q==num_k!=num_v); head_dim is
            shared across Q/K/V and inferred from the fused width. Interleaved TILE in/out, no K-transpose.
        )doc",
        &ttnn::experimental::nlp_create_qkv_heads_gdn,
        nb::arg("input").noconvert(),
        nb::kw_only(),
        nb::arg("num_q_heads").noconvert(),
        nb::arg("num_k_heads").noconvert(),
        nb::arg("num_v_heads").noconvert(),
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("output_tensors").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer::detail
