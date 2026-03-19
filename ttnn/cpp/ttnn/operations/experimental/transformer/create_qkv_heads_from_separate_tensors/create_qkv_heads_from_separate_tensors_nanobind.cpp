// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_from_separate_tensors_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "create_qkv_heads_from_separate_tensors.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_create_qkv_heads_from_separate_tensors(nb::module_& mod) {
    ttnn::bind_function<"create_qkv_heads_from_separate_tensors", "ttnn.experimental.">(
        mod,
        R"doc(
            Splits a [B, 1, Seq_len, H] q matrix and fused kv matrix (where H is num_q_heads * head_dim for q and num_kv_heads * head_dim * 2 for kv) into a Q tensor [B, num_q_heads, Seq_len, head_dim], K tensor [B, num_kv_heads, Seq_len, head_dim] (with the last two dims transposed if applicable) and V tensor [B, num_kv_heads, Seq_len, head_dim].
        )doc",
        &ttnn::experimental::create_qkv_heads_from_separate_tensors,
        nb::arg("input").noconvert(),
        nb::arg("input_kv").noconvert(),
        nb::kw_only(),
        nb::arg("num_heads").noconvert(),
        nb::arg("num_kv_heads").noconvert() = nb::none(),
        nb::arg("transpose_k_heads").noconvert() = true,
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("output_tensors").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer::detail
