// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/create_qkv_heads/create_qkv_heads.hpp"

namespace ttnn::operations::experimental::create_qkv_heads::detail {

void bind_create_qkv_heads(nb::module_& mod) {
    ttnn::bind_function<"create_qkv_heads", "ttnn.experimental.">(
        mod,
        R"doc(
            Splits a [B, 1, Seq_len, H] fused qkv matrix (where H is num_kv_heads * (num_q_heads/num_kv_heads + 2) * head_dim) into a Q tensor [B, num_q_heads, Seq_len, head_dim], K tensor [B, num_kv_heads, Seq_len, head_dim] (with the last two dims transposed if applicable) and V tensor [B, num_kv_heads, Seq_len, head_dim].
        )doc",
        &ttnn::experimental::create_qkv_heads,
        nb::arg("input").noconvert(),
        nb::kw_only(),
        nb::arg("num_heads").noconvert(),
        nb::arg("num_kv_heads").noconvert() = nb::none(),
        nb::arg("transpose_k_heads").noconvert() = true,
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("output_tensors").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::create_qkv_heads::detail
