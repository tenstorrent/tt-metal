// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "nlp_create_qkv_heads.hpp"

namespace ttnn::operations::bringup::detail {

void bind_nlp_create_qkv_heads(nb::module_& mod) {
    ttnn::bind_function<"nlp_create_q_heads_split", "ttnn.bringup.">(
        mod,
        "Create Q heads and return two contiguous channel regions. Interleaved TILE input; split and head width must "
        "be tile aligned. Fork of ttnn.experimental.nlp_create_q_heads_split (ttnn/ttnn/bringup/nlp_create_qkv_heads). "
        "max_grid (ttnn.CoreCoord, optional; bring-up option, default None = the device's whole compute grid): the "
        "work split runs on the rectangle (0, 0) .. (max_grid - 1) of the grid (deepseek_v41_flash_d_p FIX.15: a "
        "Galaxy's 110-core cap).",
        &ttnn::experimental::bringup::nlp_create_q_heads_split,
        nb::arg("input").noconvert(),
        nb::kw_only(),
        nb::arg("num_heads").noconvert(),
        nb::arg("split_head_dim").noconvert(),
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("max_grid") = nb::none());
    ttnn::bind_function<"nlp_create_qkv_heads", "ttnn.bringup.">(
        mod,
        R"doc(
            Shuffles [B, 1, S, 3 * head_dim * num_heads] fused qkv matrix into 3 Q, K, and V heads with shapes [B, num_heads, S, head_dim], [B, num_kv_heads, head_dim, S], and [B, num_kv_heads, S, head_dim]. If optional ``input_kv`` tensor is provided, K and V will be created from ``input_kv`` and ``input`` should have shape [B, 1, S, head_dim * num_heads] instead. ``num_kv_heads`` defaults to ``num_heads`` if not provided. An additional transpose along the last two dims is performed by default for K heads, but this can be skipped with ``transpose_k_heads=false``. With ``kv_tied=true`` the input carries one K/V section instead of two and V is read from the same columns as K. In fused mode this width must be unambiguous for tied layout and not simultaneously match the untied section count.
        )doc",
        &ttnn::experimental::bringup::nlp_create_qkv_heads,
        nb::arg("input").noconvert(),
        nb::arg("input_kv").noconvert() = nb::none(),
        nb::kw_only(),
        nb::arg("num_heads").noconvert(),
        nb::arg("num_kv_heads").noconvert() = nb::none(),
        nb::arg("transpose_k_heads").noconvert() = true,
        nb::arg("kv_tied").noconvert() = false,
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("output_tensors").noconvert() = nb::none());
}

}  // namespace ttnn::operations::bringup::detail
