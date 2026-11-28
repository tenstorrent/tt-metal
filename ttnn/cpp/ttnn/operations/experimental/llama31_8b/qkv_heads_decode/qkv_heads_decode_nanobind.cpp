// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qkv_heads_decode_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "qkv_heads_decode.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::experimental::llama31_8b {

void bind_qkv_heads_decode(nb::module_& llama31_8b) {
    const auto* doc = R"doc(
        Splits the fused QKV decode activations into Q, K and V heads, for Llama-3.1-8B decode on Blackhole.

        Same inputs, outputs and output layouts as ``ttnn.experimental.nlp_create_qkv_heads_decode`` with
        ``overlap_qk_coregrid=True``; outputs are bit-identical to it. Other layouts fall back to that op's program.

        Args:
            input (ttnn.Tensor): [1, 1, batch, (num_heads + 2 * num_kv_heads) * head_dim], TILE.
            num_heads (int): number of query heads.
            num_kv_heads (int): number of key / value heads.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): its shard grid places the outputs.

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]: Q, K and V heads.
        )doc";

    ttnn::bind_function<"qkv_heads_decode", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        doc,
        &qkv_heads_decode,
        nb::arg("input"),
        nb::arg("num_heads"),
        nb::arg("num_kv_heads"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::experimental::llama31_8b
