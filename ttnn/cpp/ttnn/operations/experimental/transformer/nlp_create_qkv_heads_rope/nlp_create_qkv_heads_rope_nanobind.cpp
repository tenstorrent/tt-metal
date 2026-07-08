// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_rope_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "nlp_create_qkv_heads_rope.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_nlp_create_qkv_heads_rope(nb::module_& mod) {
    ttnn::bind_function<"nlp_create_qkv_heads_rope", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused create-qkv-heads + q/k RoPE in a single dispatch. Splits a fused QKV tensor
        [1, 1, seq, (num_q_heads + 2*num_kv_heads)*head_dim] into q/k/v heads and applies RoPE
        (GPT-J/rotate_half convention, identical to ttnn.experimental.rotary_embedding) to q and k.
        v is returned un-rotated. Requires seq <= one tile (Ht == 1), transpose_k_heads == False.

        Args:
            qkv (ttnn.Tensor): fused QKV [1, 1, seq, (num_q_heads + 2*num_kv_heads)*head_dim], TILE, interleaved.
            cos_cache (ttnn.Tensor): [1, 1, seq, head_dim].
            sin_cache (ttnn.Tensor): [1, 1, seq, head_dim].
            num_q_heads (int): number of query heads.
            num_kv_heads (int): number of key/value heads.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]: (q_rotated, k_rotated, v).
        )doc",
        &ttnn::experimental::nlp_create_qkv_heads_rope,
        nb::arg("qkv"),
        nb::arg("cos_cache"),
        nb::arg("sin_cache"),
        nb::arg("num_q_heads"),
        nb::arg("num_kv_heads"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer
