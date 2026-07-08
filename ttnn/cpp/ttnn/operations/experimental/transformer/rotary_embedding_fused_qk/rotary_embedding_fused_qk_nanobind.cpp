// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_fused_qk_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "rotary_embedding_fused_qk.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_rotary_embedding_fused_qk(nb::module_& mod) {
    ttnn::bind_function<"rotary_embedding_fused_qk", "ttnn.experimental.">(
        mod,
        R"doc(
        Applies rotary embedding to q and k in a single program (one dispatch), using the same
        GPT-J / rotate_half convention as ttnn.experimental.rotary_embedding. Equivalent to calling
        rotary_embedding(q, cos, sin) and rotary_embedding(k, cos, sin), but fused to halve the
        launch overhead.

        Args:
            q (ttnn.Tensor): the Q tensor [1, num_q_heads, seq, head_dim], TILE, interleaved.
            k (ttnn.Tensor): the K tensor [1, num_kv_heads, seq, head_dim], TILE, interleaved.
            cos_cache (ttnn.Tensor): the Cosine cache [1, 1, seq, head_dim].
            sin_cache (ttnn.Tensor): the Sine cache [1, 1, seq, head_dim].

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor]: the rotated (q, k) tensors.
        )doc",
        &ttnn::experimental::rotary_embedding_fused_qk,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("cos_cache"),
        nb::arg("sin_cache"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer
