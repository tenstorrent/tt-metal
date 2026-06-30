// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_sdpa_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/kv_sdpa/kv_sdpa.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::kv_sdpa {

void bind_kv_sdpa_operation(nb::module_& mod) {
    ttnn::bind_function<"kv_sdpa">(
        mod,
        R"doc(kv_sdpa(q, k, v, *, attn_mask=None, scale=None, compute_kernel_config=None) -> ttnn.Tensor

        Specialized fused-flash scaled-dot-product attention for the small-query MQA case: Q length is
        one tile (32), K/V have a single (or grouped) KV head shared across Q heads, and attention is
        non-causal full attention. One core per Q head runs the transformer-SDPA online-softmax flash
        loop, specialized to this shape.

        Q: [1, NQH, 32, DH]; K/V: [1, NKH, KV, DH] with NKH dividing NQH. Output: [1, NQH, 32, DH].
        attn_mask is accepted for call-site compatibility but treated as a no-op.
        )doc",
        &ttnn::kv_sdpa,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("v"),
        nb::kw_only(),
        nb::arg("attn_mask") = nb::none(),
        nb::arg("scale") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::kv_sdpa
