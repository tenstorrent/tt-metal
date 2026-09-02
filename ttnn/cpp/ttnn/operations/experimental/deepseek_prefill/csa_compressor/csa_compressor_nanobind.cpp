// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "csa_compressor_nanobind.hpp"

#include "csa_compressor.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_csa_compressor(::nanobind::module_& mod) {
    namespace nb = ::nanobind;
    ttnn::bind_function<"csa_compressor", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Fused Blaze-compatible compressed sparse-attention pooling.

            ``kv`` and ``gate`` are local BFLOAT16 TILE slabs shaped
            ``[1, 1, S_local, 1024]``. The replicated position bias has shape
            ``[1, 1, 4, 1024]`` and both temporal states have local shape
            ``[1, 1, 64, 512]``.

            Returns the local pooled slab and authoritative local KV and score
            states. Incomplete and padded compression windows are zero.
        )doc",
        &csa_compressor::csa_compressor,
        nb::arg("kv").noconvert(),
        nb::arg("gate").noconvert(),
        nb::arg("position_bias").noconvert(),
        nb::arg("initial_kv_state").noconvert(),
        nb::arg("initial_score_state").noconvert(),
        nb::kw_only(),
        nb::arg("seq_len_actual"),
        nb::arg("first_token_position") = 0,
        nb::arg("cluster_axis") = 0,
        nb::arg("topology").noconvert() = ::ttnn::ccl::Topology::Linear);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
