// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/gated_delta_attn/gated_delta_attn_preprocess_nanobind.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/transformer/gated_delta_attn/gated_delta_attn_preprocess.hpp"

namespace ttnn::operations::transformer {

void bind_gated_delta_attn_preprocess(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fused preprocessing for `ttnn.transformer.gated_delta_attn_seq`.

        Inputs are already padded/chunkable head-major tensors:
            q/k/v: [BH, L, 128], beta: [BH, L, 1], g: [BH, L]

        Returns the eight scan inputs:
            L_unit, v_beta_sc, k_bd_sc, intra_attn, q_decay, k_decay_t, dl_exp, L_inv
        )doc";

    ttnn::bind_function<"gated_delta_attn_preprocess", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::gated_delta_attn_preprocess,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("triu_ones").noconvert(),
        nb::arg("tril_mask").noconvert(),
        nb::arg("eye").noconvert(),
        nb::arg("lower_causal").noconvert(),
        nb::arg("eye_32").noconvert(),
        nb::kw_only(),
        nb::arg("chunk_size") = 128,
        nb::arg("diag_alpha") = 0.25f,
        nb::arg("bf16_value_path") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
