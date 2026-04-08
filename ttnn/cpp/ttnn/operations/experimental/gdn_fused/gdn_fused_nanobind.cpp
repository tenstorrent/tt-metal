// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gdn_fused_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/gdn_fused/gdn_fused.hpp"

namespace ttnn::operations::experimental::gdn_fused::detail {

void bind_experimental_gdn_fused_operations(nb::module_& mod) {
    const auto* gdn_fused_doc =
        R"doc(
         Fused GDN (Gated Delta Network) kernel operation.

         This operation performs the full fused GDN computation including
         Q/K/V extraction from conv_out, state update, and output generation.
         It is an in-place operation that modifies state and writes to the
         pre-allocated output tensor.

         Positional Arguments:
             conv_out (ttnn.Tensor): Batched [1, B, qkv_dim_tp] conv output.
             a_fused (ttnn.Tensor): Scalar a in [1, B, Nv_TP].
             b_fused (ttnn.Tensor): Scalar b in [1, B, Nv_TP].
             neg_exp_A (ttnn.Tensor): Constant neg_exp_A in [1, 1, Nv_TP].
             dt_bias (ttnn.Tensor): Constant dt_bias in [1, 1, Nv_TP].
             norm_w (ttnn.Tensor): RMS norm weight.
             scale_tt (ttnn.Tensor): Scale tensor.
             rms_scale_tt (ttnn.Tensor): RMS scale tensor.
             rms_eps_tt (ttnn.Tensor): RMS epsilon tensor.
             state (ttnn.Tensor): State tensor (modified in-place).
             output (ttnn.Tensor): Pre-allocated output tensor.

         Keyword Args:
             num_pairs (int): Total number of head pairs to process.
             num_cores (int): Number of cores to use (default 40).
             Nv_TP (int): Number of value heads per tensor-parallel shard (default 12).
             Nk_TP (int): Number of key heads per tensor-parallel shard (default 4).
             repeat_factor (int): GQA repeat factor (default 3).
             key_dim_tp (int): Key dimension per tensor-parallel shard (default 512).

         Returns:
             ttnn.Tensor: The output tensor (same as the input output tensor).
        )doc";

    ttnn::bind_function<"gdn_fused", "ttnn.experimental.">(
        mod,
        gdn_fused_doc,
        &ttnn::experimental::gdn_fused,
        nb::arg("conv_out").noconvert(),
        nb::arg("a_fused").noconvert(),
        nb::arg("b_fused").noconvert(),
        nb::arg("neg_exp_A").noconvert(),
        nb::arg("dt_bias").noconvert(),
        nb::arg("norm_w").noconvert(),
        nb::arg("scale_tt").noconvert(),
        nb::arg("rms_scale_tt").noconvert(),
        nb::arg("rms_eps_tt").noconvert(),
        nb::arg("state").noconvert(),
        nb::arg("output").noconvert(),
        nb::kw_only(),
        nb::arg("num_pairs"),
        nb::arg("num_cores") = 40,
        nb::arg("Nv_TP") = 12,
        nb::arg("Nk_TP") = 4,
        nb::arg("repeat_factor") = 3,
        nb::arg("key_dim_tp") = 512);
}

}  // namespace ttnn::operations::experimental::gdn_fused::detail
