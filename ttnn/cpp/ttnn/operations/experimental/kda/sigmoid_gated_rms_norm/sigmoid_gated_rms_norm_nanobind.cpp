// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sigmoid_gated_rms_norm_nanobind.hpp"
#include "sigmoid_gated_rms_norm.hpp"

#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::sigmoid_gated_rms_norm::detail {

void bind_sigmoid_gated_rms_norm(nb::module_& mod) {
    ttnn::bind_function<"sigmoid_gated_rms_norm", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Fused per-head RMSNorm and sigmoid gate for tile-aligned KDA prefill.
        Input [B*H,T,V], gate [B,T,H*V], and weight [V] produce [B,T,H*V].
        )doc",
        &ttnn::experimental::kda::sigmoid_gated_rms_norm,
        nb::arg("input").noconvert(),
        nb::arg("gate").noconvert(),
        nb::arg("weight").noconvert(),
        nb::arg("num_heads"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-5f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_dtype") = ttnn::DataType::FLOAT32);
}

}  // namespace ttnn::operations::experimental::kda::sigmoid_gated_rms_norm::detail
