// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "qkv_causal_conv1d_silu_nanobind.hpp"
#include "qkv_causal_conv1d_silu.hpp"
#include "ttnn-nanobind/bind_function.hpp"
namespace ttnn::operations::experimental::kda::qkv_causal_conv1d_silu::detail {
void bind_qkv_causal_conv1d_silu(nb::module_& mod) {
    ttnn::bind_function<"qkv_causal_conv1d_silu", "ttnn.experimental.kda.">(
        mod,
        R"doc(Four-tap KDA causal convolution with direct tiled Q/K/V outputs.)doc",
        &ttnn::experimental::kda::qkv_causal_conv1d_silu,
        nb::arg("input").noconvert(),
        nb::arg("history").noconvert(),
        nb::arg("tap0").noconvert(),
        nb::arg("tap1").noconvert(),
        nb::arg("tap2").noconvert(),
        nb::arg("tap3").noconvert(),
        nb::arg("q_width"),
        nb::arg("k_width"),
        nb::arg("v_width"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}
}  // namespace ttnn::operations::experimental::kda::qkv_causal_conv1d_silu::detail
