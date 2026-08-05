// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "kda_causal_conv_nanobind.hpp"
#include "kda_causal_conv.hpp"
#include "ttnn-nanobind/bind_function.hpp"
namespace ttnn::operations::transformer {
void bind_kda_causal_conv(nb::module_& mod) {
    ttnn::bind_function<"kda_causal_conv1d_split", "ttnn.transformer.">(
        mod,
        R"doc(Four-tap KDA causal convolution with direct tiled Q/K/V outputs.)doc",
        &ttnn::transformer::kda_causal_conv1d_split,
        nb::arg("input").noconvert(),
        nb::arg("state").noconvert(),
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
}  // namespace ttnn::operations::transformer
