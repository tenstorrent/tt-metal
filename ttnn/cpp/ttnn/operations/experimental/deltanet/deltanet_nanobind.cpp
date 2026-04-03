// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/deltanet/deltanet.hpp"

namespace ttnn::operations::experimental::deltanet::detail {

void bind_deltanet_recurrence(nb::module_& mod) {
    mod.def(
        "deltanet_recurrence",
        &ttnn::experimental::deltanet_recurrence,
        nb::arg("conv_out"),
        nb::arg("b_proj"),
        nb::arg("a_proj"),
        nb::arg("z_proj"),
        nb::arg("dt_bias"),
        nb::arg("A_exp"),
        nb::arg("norm_weight"),
        nb::arg("state"),
        nb::kw_only(),
        nb::arg("num_heads"),
        nb::arg("head_k_dim"),
        nb::arg("head_v_dim"),
        nb::arg("num_k_heads"),
        nb::arg("gqa_ratio"),
        nb::arg("scale"),
        nb::arg("norm_eps"),
        "Performs fused DeltaNet recurrence on device.");
}

}  // namespace ttnn::operations::experimental::deltanet::detail
