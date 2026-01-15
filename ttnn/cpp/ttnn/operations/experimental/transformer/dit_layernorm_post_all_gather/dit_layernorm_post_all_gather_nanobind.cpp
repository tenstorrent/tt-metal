// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_layernorm_post_all_gather_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "dit_layernorm_post_all_gather.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_dit_layernorm_post_all_gather(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::dit_layernorm_post_allgather,
        R"doc(
            Applies LayerNorm using gathered Welford statistics. Expects stats to contain
            sum(x) and sum(x**2) tile columns per device (2 tile columns per device).

            Limitations:
              - Inputs and stats must be TILE layout, on device, non-sharded.
              - Supported input dtypes: BF16, BF8_B, FP32. Stats must be BF16.
              - Stats last padded dim must be a multiple of 64 (2 tiles).
              - If gamma is provided, beta must also be provided; both must match input hidden dim.
            )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("stats"),
            nb::kw_only(),
            nb::arg("epsilon") = 1e-5,
            nb::arg("weight") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("dtype") = nb::none()});
}

}  // namespace ttnn::operations::experimental::transformer
