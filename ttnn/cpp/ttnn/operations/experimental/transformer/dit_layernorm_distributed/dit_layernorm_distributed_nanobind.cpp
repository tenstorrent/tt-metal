// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_layernorm_distributed_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "dit_layernorm_pre_all_gather.hpp"
#include "dit_layernorm_post_all_gather.hpp"

namespace ttnn::operations::experimental::transformer {

namespace {
void bind_dit_layernorm_pre_all_gather_operation(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::dit_layernorm_pre_allgather,
        R"doc(
            Computes per-row LayerNorm Welford statistics over the last dimension of :attr:`input_tensor`,
            producing a 2-tile-wide tensor with sum(x) and sum(x**2) per row (tile columns 0 and 1).
            Intended to be followed by an all-gather across devices, then ``dit_layernorm_post_allgather``.

            Limitations:
              - Input must be TILE layout, on device, non-sharded.
              - Supported dtypes: BF16, BF8_B, FP32 for input; output stats are BF16.
              - Inputs must be interleaved memory layout.
            )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("dtype") = nb::cast(DataType::BFLOAT16),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("memory_config") = nb::none()});
}

void bind_dit_layernorm_post_all_gather_operation(nb::module_& mod) {
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
}  // namespace

void bind_dit_layernorm_distributed(nb::module_& mod) {
    bind_dit_layernorm_pre_all_gather_operation(mod);
    bind_dit_layernorm_post_all_gather_operation(mod);
}

}  // namespace ttnn::operations::experimental::transformer
