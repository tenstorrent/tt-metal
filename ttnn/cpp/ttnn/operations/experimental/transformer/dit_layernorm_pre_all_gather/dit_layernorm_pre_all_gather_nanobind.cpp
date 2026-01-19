// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_layernorm_pre_all_gather_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "dit_layernorm_pre_all_gather.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_dit_layernorm_pre_all_gather(nb::module_& mod) {
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

}  // namespace ttnn::operations::experimental::transformer
