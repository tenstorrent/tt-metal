// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_avg_w_rm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "reduce_avg_w_rm.hpp"

namespace ttnn::operations::reduce_avg_w_rm {

void bind_reduce_avg_w_rm_operation(nb::module_& mod) {
    const auto doc =
        R"doc(Compute the average of all elements along the width (last) dimension of a row-major input tensor.

This operation internally tilizes row-major input, performs width reduction with 1/W scaling,
and untilizes back to row-major format. The fused approach avoids intermediate memory round-trips.

Args:
    input_tensor: Input tensor in ROW_MAJOR layout with shape [N, C, H, W]
    memory_config: Optional output memory configuration
    compute_kernel_config: Optional compute kernel configuration for FP32 accumulation

Returns:
    Output tensor with shape [N, C, H, 32] (physical), where only first element per row is valid (logical width=1))doc";

    bind_registered_operation(
        mod,
        ttnn::reduce_avg_w_rm,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("compute_kernel_config") = std::nullopt,
            nb::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::reduce_avg_w_rm
