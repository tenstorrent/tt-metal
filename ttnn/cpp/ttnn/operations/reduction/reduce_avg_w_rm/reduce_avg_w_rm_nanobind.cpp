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
    const auto doc = R"doc(Computes the average of elements along the width dimension for a ROW_MAJOR tensor.

This operation performs a fused tilize-reduce-untilize pipeline that:
1. Converts ROW_MAJOR input to TILE_LAYOUT
2. Computes the sum along width with scaler (1/W)
3. Converts back to ROW_MAJOR

The output shape is [N, C, H, 32] where only the first element (index 0) in each width row contains the valid average value.

Args:
    input_tensor: Input tensor with shape [N, C, H, W], must be ROW_MAJOR, INTERLEAVED, BFLOAT16, and tile-aligned (H and W multiples of 32).
    output_mem_config: Optional output memory configuration. Defaults to input's memory config.
    compute_kernel_config: Optional compute kernel configuration.

Returns:
    Tensor with shape [N, C, H, 32] containing row-wise width averages in ROW_MAJOR layout.)doc";

    bind_registered_operation(
        mod,
        ttnn::reduce_avg_w_rm,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("output_mem_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::reduce_avg_w_rm
