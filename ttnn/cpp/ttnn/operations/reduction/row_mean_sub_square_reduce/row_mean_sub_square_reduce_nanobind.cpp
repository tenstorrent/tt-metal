// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "row_mean_sub_square_reduce_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "row_mean_sub_square_reduce.hpp"

namespace ttnn::operations::row_mean_sub_square_reduce {

void bind_row_mean_sub_square_reduce_operation(nb::module_& mod) {
    const auto doc = R"doc(Computes variance along the width (W) dimension of a 4D tensor.

This operation computes variance as E[(x - E[x])^2] for each (N, C, H) position.
The output has width padded to TILE_WIDTH=32.

Args:
    input_tensor: Input tensor in ROW_MAJOR layout with shape [N, C, H, W]
    memory_config: Memory configuration for output tensor (defaults to input memory config)
    output_dtype: Output data type (defaults to input dtype)

Returns:
    Tensor with shape [N, C, H, 32] containing variance values in the first element of each row)doc";

    bind_registered_operation(
        mod,
        ttnn::row_mean_sub_square_reduce,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"), nb::arg("output_dtype") = std::nullopt, nb::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::row_mean_sub_square_reduce
