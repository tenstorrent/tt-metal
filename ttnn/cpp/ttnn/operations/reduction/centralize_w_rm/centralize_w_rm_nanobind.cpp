// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "centralize_w_rm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "centralize_w_rm.hpp"

namespace ttnn::operations::centralize_w_rm {

void bind_centralize_w_rm_operation(nb::module_& mod) {
    const auto doc = R"doc(Centralizes data by subtracting the row-wise mean from each element.

For each row (along the last dimension), computes the arithmetic mean and subtracts it from all elements in that row.
The result has zero mean along each row.

Args:
    input_tensor (ttnn.Tensor): Input tensor in row-major layout (must be at least 2D)
    memory_config (Optional[ttnn.MemoryConfig]): Output memory configuration

Returns:
    ttnn.Tensor: Centralized tensor with same shape as input

Example:
    >>> input = ttnn.from_torch(torch.randn(32, 64), device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    >>> output = ttnn.centralize_w_rm(input)
    >>> # Each row in output has mean approximately 0)doc";

    bind_registered_operation(
        mod,
        ttnn::centralize_w_rm,
        doc,
        ttnn::nanobind_arguments_t{nb::arg("input_tensor"), nb::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::centralize_w_rm
