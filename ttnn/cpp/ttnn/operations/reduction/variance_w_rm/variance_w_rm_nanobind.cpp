// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "variance_w_rm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "variance_w_rm.hpp"

namespace ttnn::operations::variance_w_rm {

void bind_variance_w_rm_operation(nb::module_& mod) {
    const auto doc = R"doc(Computes the variance along the width dimension for row-major tensors.

For each row (along the last dimension), computes the variance as the mean of squared deviations from the mean.
Uses population variance formula (divide by N, not N-1).

Args:
    input_tensor (ttnn.Tensor): Input tensor in row-major layout (must be at least 2D)
    memory_config (Optional[ttnn.MemoryConfig]): Output memory configuration

Returns:
    ttnn.Tensor: Variance tensor with shape [..., 1] (logical), padded to [..., 32]

Example:
    >>> input = ttnn.from_torch(torch.randn(32, 64), device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    >>> output = ttnn.variance_w_rm(input)
    >>> # output shape: [32, 1] logical, [32, 32] padded)doc";

    bind_registered_operation(
        mod,
        ttnn::variance_w_rm,
        doc,
        ttnn::nanobind_arguments_t{nb::arg("input_tensor"), nb::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::variance_w_rm
