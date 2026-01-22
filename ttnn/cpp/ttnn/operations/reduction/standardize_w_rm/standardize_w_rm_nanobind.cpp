// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "standardize_w_rm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "standardize_w_rm.hpp"

namespace ttnn::operations::standardize_w_rm {

void bind_standardize_w_rm_operation(nb::module_& mod) {
    const auto doc = R"doc(Performs row-wise standardization (z-score normalization) on row-major interleaved tensors.

For each row of width W, computes:
1. Mean across the row
2. Centralization (subtract mean)
3. Variance (mean of squared deviations)
4. Reciprocal square root of (variance + epsilon)
5. Standardized output = centralized * rsqrt(variance + epsilon)

Args:
    input_tensor: Input tensor in ROW_MAJOR layout (at least 2D)
    epsilon: Small constant for numerical stability (default: 1e-5)
    memory_config: Output memory configuration

Returns:
    Tensor with same shape as input, standardized along last dimension)doc";

    bind_registered_operation(
        mod,
        ttnn::standardize_w_rm,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"), nb::arg("epsilon") = 1e-5, nb::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::standardize_w_rm
