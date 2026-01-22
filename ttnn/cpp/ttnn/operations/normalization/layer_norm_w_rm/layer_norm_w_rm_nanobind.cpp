// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm_w_rm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "layer_norm_w_rm.hpp"

namespace ttnn::operations::layer_norm_w_rm {

void bind_layer_norm_w_rm_operation(nb::module_& mod) {
    const auto doc = R"doc(Layer normalization across the last dimension (W) with learnable affine transformation. Computes mean and variance along W, standardizes values, then applies gamma (scale) and beta (shift): output = (input - mean) / sqrt(variance + epsilon) * gamma + beta)doc";

    // Force the operation constant to be used/instantiated
    constexpr auto& op_ref = ttnn::layer_norm_w_rm;
    (void)op_ref;

    bind_registered_operation(
        mod,
        ttnn::layer_norm_w_rm,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("gamma"),
            nb::arg("beta"),
            nb::arg("epsilon") = 1e-5f,
            nb::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::layer_norm_w_rm