// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh::moreh_layer_norm_backward {
void bind_moreh_layer_norm_backward_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_layer_norm_backward,
        "Moreh Layer Norm Backward Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("output_grad"),
            nb::arg("input"),
            nb::arg("mean"),
            nb::arg("rstd"),
            nb::arg("normalized_dims"),
            nb::kw_only(),
            nb::arg("gamma") = std::nullopt,
            nb::arg("input_grad") = std::nullopt,
            nb::arg("gamma_grad") = std::nullopt,
            nb::arg("beta_grad") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward
