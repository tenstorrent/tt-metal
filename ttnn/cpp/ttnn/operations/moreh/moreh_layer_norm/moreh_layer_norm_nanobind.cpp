// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "ttnn/operations/moreh/moreh_layer_norm/moreh_layer_norm.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh::moreh_layer_norm {
void bind_moreh_layer_norm_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_layer_norm,
        "Moreh Layer Norm Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::arg("normalized_dims"),
            nb::arg("eps") = 1e-5f,
            nb::arg("gamma") = std::nullopt,
            nb::arg("beta") = std::nullopt,
            nb::kw_only(),
            nb::arg("output") = std::nullopt,
            nb::arg("mean") = std::nullopt,
            nb::arg("rstd") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm
