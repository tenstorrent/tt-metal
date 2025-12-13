// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm/moreh_layer_norm.hpp"

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
            nb::arg("gamma") = nb::none(),
            nb::arg("beta") = nb::none(),
            nb::kw_only(),
            nb::arg("output") = nb::none(),
            nb::arg("mean") = nb::none(),
            nb::arg("rstd") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm
