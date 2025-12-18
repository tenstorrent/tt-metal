// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward.hpp"

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
            nb::arg("gamma") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("gamma_grad") = nb::none(),
            nb::arg("beta_grad") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward
