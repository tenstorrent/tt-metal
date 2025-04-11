// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_mean_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward.hpp"

namespace ttnn::operations::moreh::moreh_mean_backward {
void bind_moreh_mean_backward_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_mean_backward,
        "Moreh Mean Backward Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("output_grad"),
            nb::kw_only(),
            nb::arg("dim"),
            nb::arg("keepdim"),
            nb::arg("input_grad_shape") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_mean_backward
