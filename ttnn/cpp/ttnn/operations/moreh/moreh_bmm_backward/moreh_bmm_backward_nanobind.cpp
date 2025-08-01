// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm_backward_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_bmm_backward/moreh_bmm_backward.hpp"

namespace ttnn::operations::moreh::moreh_bmm_backward {
void bind_moreh_bmm_backward_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_bmm_backward,
        "Moreh BMM Backward Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("output_grad"),
            nb::arg("input"),
            nb::arg("mat2"),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("input_grad") = std::nullopt,
            nb::arg("mat2_grad") = std::nullopt,
            nb::arg("input_grad_memory_config") = std::nullopt,
            nb::arg("mat2_grad_memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_bmm_backward
