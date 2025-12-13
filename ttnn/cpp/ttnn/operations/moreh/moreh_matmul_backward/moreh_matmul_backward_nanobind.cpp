// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_matmul_backward_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_matmul_backward/moreh_matmul_backward.hpp"

namespace ttnn::operations::moreh::moreh_matmul_backward {
void bind_moreh_matmul_backward_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_matmul_backward,
        "Moreh Matmul Backward Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("output_grad"),
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("input_a_grad") = nb::none(),
            nb::arg("input_b_grad") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_matmul_backward
