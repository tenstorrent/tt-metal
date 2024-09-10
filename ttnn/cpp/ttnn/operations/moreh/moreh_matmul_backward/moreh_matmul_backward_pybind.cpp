// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_matmul_backward_pybind.hpp"

#include "pybind11/cast.h"
#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_matmul_backward/moreh_matmul_backward.hpp"

namespace ttnn::operations::moreh::moreh_matmul_backward {
void bind_moreh_matmul_backward_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_matmul_backward,
        "Moreh moreh_matmul_backward Operation",
        ttnn::pybind_arguments_t{
            py::arg("output_grad"),
            py::arg("input_a"),
            py::arg("input_b"),
            py::kw_only(),
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("input_a_grad") = std::nullopt,
            py::arg("input_b_grad") = std::nullopt,
            py::arg("output_mem_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_matmul_backward
