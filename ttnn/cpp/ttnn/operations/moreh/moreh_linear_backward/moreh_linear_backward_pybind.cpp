// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_linear_backward_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_linear_backward/moreh_linear_backward.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {
void bind_moreh_linear_backward_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_linear_backward,
        "Moreh Linear Backward Operation",
        ttnn::pybind_arguments_t{
            py::arg("output_grad"),
            py::arg("input"),
            py::arg("weight"),
            py::arg("are_required_outputs") = std::vector<bool>{true, true, true},

            py::arg("bias") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("weight_grad") = std::nullopt,
            py::arg("bias_grad") = std::nullopt,

            py::arg("input_grad_memory_config") = std::nullopt,
            py::arg("weight_grad_memory_config") = std::nullopt,
            py::arg("bias_grad_memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_linear_backward
