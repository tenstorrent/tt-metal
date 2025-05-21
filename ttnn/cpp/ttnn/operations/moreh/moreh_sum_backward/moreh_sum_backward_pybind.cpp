// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum_backward_pybind.hpp"

#include "moreh_sum_backward.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::moreh::moreh_sum_backward {
void bind_moreh_sum_backward_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_sum_backward,
        "Moreh Sum Backward Operation",
        ttnn::pybind_arguments_t{
            py::arg("output_grad"),
            py::kw_only(),
            py::arg("input") = std::nullopt,
            py::arg("dim") = std::nullopt,
            py::arg("keepdim") = false,
            py::arg("input_grad") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
        });
}
}  // namespace ttnn::operations::moreh::moreh_sum_backward
