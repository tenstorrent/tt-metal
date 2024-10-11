// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_mean_backward_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward.hpp"

namespace ttnn::operations::moreh::moreh_mean_backward {
void bind_moreh_mean_backward_operation(py::module &module) {
    bind_registered_operation(
        module,
        ttnn::moreh_mean_backward,
        "Moreh Mean Backward Operation",
        ttnn::pybind_arguments_t{
            py::arg("output_grad"),
            py::kw_only(),
            py::arg("dim"),
            py::arg("keepdim"),
            py::arg("input_grad_shape") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_mean_backward
