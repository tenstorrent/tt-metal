// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_pybind.hpp"

#include "moreh_norm.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::moreh::moreh_norm {
void bind_moreh_norm_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_norm,
        "Moreh Norm Operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("p"),
            py::kw_only(),
            py::arg("dim") = std::nullopt,
            py::arg("keepdim") = false,
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
        });
}
}  // namespace ttnn::operations::moreh::moreh_norm
