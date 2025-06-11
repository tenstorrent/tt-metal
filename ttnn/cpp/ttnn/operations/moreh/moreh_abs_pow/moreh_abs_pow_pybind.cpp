// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_abs_pow_pybind.hpp"

#include "moreh_abs_pow.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::moreh::moreh_abs_pow {
void bind_moreh_abs_pow_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_abs_pow,
        "Moreh Pow Operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("p"),
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
        });
}
}  // namespace ttnn::operations::moreh::moreh_abs_pow
