// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"

namespace ttnn::operations::moreh::moreh_sum {
void bind_moreh_sum_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_sum,
        "Moreh moreh_sum Operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("keepdim") = false,
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_sum
