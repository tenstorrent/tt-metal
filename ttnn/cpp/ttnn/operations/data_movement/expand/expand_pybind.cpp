// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "expand_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/data_movement/expand/expand.hpp"

namespace ttnn::operations::expand {
void bind_expand_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::expand,
        "Moreh expand Operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("sizes"),
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::expand
