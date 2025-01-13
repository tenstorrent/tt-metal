// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_arange_pybind.hpp"

#include "moreh_arange.hpp"
#include "pybind11/decorators.hpp"

namespace ttnn::operations::moreh::moreh_arange {
void bind_moreh_arange_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_arange,
        "Moreh Arange Operation",
        ttnn::pybind_arguments_t{
            py::arg("start") = 0,
            py::arg("end"),
            py::arg("step") = 1,
            py::arg("any"),
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("untilize_out") = false,
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
        });
}
}  // namespace ttnn::operations::moreh::moreh_arange
