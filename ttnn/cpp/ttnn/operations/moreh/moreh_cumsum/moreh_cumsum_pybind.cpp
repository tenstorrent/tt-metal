// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_cumsum_pybind.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"

namespace ttnn::operations::moreh::moreh_cumsum {

void bind_moreh_cumsum_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_cumsum,
        "Moreh Cumsum Operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
        });
}

void bind_moreh_cumsum_backward_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_cumsum_backward,
        "Moreh Cumsum Backward Operation",
        ttnn::pybind_arguments_t{
            py::arg("output_grad"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("input_grad") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
        });
}

}  // namespace ttnn::operations::moreh::moreh_cumsum
