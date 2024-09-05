// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_cumsum {

void bind_moreh_cumsum_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_cumsum,
        "ttnn::moreh_cumsum",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("output"),
            py::arg("dim")});
}

void bind_moreh_cumsum_backward_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_cumsum_backward,
        "ttnn::moreh_cumsum_backward",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("output"),
            py::arg("dim")});
}

}  // namespace ttnn::operations::moreh::moreh_cumsum
