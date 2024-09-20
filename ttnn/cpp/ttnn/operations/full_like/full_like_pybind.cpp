// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "full_like_pybind.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/full_like/device/full_like_device_operation.hpp"
#include "full_like.hpp"

namespace py = pybind11;

namespace ttnn::operations::full_like {

void bind_full_like_operation(py::module& module) {

bind_registered_operation(
    module,
    ttnn::full_like,
    "Full like Operation",
    ttnn::pybind_arguments_t{
        py::arg("input"),
        py::arg("fill_value"),
        py::kw_only(),
        py::arg("dtype") = std::nullopt,
        py::arg("layout") = std::nullopt,
        py::arg("memory_config") = std::nullopt});
}

}
