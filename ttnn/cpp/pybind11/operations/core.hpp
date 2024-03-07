// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace core {

void py_module(py::module& module) {
    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const ttnn::Shape& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 1>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 2>& shape) -> ttnn::Tensor { return ttnn::reshape(tensor, shape); },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 3>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 4>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "unsqueeze_to_4D",
        [](const ttnn::Tensor& tensor) -> ttnn::Tensor { return ttnn::unsqueeze_to_4D(tensor); },
        py::arg("tensor"));
}

}  // namespace core
}  // namespace operations
}  // namespace ttnn
