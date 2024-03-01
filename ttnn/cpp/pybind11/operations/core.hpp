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
        [](const ttnn::TensorWrapper& tensor, const ttnn::Shape& shape) {
            return ttnn::TensorWrapper{ttnn::reshape(tensor.value, shape)};
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::TensorWrapper& tensor, const std::array<int32_t, 1>& shape) {
            return ttnn::TensorWrapper{ttnn::reshape(tensor.value, shape)};
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::TensorWrapper& tensor, const std::array<int32_t, 2>& shape) {
            return ttnn::TensorWrapper{ttnn::reshape(tensor.value, shape)};
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::TensorWrapper& tensor, const std::array<int32_t, 3>& shape) {
            return ttnn::TensorWrapper{ttnn::reshape(tensor.value, shape)};
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "reshape",
        [](const ttnn::TensorWrapper& tensor, const std::array<int32_t, 4>& shape) {
            return ttnn::TensorWrapper{ttnn::reshape(tensor.value, shape)};
        },
        py::arg("tensor"),
        py::arg("shape"));

    module.def(
        "unsqueeze_to_4D",
        [](const ttnn::TensorWrapper& tensor) { return ttnn::TensorWrapper{ttnn::unsqueeze_to_4D(tensor.value)}; },
        py::arg("tensor"));
}

}  // namespace core
}  // namespace operations
}  // namespace ttnn
