// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/core.hpp"

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
        [](const ttnn::Tensor& tensor, const std::array<int32_t, 2>& shape) -> ttnn::Tensor {
            return ttnn::reshape(tensor, shape);
        },
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

    module.def(
        "to_memory_config",
        &ttnn::operations::core::to_memory_config,
        py::arg("tensor"),
        py::arg("memory_config"),
        py::arg("dtype") = std::nullopt);

    module.def(
        "reallocate",
        [](ttnn::Tensor& input_tensor, const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt)
            -> ttnn::Tensor { return reallocate(input_tensor, memory_config); },
        py::arg("tensor"),
        py::arg("memory_config") = std::nullopt,
        R"doc(
Deallocates device tensor and returns a reallocated tensor

Args:
    * :attr:`input_tensor`: Input Tensor
    )doc");

    module.def(
        "to_layout",
        &to_layout,
        py::arg("tensor"),
        py::arg("layout") = std::nullopt,
        py::arg("dtype") = std::nullopt,
        py::arg("memory_config") = std::nullopt,
        R"doc(
Changes the layout of the tensor. Optionally, changes dtype and memory config.

Args:
    * :attr:`tensor`: Input Tensor
    * :attr:`layout`: Layout to change to
    * :attr:`dtype`: Data type to change to
    * :attr:`memory_config`: Memory config to change to
    )doc");
}

}  // namespace core
}  // namespace operations
}  // namespace ttnn
