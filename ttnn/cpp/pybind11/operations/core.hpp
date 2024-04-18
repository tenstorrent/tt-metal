// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tt_dnn/op_library/move/move_op.hpp"

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
        [](ttnn::Tensor& input_tensor,
           const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) -> ttnn::Tensor {
            return reallocate(input_tensor, memory_config);
        },
        py::arg("tensor"),
        py::arg("memory_config") = std::nullopt,
        R"doc(
Deallocates device tensor and returns a reallocated tensor

Args:
    * :attr:`input_tensor`: Input Tensor
    )doc");
}

}  // namespace core
}  // namespace operations
}  // namespace ttnn
