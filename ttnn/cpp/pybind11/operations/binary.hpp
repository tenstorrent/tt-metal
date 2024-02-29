// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/binary.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace binary {

void py_module(py::module& module) {
    module.def(
        "add",
        [](const ttnn::TensorWrapper& input_tensor_a,
           const float scalar,
           const tt::tt_metal::MemoryConfig& memory_config,
           std::optional<DataType> dtype) {
            return ttnn::TensorWrapper{
                ttnn::operations::binary::add(input_tensor_a.value, scalar, memory_config, dtype)};
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = DRAM_MEMORY_CONFIG,
        py::arg("dtype") = std::nullopt);

    module.def(
        "add",
        [](const ttnn::TensorWrapper& input_tensor_a,
           const ttnn::TensorWrapper& input_tensor_b,
           const tt::tt_metal::MemoryConfig& memory_config,
           std::optional<DataType> dtype) {
            return ttnn::TensorWrapper{
                ttnn::operations::binary::add(input_tensor_a.value, input_tensor_b.value, memory_config, dtype)};
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = DRAM_MEMORY_CONFIG,
        py::arg("dtype") = std::nullopt);
}

}  // namespace binary
}  // namespace operations
}  // namespace ttnn
