// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/core.hpp"
#include "ttnn/operations/binary.hpp"

namespace py = pybind11;

namespace ttnn{
namespace operations {
namespace binary {

void py_module(py::module& m_binary) {
    m_binary.def(
        "add",
          static_cast<ttnn::Tensor (*)(const ttnn::Tensor&, const ttnn::Tensor&, const tt::tt_metal::MemoryConfig&, std::optional<DataType>)>(&ttnn::operations::binary::add),
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = DRAM_MEMORY_CONFIG,
        py::arg("dtype") = std::nullopt
    );

    m_binary.def(
        "add",
        static_cast<ttnn::Tensor (*)(const ttnn::Tensor&, const float, const tt::tt_metal::MemoryConfig&, std::optional<DataType>)>(&ttnn::operations::binary::add),
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = DRAM_MEMORY_CONFIG,
        py::arg("dtype") = std::nullopt
    );
}

}  // namespace binary
}  // namespace operations
}  // namespace ttnn
