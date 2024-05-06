// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn/operations/normalization.hpp"

namespace py = pybind11;

namespace {
    MemoryConfig dram_memory_config = tt::tt_metal::MemoryConfig{.memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED,.buffer_type=tt::tt_metal::BufferType::DRAM};
}

namespace ttnn {
namespace operations {
namespace normalization {
void py_module(py::module& module) {

    module.def("layer_norm", &layer_norm,
        py::arg("input_tensor"),
        py::kw_only(),
        py::arg("epsilon") = 1e-12,
        py::arg("weight") = std::nullopt,
        py::arg("bias") = std::nullopt,
        py::arg("residual_input_tensor") = std::nullopt,
        py::arg("memory_config") = ::dram_memory_config,
        py::arg("program_config") = std::nullopt,
        R"doc(
Compute layer_norm over :attr:`input_tensor`.
    )doc");

    module.def("rms_norm", &rms_norm,
        py::arg("input_tensor"),
        py::arg("weight"),
        py::kw_only(),
        py::arg("epsilon") = 1e-12,
        R"doc(
Compute rms_norm over :attr:`input_tensor`.
    )doc");
}

}  // namespace normalization
}  // namespace operations
}  // namespace ttnn
