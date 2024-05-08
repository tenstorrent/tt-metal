// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../decorators.hpp"
#include "ttnn/operations/normalization.hpp"

namespace py = pybind11;

namespace {
    MemoryConfig dram_memory_config = tt::tt_metal::MemoryConfig{.memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED,.buffer_type=tt::tt_metal::BufferType::DRAM};
}

namespace ttnn {
namespace operations {
namespace normalization {
void py_module(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::layer_norm,
        R"doc(rms_norm(input_tensor: ttnn.Tensor, epsilon: float = 1e-12, weight: ttnn.Tensor, bias: ttnn.Tensor, residual_input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig, program_config: ttnn.LayerNormProgramConfig) -> ttnn.Tensor
            Compute layer_norm over :attr:`input_tensor`.
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("epsilon") = 1e-12,
            py::arg("weight") = std::nullopt,
            py::arg("bias") = std::nullopt,
            py::arg("residual_input_tensor") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("program_config") = std::nullopt});

    ttnn::bind_registered_operation(
        module,
        ttnn::rms_norm,
        R"doc(rms_norm(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, epsilon: float = 1e-12, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor
            Compute rms_norm over :attr:`input_tensor`.
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("weight"),
            py::kw_only(),
            py::arg("epsilon") = 1e-12,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace normalization
}  // namespace operations
}  // namespace ttnn
