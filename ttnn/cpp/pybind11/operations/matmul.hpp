// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/matmul.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace matmul {

void py_module(py::module& module) {
    module.def(
        "matmul",
        [](const ttnn::Tensor& input_tensor_a,
           const ttnn::Tensor& input_tensor_b,
           const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
           const std::optional<const DataType> dtype = std::nullopt,
           const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) -> ttnn::Tensor {
            return ttnn::operations::matmul::matmul(
                input_tensor_a, input_tensor_b, memory_config, dtype, compute_kernel_config);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = DRAM_MEMORY_CONFIG,
        py::arg("dtype") = std::nullopt,
        py::arg("compute_kernel_config") = std::nullopt);

    module.def(
        "matmul",
        [](const ttnn::Tensor& input_tensor_a,
           const ttnn::Tensor& input_tensor_b,
           const ttnn::MatmulProgramConfig& program_config,
           const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
           const std::optional<const DataType> dtype = std::nullopt,
           const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) -> ttnn::Tensor  {
            return ttnn::operations::matmul::matmul(
                input_tensor_a,
                input_tensor_b,
                program_config,
                memory_config,
                dtype,
                compute_kernel_config);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("program_config"),
        py::arg("memory_config") = DRAM_MEMORY_CONFIG,
        py::arg("dtype") = std::nullopt,
        py::arg("compute_kernel_config") = std::nullopt);

    module.def(
        "linear",
        [](const ttnn::Tensor& input_tensor_a,
           const ttnn::Tensor& input_tensor_b,
           const std::optional<const ttnn::Tensor>& bias,
           const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
           const std::optional<const DataType> dtype = std::nullopt,
           const std::optional<const std::string>& activation = std::nullopt,
           const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) -> ttnn::Tensor {
            return ttnn::operations::matmul::linear(
                input_tensor_a,
                input_tensor_b,
                bias,
                memory_config,
                dtype,
                activation,
                compute_kernel_config);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("bias") = std::nullopt,
        py::arg("memory_config") = DRAM_MEMORY_CONFIG,
        py::arg("dtype") = std::nullopt,
        py::arg("activation") = std::nullopt,
        py::arg("compute_kernel_config") = std::nullopt);

    module.def(
        "linear",
        [](const ttnn::Tensor& input_tensor_a,
           const ttnn::Tensor& input_tensor_b,
           const std::optional<const ttnn::Tensor>& bias,
           const ttnn::MatmulProgramConfig& program_config = ttnn::MatmulDefaultProgramConfig{},
           const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
           const std::optional<const DataType> dtype = std::nullopt,
           const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) -> ttnn::Tensor  {
            return ttnn::operations::matmul::linear(
                input_tensor_a,
                input_tensor_b,
                bias,
                program_config,
                memory_config,
                dtype,
                compute_kernel_config);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("bias") = std::nullopt,
        py::arg("program_config"),
        py::arg("memory_config") = DRAM_MEMORY_CONFIG,
        py::arg("dtype") = std::nullopt,
        py::arg("compute_kernel_config") = std::nullopt);
}

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn
