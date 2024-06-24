// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
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
        ttnn::softmax,
        R"doc(softmax(input_tensor: ttnn.Tensor, dim: int, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Compute softmax over :attr:`input_tensor` along :attr:`dim`.

            Args:
                * :attr:`input_tensor`: the input tensor
                * :attr:`dim`: the dimension along which to compute softmax.

            Keyword Args:
                * :attr:`memory_config`: the memory configuration for the output tensor. If not provided, the memory configuration of the input tensor is used.

            Example:

                >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
                >>> output = ttnn.softmax(tensor, -1)
                >>> print(output[0, 0, 0, :3])
                ttnn.Tensor([ 0.0310059, 0.0310059, 0.0310059], dtype=bfloat16 )
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"), py::arg("dim"), py::kw_only(), py::arg("memory_config") = std::nullopt});

    ttnn::bind_registered_operation(
        module,
        ttnn::layer_norm,
        R"doc(rms_norm(input_tensor: ttnn.Tensor, epsilon: float = 1e-12, weight: Optional[ttnn.Tensor] = None, bias: Optional[ttnn.Tensor] = None, residual_input_tensor: Optional[ttnn.Tensor] = None, memory_config: Optional[ttnn.MemoryConfig] = None, program_config: Optional[ttnn.ProgramConfig] = None) -> ttnn.Tensor
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
        R"doc(rms_norm(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, epsilon: float = 1e-12, Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
            Compute rms_norm over :attr:`input_tensor`.
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("weight"),
            py::kw_only(),
            py::arg("epsilon") = 1e-12,
            py::arg("memory_config") = std::nullopt});

    ttnn::bind_registered_operation(
        module,
        ttnn::group_norm,
        R"doc(group_norm(input_tensor: ttnn.Tensor, *, num_groups: int, epsilon: float = 1e-12, weight: Optional[ttnn.Tensor] = None, bias: Optional[ttnn.Tensor] = None) -> ttnn.Tensor
          Compute group_norm over :attr:`input_tensor`.
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("num_groups"),
            py::arg("epsilon") = 1e-12,
            py::arg("input_mask") = std::nullopt,
            py::arg("weight") = std::nullopt,
            py::arg("bias") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("core_grid") = std::nullopt,
            py::arg("inplace") = true,
            py::arg("output_layout") = std::nullopt
        }
    );
}

}  // namespace normalization
}  // namespace operations
}  // namespace ttnn
