// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/moreh_embedding/moreh_embedding.hpp"

namespace ttnn::operations::moreh_embedding {
namespace py = pybind11;

void py_module(py::module& module) {
    const auto doc =
        R"doc(moreh_embedding(input: ttnn.Tensor, weight: ttnn.Tensor, *, max_norm: Optional[float] = None, norm_type = float = 2.0 , output: Optional[ttnn.Tensor] = None, dtype: Optional[ttnn.DataType] = None, memory_config: Optional[ttnn.MemoryConfig] = None, compute_kernel_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Retrieves word embeddings using input. The input is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.

            Args:
                * :attr:`input`: the indices ttnn.Tensor
                * :attr:`weight`: the embeddings ttnn.Tensor that correspond to the indices ttnn.Tensor

            Keyword Args:
                * :attr:`max_norm`: If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm. Note: this will modify weight in-place.
                * :attr:`norm_type`: The p of the p-norm to compute for the max_norm option. Default 2.
                * :attr:`output`: the optional output tensor. Default is None.
                * :attr:`dtype`: the data type for the output tensor. Default is None.
                * :attr:`memory_config`: the memory configuration of the output tensor. Default is input tensor memory config.
                * :attr:`compute_kernel_config`: the compute kernel configuration for the op. If not provided, the default configuration of the op is used.
                * :attr:`queue_id`: the command queue id. Default is 0.
            )doc";
    using OperationType = decltype(ttnn::moreh_embedding);
    bind_registered_operation(
        module,
        ttnn::moreh_embedding,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input,
               const ttnn::Tensor& weight,
               std::optional<float> max_norm,
               float norm_type,
               std::optional<ttnn::Tensor>& output,
               const std::optional<const DataType> dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
               uint8_t queue_id) {
                return self(queue_id, input, weight, max_norm, norm_type, output, dtype, memory_config);
            },
            py::arg("input").noconvert(),
            py::arg("weight").noconvert(),
            py::kw_only(),
            py::arg("max_norm") = std::nullopt,
            py::arg("norm_type") = 2.0,
            py::arg("output").noconvert() = std::nullopt,
            py::arg("dtype").noconvert() = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::moreh_embedding
