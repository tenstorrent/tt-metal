// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward.hpp"

namespace ttnn::operations::embedding_backward {
namespace py = pybind11;

void py_module(py::module& module) {

    const auto doc =
        R"doc(embedding_bw(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, padding_idx: Optional[int] = None, layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Returns specific indices of the embedding table specified by the input tensor

            /// TODO: Clean me up
            :header: "Argument", "Description", "Data type", "Valid range", "Required"
            "input", "Tensor containing rows we want", "UInt32 Tensor", "Each element greater than 0 and less than number of embeddings in table.  Shape [batch_size, 1, 1, num_rows]", "Yes"
            "weights", "Entire embedding table", "Tensor", "Tensor shape is [1,1, num_embeddings, num_columns]. Num_columns must be divisible by 32.", "Yes"
            "output_gradient", "Loss function gradient with respect to embedding output", "Tensor", "Tensor shape is [batch_size, 1, num_rows, num_columns]", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            "output_dtype", "DataType of output tensor", "DataType", "Default is weights dtype", "No"
            ///

            Args:
                * :attr:`input_tensor`: the indices ttnn.Tensor
                * :attr:`weight`: the embeddings ttnn.Tensor that correspond to the indices ttnn.Tensor

            Keyword Args:
                * :attr:`padding_idx`: the padding token. Default is None.
                * :attr:`dtype`: the data type for the output tensor. Default is None.
                * :attr:`output_tensor`: the optional output tensor. Default is None.
                * :attr:`memory_config`: the memory configuration of the output tensor. Default is input tensor memory config.
                * :attr:`queue_id`: the command queue id. Default is 0.

            Example:
                >>> device_id = 0
                >>> device = ttnn.open_device(device_id=device_id)
                >>> input_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]), dtype=ttnn.uint32), device)
                >>> # an embedding matrix containing 10 tensors of size 4
                >>> weight = ttnn.to_device(ttnn.from_torch(torch.rand(10, 4), dtype=ttnn.bfloat16), device)
                >>> ttnn.embedding(input_tensor, weight)
                ttnn.Tensor([ [[1, 0.106445, 0.988281, 0.59375],
                    [0.212891, 0.964844, 0.199219, 0.996094],
                    [3.78362e-38, 0, 7.89785e-39, 0],
                    [8.04479e-38, 0, 1.25815e-38, 0]],
                [[2.71833e-38, 0, 3.59995e-38, 0],
                    [7.60398e-38, 0, 1.83671e-38, 0],
                    [2.22242e-38, 0, 1.88263e-38, 0],
                    [1.35917e-38, 0, 4.49994e-39, 0]]], dtype=bfloat16 ))doc";
    using OperationType = decltype(ttnn::embedding_bw);
    bind_registered_operation(
        module,
        ttnn::embedding_bw,
        doc,
        ttnn::pybind_overload_t{
        [] (const OperationType& self,
            const ttnn::Tensor& input_tensor,
            const ttnn::Tensor& weight_tensor,
            const ttnn::Tensor& output_gradient_tensor,
            const std::optional<const DataType> dtype,
            std::optional<ttnn::Tensor> &optional_output_tensor,
            const std::optional<ttnn::MemoryConfig>& memory_config,
            uint8_t queue_id) {
                return self(queue_id, input_tensor, weight_tensor, output_gradient_tensor, dtype, memory_config, optional_output_tensor);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("weight_tensor").noconvert(),
            py::arg("output_gradient_tensor").noconvert(),
            py::kw_only(),
            py::arg("dtype").noconvert() = std::nullopt,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::embedding_backward
