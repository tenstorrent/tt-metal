// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/embedding/embedding.hpp"

namespace ttnn::operations::embedding {
namespace py = pybind11;

void py_module(py::module& module) {
    py::enum_<ttnn::operations::embedding::EmbeddingsType>(module, "EmbeddingsType")
        .value("GENERIC", ttnn::operations::embedding::EmbeddingsType::GENERIC)
        .value("PADDED", ttnn::operations::embedding::EmbeddingsType::PADDED)
        .value("BINARY", ttnn::operations::embedding::EmbeddingsType::BINARY);

    const auto doc =
        R"doc(

        Retrieves word embeddings using input_tensor. The input_tensor is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.

        Args:
            input_tensor (ttnn.Tensor): the input indices tensor.
            weight (ttnn.Tensor): the embeddings tensor that corresponds to the indices tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.
            padding_idx (int, optional): the padding token. Default to `None`.
            layout (ttnn.Layout): the layout of the output tensor. Defaults to `ttnn.ROW_MAJOR_LAYOUT`.
            embeddings_type (ttnn.EmbeddingsType): the type of embeddings. Defaults to `ttnn._ttnn.operations.embedding.EmbeddingsType.GENERIC`.
            dtype (ttnn.DataType, optional): the data type for the output tensor. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        Example:
            >>> device_id = 0
            >>> device = ttnn.open_device(device_id=device_id)
            >>> tensor = ttnn.to_device(ttnn.from_torch(torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]), dtype=ttnn.uint32), device=device)
            >>> # an embedding matrix containing 10 tensors of size 4
            >>> weight = ttnn.to_device(ttnn.from_torch(torch.rand(10, 4), dtype=ttnn.bfloat16), device=device)
            >>> output = ttnn.embedding(tensor, weight)
            ttnn.Tensor([ [[1, 0.106445, 0.988281, 0.59375],
                [0.212891, 0.964844, 0.199219, 0.996094],
                [3.78362e-38, 0, 7.89785e-39, 0],
                [8.04479e-38, 0, 1.25815e-38, 0]],
            [[2.71833e-38, 0, 3.59995e-38, 0],
                [7.60398e-38, 0, 1.83671e-38, 0],
                [2.22242e-38, 0, 1.88263e-38, 0],
                [1.35917e-38, 0, 4.49994e-39, 0]]], dtype=bfloat16))doc";

    using OperationType = decltype(ttnn::embedding);
    bind_registered_operation(
        module,
        ttnn::embedding,
        doc,
        ttnn::pybind_overload_t{
        [] (const OperationType& self,
            const ttnn::Tensor& input_tensor,
            const ttnn::Tensor& weight,
            const std::optional<int>& padding_idx,
            const ttnn::Layout& layout,
            EmbeddingsType embeddings_type,
            const std::optional<const DataType> dtype,
            std::optional<ttnn::Tensor> &optional_output_tensor,
            const std::optional<ttnn::MemoryConfig>& memory_config,
            uint8_t queue_id) {
                return self(queue_id, input_tensor, weight, padding_idx, layout, embeddings_type, dtype, memory_config, optional_output_tensor);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("weight").noconvert(),
            py::kw_only(),
            py::arg("padding_idx") = std::nullopt,
            py::arg("layout") = ttnn::ROW_MAJOR_LAYOUT,
            py::arg("embeddings_type").noconvert() = EmbeddingsType::GENERIC,
            py::arg("dtype").noconvert() = std::nullopt,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::embedding
