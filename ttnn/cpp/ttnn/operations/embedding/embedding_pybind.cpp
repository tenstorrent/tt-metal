// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "embedding_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/embedding/embedding.hpp"

namespace ttnn::operations::embedding {
namespace py = pybind11;

void py_module(py::module& module) {
    py::enum_<ttnn::operations::embedding::EmbeddingsType>(module, "EmbeddingsType")
        .value("GENERIC", ttnn::operations::embedding::EmbeddingsType::GENERIC)
        .value("PADDED", ttnn::operations::embedding::EmbeddingsType::PADDED)
        .value("BINARY", ttnn::operations::embedding::EmbeddingsType::BINARY);

    const auto* const doc =
        R"doc(
        Retrieves word embeddings using input_tensor. The input_tensor is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.

        Args:
            input_tensor (ttnn.Tensor): the input indices tensor.
            weight (ttnn.Tensor): the embeddings tensor that corresponds to the indices tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            padding_idx (int, optional): the padding token. Default to `None`.
            layout (ttnn.Layout): the layout of the output tensor. Defaults to `ttnn.ROW_MAJOR_LAYOUT`.
            embeddings_type (ttnn.EmbeddingsType): the type of embeddings. Defaults to `ttnn._ttnn.operations.embedding.EmbeddingsType.GENERIC`.
            dtype (ttnn.DataType, optional): the data type for the output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor of layout == layout or layout of the weights tensor.
        )doc";

    using OperationType = decltype(ttnn::embedding);
    bind_registered_operation(
        module,
        ttnn::embedding,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight,
               const std::optional<int>& padding_idx,
               const std::optional<ttnn::Layout>& layout,
               EmbeddingsType embeddings_type,
               const std::optional<const DataType> dtype,
               std::optional<ttnn::Tensor>& optional_output_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(
                    input_tensor,
                    weight,
                    padding_idx,
                    layout,
                    embeddings_type,
                    dtype,
                    memory_config,
                    optional_output_tensor);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("weight").noconvert(),
            py::kw_only(),
            py::arg("padding_idx") = std::nullopt,
            py::arg("layout") = std::nullopt,
            py::arg("embeddings_type").noconvert() = EmbeddingsType::GENERIC,
            py::arg("dtype").noconvert() = std::nullopt,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::embedding
