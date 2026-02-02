// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "embedding_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn/operations/embedding/embedding.hpp"

namespace ttnn::operations::embedding {

void py_module(nb::module_& mod) {
    export_enum<ttnn::prim::EmbeddingsType>(mod, "EmbeddingsType");
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
        mod,
        ttnn::embedding,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight,
               const std::optional<int>& padding_idx,
               const std::optional<ttnn::Layout>& layout,
               ttnn::prim::EmbeddingsType embeddings_type,
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
            nb::arg("input_tensor").noconvert(),
            nb::arg("weight").noconvert(),
            nb::kw_only(),
            nb::arg("padding_idx") = nb::none(),
            nb::arg("layout") = nb::none(),
            nb::arg("embeddings_type").noconvert() = nb::cast(ttnn::prim::EmbeddingsType::GENERIC),
            nb::arg("dtype").noconvert() = nb::none(),
            nb::arg("output_tensor").noconvert() = nb::none(),
            nb::arg("memory_config") = nb::none()});
}

}  // namespace ttnn::operations::embedding
