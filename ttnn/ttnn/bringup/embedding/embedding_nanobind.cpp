// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embedding_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "embedding.hpp"

namespace ttnn::operations::bringup::embedding {

void bind_embedding(nb::module_& mod) {
    export_enum<ttnn::prim::bringup::EmbeddingsType>(mod, "EmbeddingsType");
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
            max_grid (ttnn.CoreCoord, optional): bring-up option (ttnn/ttnn/bringup/embedding, a fork of
                ttnn.embedding), default None (the device's whole compute grid, the source op). An interleaved
                output's work split runs on the rectangle (0, 0) .. (max_grid - 1) of the grid
                (deepseek_v41_flash_d_p FIX.15: a Galaxy's 110-core cap).

        Returns:
            ttnn.Tensor: the output tensor of layout == layout or layout of the weights tensor.
        )doc";

    ttnn::bind_function<"embedding", "ttnn.bringup.">(
        mod,
        doc,
        &ttnn::bringup::embedding,
        nb::arg("input_tensor").noconvert(),
        nb::arg("weight").noconvert(),
        nb::kw_only(),
        nb::arg("padding_idx") = nb::none(),
        nb::arg("layout") = nb::none(),
        nb::arg("embeddings_type").noconvert() = nb::cast(ttnn::prim::bringup::EmbeddingsType::GENERIC),
        nb::arg("dtype").noconvert() = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor").noconvert() = nb::none(),
        nb::arg("max_grid") = nb::none());
}

}  // namespace ttnn::operations::bringup::embedding
