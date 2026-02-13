// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "embedding_backward_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward.hpp"

namespace ttnn::operations::embedding_backward {

void bind_embedding_backward(nb::module_& mod) {
    const auto* const doc =
        R"doc(
        Returns the input gradients of the output gradients tensor with respect to the input indices.


        Args:
            input_tensor (ttnn.Tensor): the input indices tensor.
            weight (ttnn.Tensor): the embeddings tensor that corresponds to the indices tensor. This tensor is only used to extract the vocabulary size.
            output_gradient_tensor (ttnn.Tensor): the output gradient tensor from the previous backwards op.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            dtype (ttnn.DataType, optional): the data type for the output tensor. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            The input and the output gradient tensors must have the same datatype.
        )doc";

    ttnn::bind_function<"embedding_bw">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::embedding_bw,
            nb::arg("input_tensor").noconvert(),
            nb::arg("weight_tensor").noconvert(),
            nb::arg("output_gradient_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("dtype").noconvert() = nb::none(),
            nb::arg("output_tensor").noconvert() = nb::none(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::embedding_backward
