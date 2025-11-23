// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embedding_backward_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward.hpp"

namespace ttnn::operations::embedding_backward {
namespace py = pybind11;

void py_bind_embedding_backward(py::module& module) {
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

    using OperationType = decltype(ttnn::embedding_bw);
    bind_registered_operation(
        module,
        ttnn::embedding_bw,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               const ttnn::Tensor& output_gradient_tensor,
               const std::optional<const DataType> dtype,
               std::optional<ttnn::Tensor>& optional_output_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(
                    input_tensor, weight_tensor, output_gradient_tensor, dtype, memory_config, optional_output_tensor);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("weight_tensor").noconvert(),
            py::arg("output_gradient_tensor").noconvert(),
            py::kw_only(),
            py::arg("dtype").noconvert() = std::nullopt,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::embedding_backward
