// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward_pybind.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward.hpp"

namespace ttnn::operations::embedding_backward {
namespace py = pybind11;

void py_bind_embedding_backward(py::module& module) {
    const auto doc =
        R"doc(

        Returns the input gradients of the output gradients tensor with respect to the input indices.


        Args:
            input_tensor (ttnn.Tensor): the input indices tensor.
            weight (ttnn.Tensor): the embeddings tensor that corresponds to the indices tensor. This tensor is only used to extract the vocabulary size.
            output_gradient_tensor (ttnn.Tensor): the output gradient tensor from the previous backwards op.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.
            dtype (ttnn.DataType, optional): the data type for the output tensor. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        Example:
            >>> device_id = 0
            >>> device = ttnn.open_device(device_id=device_id)
            >>> batch_size, seq_len, embedding_dim, num_embeddings = 2, 1024, 4096, 3200

            >>> input_shape = (batch_size, seq_len)
            >>> input_index = torch.randint(0, num_embeddings, input_shape)
            >>> input_tensor = ttnn.from_torch(input_index, dtype=ttnn.uint32, device=device)

            >>> weights_shape = (num_embeddings, embedding_dim)
            >>> weights = torch.randn(weights_shape, requires_grad=True)
            >>> weights_ttnn = ttnn.from_torch(weights, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            >>> grad_shape = (1, 1, batch_size * seq_len, embedding_dim)
            >>> grad_data = torch.randn(grad_shape, requires_grad=True)
            >>> grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            >>> output = ttnn.embedding_bw(input_tensor, weights_ttnn, grad_tensor, dtype=ttnn.bfloat16))doc";

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
               const std::optional<ttnn::MemoryConfig>& memory_config,
               uint8_t queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    weight_tensor,
                    output_gradient_tensor,
                    dtype,
                    memory_config,
                    optional_output_tensor);
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
