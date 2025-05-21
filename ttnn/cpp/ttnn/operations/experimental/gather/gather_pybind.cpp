// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_pybind.hpp"

#include "gather.hpp"

#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::gather::detail {
namespace py = pybind11;

void bind_gather_operation(py::module& module) {
    auto doc =
        R"doc(
        The `gather` operation extracts values from the input tensor based on indices provided in the index tensor along a specified dimension.

        The input tensor and the index tensor must have the same number of dimensions.
        For all dimensions except the specified one (`dim`), the size of the index tensor must not exceed the size of the input tensor.
        The output tensor will have the same shape as the index tensor. Note that the input and index tensors do not broadcast against each other.

        .. code-block:: python

        Parameters:
            * `input` (Tensor): The source tensor from which values are gathered.
            * `dim` (int): The dimension along which values are gathered.
            * `index` (Tensor): A tensor containing the indices of elements to gather, with the same number of dimensions as the input tensor.

        Keyword Arguments:
            * `sparse_grad` (bool, optional): If `True`, the gradient computation will be sparse. Defaults to `False`.
            * `memory_config` (MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.
            * `out` (Tensor, optional): A preallocated tensor to store the gathered values. Defaults to `None`.

        Additional Information:
            * Currently, the `sparse_grad` argument is not supported.

        Example:

        .. code-block:: python

            import ttnn
            import torch

            # Create a 2D input tensor
            input_tensor = torch.tensor([[10, 20, 30, 40],
                                         [50, 60, 70, 80]])

            # Create a 2D index tensor
            index_tensor = torch.tensor([[3, 0],
                                         [2, 1]])

            # Convert tensors to ttnn format
            input_tensor_ttnn = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
            index_tensor_ttnn = ttnn.from_torch(index_tensor, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

            # Perform the gather operation along dimension 1
            gathered_tensor = ttnn.experimental.gather(input_tensor_ttnn, dim=1, index=index_tensor_ttnn)
            # Result: gathered_tensor = [[40, 10], [70, 60]]
    )doc";

    using OperationType = decltype(ttnn::experimental::gather);
    bind_registered_operation(
        module,
        ttnn::experimental::gather,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int8_t dim,
               const ttnn::Tensor& input_index_tensor,
               const bool sparse_grad,
               std::optional<ttnn::Tensor> optional_output_tensor,
               const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
               QueueId queue_id) -> Tensor {
                return self(
                    queue_id,
                    input_tensor,
                    dim,
                    input_index_tensor,
                    sparse_grad,
                    memory_config,
                    optional_output_tensor);
            },
            py::arg("input").noconvert(),
            py::arg("dim"),
            py::arg("index"),
            py::kw_only(),
            py::arg("sparse_grad") = false,
            py::arg("out") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::gather::detail
