// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_tosa_pybind.hpp"

#include "gather_tosa.hpp"

#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::tosa::gather::detail {
namespace py = pybind11;

void bind_gather_tosa_operation(py::module& module) {
    auto doc =
        R"doc(
        Generate a tensor for which each element in the output is a subtensor of the values tensor based on the indices.
        N is the number of batches, W the number of indices in each batch, K the range of each index and C the number data channels for each index.

        Parameters:
            * `input` (Tensor) - Shape([N,K,C]): The source tensor from which values are gathered.
            * `index` (Tensor) - Shape([N,W]): A tensor containing the indices of elements to gather, with the same number of dimensions as the input tensor.

        Output:
            * `output` (Tensor) - Shape([N,W,C]): The gathered tensor.

        Keyword Arguments:
            * `memory_config` (MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.

        .. code-block:: python

        Logic:
            for_each(0 <= n < N, 0 <= w < W, 0 <= c < C) {
                index_t k = tensor_read<index_t>(indices, [N,W], [n,w]);
                REQUIRE(0 <= k && k < K);
                in_out_t value = tensor_read<in_out_t>(values, [N,K,C], [n,k,c]);
                tensor_write<in_out_t>(output, [N,W,C], [n,w,c], value);
            }

        Example:

        .. code-block:: python

            import ttnn
            import torch

            N, K, C = 2, 5, 3
            W = 4

            # Create a 3D input tensor
            input_tensor = torch.randn((N, K, C), dtype=torch.bfloat16)

            # Create a 2D index tensor
            indices = torch.randint(0, K, (N, W), dtype=torch.uint16)

            # Convert tensors to ttnn format
            input_tensor_ttnn = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
            index_tensor_ttnn = ttnn.from_torch(index_tensor, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

            # Perform the gather operation
            gathered_tensor = ttnn.experimental.tosa_gather(input_tensor_ttnn, index_tensor_ttnn)

            # Equivalent to PyTorch: gathered_tensor = torch.gather(input_tensor, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, C))
    )doc";

    using OperationType = decltype(ttnn::experimental::tosa::gather);
    bind_registered_operation(
        module,
        ttnn::experimental::tosa::gather,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& input_index_tensor,
               const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
               QueueId queue_id) -> Tensor { return self(queue_id, input_tensor, input_index_tensor, memory_config); },
            py::arg("input").noconvert(),
            py::arg("index"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}
}  // namespace ttnn::operations::experimental::tosa::gather::detail
