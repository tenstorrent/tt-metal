// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cumprod_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/reduction/cumprod/cumprod.hpp"
#include "ttnn/types.hpp"
#include "ttnn/common/queue_id.hpp"

namespace ttnn::operations::experimental::reduction::cumprod::detail {
void bind_cumprod_operation(py::module& module) {
    auto doc =
        R"doc(

            cumprod(input_tensor: ttnn.Tensor, dim: int) -> ttnn.Tensor

            Returns a tensor witth cumulative product calculated along a given axis (`dim`).

            Args:
                input_tensor (ttnn.Tensor): the input tensor to calculate cumulative product of.
                dim (int): the axis of product cumulation.

            Keyword Args:
                dtype (ttnn.DataType, optional): the underlying type to which the input
                queue_id (int, optional): the command queue's ID, defaults to 0.
                memory_config (ttnn.MemoryConfig, optional): the memory configuration for the operation. Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor with a cumulative product.

            Example:
                >>> # return a ref

                >>> tensor = ttnn.from_torch(torch.tensor((1, 2, 3), dtype=torch.bfloat16), device=device)
                >>> # Note that the call below will output the same tensor it was fed for the time being,
                >>> # until the actual implementation is provided.
                >>> output = ttnn.experimental.cumprod(tensor, 1)
                >>> assert tensor.shape == output.shape
                >>> assert tensor.dtype == output.dtyoe

                >>> # preallocation and return another ref

                >>> tensor = ttnn.from_torch(torch.tensor((1, 2, 3), dtype=torch.uint8), device=device)
                >>> # Note that the call below will output the same tensor it was fed for the time being,
                >>> # until the actual implementation is provided.
                >>> tensor_copy = ttnn.zeros_like(tensor)
                >>> output = ttnn.experimental.cumprod(tensor, 1, out=tensor_copy)
                >>> assert tensor.shape == output.shape
                >>> assert tensor.dtype == output.dtype
            )doc";

    using OperationType = decltype(ttnn::experimental::cumprod);
    bind_registered_operation(
        module,
        ttnn::experimental::cumprod,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               std::optional<DataType>& dtype,
               std::optional<Tensor> optional_out,
               const std::optional<MemoryConfig>& memory_config,
               const QueueId& queue_id = DefaultQueueId) {
                return self(input_tensor, dim, dtype, optional_out, memory_config, queue_id);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dim"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("out") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::reduction::cumprod::detail
