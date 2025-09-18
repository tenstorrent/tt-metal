// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "concat.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_concat(py::module& module) {
    const auto doc = R"doc(

Args:
    input_tensor (List of ttnn.Tensor): the input tensors.
    dim (number): the concatenating dimension.

Keyword Args:
    memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
    queue_id (int, optional): command queue id. Defaults to `0`.
    output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
    groups (int, optional): When `groups` is set to a value greater than 1, the inputs are split into N `groups` partitions, and elements are interleaved from each group into the output tensor. Each group is processed independently, and elements from each group are concatenated in an alternating pattern based on the number of groups. This is useful for recombining grouped convolution outputs during residual concatenation. Defaults to `1`. Currently, groups > `1` is only supported for two height sharded input tensors.

Returns:
    ttnn.Tensor: the output tensor.

Example:

    >>> tensor1 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> tensor2 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> output = ttnn.concat([tensor1, tensor2], dim=3)
    >>> print(output.shape)
    [1, 1, 64, 64]

    )doc";

    using OperationType = decltype(ttnn::concat);
    ttnn::bind_registered_operation(
        module,
        ttnn::concat,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const std::vector<ttnn::Tensor>& tensors,
               const int dim,
               std::optional<ttnn::Tensor>& optional_output_tensor,
               std::optional<ttnn::MemoryConfig>& memory_config,
               const int groups,
               QueueId queue_id) {
                return self(queue_id, tensors, dim, memory_config, optional_output_tensor, groups);
            },
            py::arg("tensors"),
            py::arg("dim") = 0,
            py::kw_only(),
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("groups") = 1,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::data_movement::detail
