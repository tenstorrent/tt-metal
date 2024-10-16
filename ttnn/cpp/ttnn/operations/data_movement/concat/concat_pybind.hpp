// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "concat.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_concat(py::module& module) {
    const auto doc = R"doc(

Args:
    input_tensor (ttnn.Tensor): the input tensor.
    dim (number): the concatenating dimension.

Keyword Args:
    memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
    queue_id (int, optional): command queue id. Defaults to `0`.
    output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.

Returns:
    ttnn.Tensor: the output tensor.

Example:

    >>> tensor = ttnn.concat(ttnn.from_torch(torch.zeros((1, 1, 64, 32), ttnn.from_torch(torch.zeros((1, 1, 64, 32), dim=3)), device)

    >>> tensor1 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> tensor2 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> output = ttnn.concat([tensor1, tensor2], dim=4)
    >>> print(output.shape)
    [1, 1, 32, 64]

    )doc";

    using OperationType = decltype(ttnn::concat);
    ttnn::bind_registered_operation(module,
                                    ttnn::concat,
                                    doc,
                                    ttnn::pybind_overload_t{
                                        [](const OperationType& self,
                                           const std::vector<ttnn::Tensor>& tensors,
                                           const int dim,
                                           std::optional<ttnn::Tensor>& optional_output_tensor,
                                           std::optional<ttnn::MemoryConfig>& memory_config,
                                           uint8_t queue_id) {
                                            return self(queue_id, tensors, dim, memory_config, optional_output_tensor);
                                        },
                                        py::arg("tensors"),
                                        py::arg("dim") = 0,
                                        py::kw_only(),
                                        py::arg("output_tensor").noconvert() = std::nullopt,
                                        py::arg("memory_config") = std::nullopt,
                                        py::arg("queue_id") = 0,
                                    });
}

}  // namespace ttnn::operations::data_movement::detail
