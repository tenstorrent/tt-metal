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
Concats :attr:`tensors` in the given :attr:`dim`.

Args:
    * :attr:`tensors`: the tensors to be concatenated.
    * :attr:`dim`: the concatenating dimension.

Keyword Args:
    * :attr:`memory_config`: the memory configuration to use for the operation
    * :attr:`queue_id` (Optional[uint8]): command queue id
    * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor

Example:

    >>> tensor = ttnn.concat(ttnn.from_torch(torch.zeros((1, 1, 64, 32), ttnn.from_torch(torch.zeros((1, 1, 64, 32), dim=3)), device)

    >>> tensor1 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> tensor2 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> output = ttnn.concat([tensor1, tensor2], dim=4)
    >>> print(output.shape)
    [1, 1, 32, 64]

    )doc";

    using OperationType = decltype(ttnn::concat);
    ttnn::bind_registered_operation(
        module,
        ttnn::concat,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const std::vector<ttnn::Tensor>& tensors,
                const int dim,
                std::optional<ttnn::Tensor> &optional_output_tensor,
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
