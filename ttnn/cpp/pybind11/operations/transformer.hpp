// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn/operations/transformer.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace transformer {

void py_module(py::module& module) {

    module.def("concatenate_heads",
     [](const ttnn::Tensor& input_tensor,
           const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) -> ttnn::Tensor {
            return ttnn::operations::transformer::concatenate_heads(input_tensor, memory_config);
        },
            py::arg().noconvert(), py::arg("memory_config") = std::nullopt, R"doc(
concatenate_heads(input_tensor: ttnn.Tensor, *, memory_config: MemoryConfig = input_tensor.memory_config()) -> ttnn.Tensor

Takes in a tensor of shape ``[batch_size, num_heads, sequence_size, head_size]``, concatenates heads back along the width dimension and returns the tensor of shape ``[batch_size, sequence_size, num_heads * head_size]``

Args:
    * :attr:`input_tensor`: Input Tensor
    * :attr:`memory_config`: Memory Config of the output tensor, defaults to input_tensor.memory_config()
    )doc");

}

}  // namespace transformer
}  // namespace operations
}  // namespace ttnn
