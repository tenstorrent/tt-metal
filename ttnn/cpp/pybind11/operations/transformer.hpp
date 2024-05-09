// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../decorators.hpp"
#include "ttnn/operations/transformer.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace transformer {

void py_module(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::concatenate_heads,
        R"doc(concatenate_heads(input_tensor: ttnn.Tensor, *, memory_config: Optional[MemoryConfig] = None) -> ttnn.Tensor

            Takes in a tensor of shape ``[batch_size, num_heads, sequence_size, head_size]``, concatenates heads back along the width dimension and returns the tensor of shape ``[batch_size, sequence_size, num_heads * head_size]``

            Args:
                * :attr:`input_tensor`: Input Tensor
                * :attr:`memory_config`: Memory Config of the output tensor, if None then it gets set to input_tensor.memory_config()
        )doc",
        ttnn::pybind_arguments_t{py::arg("input_tensor"), py::kw_only(), py::arg("memory_config") = std::nullopt});

    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::attention_softmax_,
        R"doc(attention_softmax_(tensor: ttnn.Tensor, *, head_size: Optional[int] = None, attention_mask: Optional[ttnn.Tensor] = None, program_config: Optional[SoftmaxProgramConfig] = SoftmaxDefaultProgramConfig(), causal_mask: bool = False,  memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            In-Place divides :attr:`tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax.

            Args:
                * :attr:`tensor`: Input Tensor
                * :attr:`head_size`: Number of heads
                * :attr:`attention_mask`: Attention Mask
                * :attr:`program_config`: Program Config of the output tensor
                * :attr:`memory_config`: Memory Config of the output tensor, defaults to input_tensor.memory_config()
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("tensor"),
            py::kw_only(),
            py::arg("head_size") = std::nullopt,
            py::arg("attention_mask") = std::nullopt,
            py::arg("program_config").noconvert() =
                tt::operations::primary::transformers::SoftmaxDefaultProgramConfig{},
            py::arg("causal_mask") = false,
            py::arg("memory_config") = std::nullopt});

    module.def("split_query_key_value_and_split_heads",
    [](const Tensor &input_tensor, const std::optional<Tensor> &input_tensor_kv,
        const uint32_t num_heads, const std::optional<uint32_t> num_kv_heads,
        const bool transpose_k_heads,
        const std::optional<MemoryConfig>& mem_config) -> std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> {
            return ttnn::operations::transformer::split_query_key_value_and_split_heads(input_tensor, input_tensor_kv, num_heads,
            num_kv_heads, transpose_k_heads, mem_config);
        },

        py::arg("input_tensor").noconvert(), py::arg("kv_input_tensor") = std::nullopt, py::kw_only(), py::arg("num_heads"),
        py::arg("num_kv_heads") = std::nullopt, py::arg("transpose_key") = true, py::arg("memory_config") = std::nullopt);

}
}  // namespace transformer
}  // namespace operations
}  // namespace ttnn
