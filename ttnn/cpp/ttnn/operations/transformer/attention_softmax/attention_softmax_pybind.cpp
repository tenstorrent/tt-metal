// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attention_softmax_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "attention_softmax.hpp"

namespace ttnn::operations::transformer {

void py_bind_attention_softmax(pybind11::module& module) {
    namespace py = pybind11;
    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::attention_softmax,
        R"doc(attention_softmax(tensor: ttnn.Tensor, *, head_size: Optional[int] = None, attention_mask: Optional[ttnn.Tensor] = None, program_config: Optional[SoftmaxProgramConfig] = SoftmaxDefaultProgramConfig(), causal_mask: bool = False,  memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Divides :attr:`tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax.

            Args:
                * :attr:`tensor`: Input Tensor

            Keyword Args:
                * :attr:`head_size`: Number of heads
                * :attr:`attention_mask`: Attention Mask
                * :attr:`program_config`: Program Config of the output tensor
                * :attr:`causal_mask`: the attention mask is causal
                * :attr:`memory_config`: Memory Config of the output tensor, defaults to input_tensor.memory_config()
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("tensor"),
            py::kw_only(),
            py::arg("head_size") = std::nullopt,
            py::arg("attention_mask") = std::nullopt,
            py::arg("program_config").noconvert() =
                ttnn::operations::normalization::SoftmaxDefaultProgramConfig{},
            py::arg("causal_mask") = false,
            py::arg("memory_config") = std::nullopt});


    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::attention_softmax_,
        R"doc(attention_softmax_(tensor: ttnn.Tensor, *, head_size: Optional[int] = None, attention_mask: Optional[ttnn.Tensor] = None, program_config: Optional[SoftmaxProgramConfig] = SoftmaxDefaultProgramConfig(), causal_mask: bool = False,  memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            In-Place divides :attr:`tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax.

            Args:
                * :attr:`tensor`: Input Tensor

            Keyword Args:
                * :attr:`head_size`: Number of heads
                * :attr:`attention_mask`: Attention Mask
                * :attr:`program_config`: Program Config of the output tensor
                * :attr:`causal_mask`: the attention mask is causal
                * :attr:`memory_config`: Memory Config of the output tensor, defaults to input_tensor.memory_config()
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("tensor"),
            py::kw_only(),
            py::arg("head_size") = std::nullopt,
            py::arg("attention_mask") = std::nullopt,
            py::arg("program_config").noconvert() =
                ttnn::operations::normalization::SoftmaxDefaultProgramConfig{},
            py::arg("causal_mask") = false,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::transformer
