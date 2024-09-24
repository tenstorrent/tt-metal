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
        R"doc(
        Divides :attr:`tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax.


        Args:
            input_tensor (ttnn.Tensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            head_size (int, optional): Number of heads. Defaults to `None`.
            attention_mask(ttnn.Tensor, optional): Attention Mask. Defaults to `None`.
            program_config (SoftmaxProgramConfig): Program Config of the output tensor. Defaults to `SoftmaxDefaultProgramConfig()`.
            causal_mask (bool, optional): the attention mask is causal. Defaults to `false`.


        Returns:
            ttnn.Tensor: the output tensor.

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
        R"doc(
        In-Place divides :attr:`tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax.


        Args:
            input_tensor (ttnn.Tensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            head_size (int, optional): Number of heads. Defaults to `None`.
            attention_mask(ttnn.Tensor, optional): Attention Mask. Defaults to `None`.
            program_config (SoftmaxProgramConfig): Program Config of the output tensor. Defaults to `SoftmaxDefaultProgramConfig()`.
            causal_mask (bool, optional): the attention mask is causal. Defaults to `false`.


        Returns:
            ttnn.Tensor: the output tensor.

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
