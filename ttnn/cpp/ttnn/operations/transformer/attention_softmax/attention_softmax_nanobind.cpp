// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "attention_softmax_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "attention_softmax.hpp"

namespace ttnn::operations::transformer {

void bind_attention_softmax(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
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
        ttnn::nanobind_arguments_t{
            nb::arg("tensor"),
            nb::kw_only(),
            nb::arg("head_size") = nb::none(),
            nb::arg("attention_mask") = nb::none(),
            nb::arg("program_config").noconvert() = ttnn::SoftmaxDefaultProgramConfig{},
            nb::arg("causal_mask") = false,
            nb::arg("memory_config") = nb::none()});

    ttnn::bind_registered_operation(
        mod,
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
        ttnn::nanobind_arguments_t{
            nb::arg("tensor"),
            nb::kw_only(),
            nb::arg("head_size") = nb::none(),
            nb::arg("attention_mask") = nb::none(),
            nb::arg("program_config").noconvert() = ttnn::SoftmaxDefaultProgramConfig{},
            nb::arg("causal_mask") = false,
            nb::arg("memory_config") = nb::none()});
}

}  // namespace ttnn::operations::transformer
