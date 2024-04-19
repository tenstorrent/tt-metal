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
    module.def(
        "concatenate_heads",
        [](const ttnn::Tensor& input_tensor, const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt)
            -> ttnn::Tensor { return ttnn::operations::transformer::concatenate_heads(input_tensor, memory_config); },
        py::arg("input_tensor"),
        py::kw_only(),
        py::arg("memory_config") = std::nullopt);

    module.def(
        "attention_softmax_",
        [](const ttnn::Tensor& tensor,
           const std::optional<int> head_size,
           const std::optional<const ttnn::Tensor>& attention_mask,
           const tt::operations::primary::transformers::SoftmaxProgramConfig& program_config,
           const std::optional<bool> causal_mask,
           const std::optional<ttnn::MemoryConfig>& memory_config) -> ttnn::Tensor {
            return ttnn::operations::transformer::attention_softmax_(
                tensor, head_size, attention_mask, program_config, causal_mask, memory_config);
        },
        py::arg("tensor"),
        py::kw_only(),
        py::arg("head_size") = std::nullopt,
        py::arg("attention_mask") = std::nullopt,
        py::arg("program_config").noconvert() = tt::operations::primary::transformers::SoftmaxDefaultProgramConfig{},
        py::arg("causal_mask") = false,
        py::arg("memory_config") = std::nullopt);
}
}  // namespace transformer
}  // namespace operations
}  // namespace ttnn
