// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/transformer/concatenate_heads/concatenate_heads.hpp"


namespace ttnn::operations::experimental::transformer::detail {

namespace py = pybind11;

void bind_concatenate_heads(py::module& module) {

    auto concatenate_heads_doc =
        R"doc(concatenate_heads(input_tensor: ttnn.Tensor, compute_with_storage_grid_size: ttnn.CoreCoord: *, memory_config: Optional[MemoryConfig] = None) -> ttnn.Tensor

            Reshuffles a [9, 16, 384, 64] ttnn.Layout.TILE BFLOAT8_B tensor into a tensor with shape [9, 1, 384, 1024].

            Args:
                * :attr:`input_tensor`: Input Tensor with shape [9, 16, 384, 64], dtype BFLOAT8_B and layout ttnn.Layout.TILE
                * :attr:`compute_with_storage_grid_size`: Compute Grid

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor, if None then it gets set to input_tensor.memory_config()
        )doc";

    using OperationType = decltype(ttnn::experimental::concatenate_heads);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::concatenate_heads,
        concatenate_heads_doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const CoreCoord& compute_with_storage_grid_size,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                std::optional<ttnn::Tensor> optional_output_tensor,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, compute_with_storage_grid_size, memory_config, optional_output_tensor);
                },
                py::arg("input_tensor").noconvert(),
                py::arg("compute_with_storage_grid_size").noconvert(),
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("output_tensor") = std::nullopt,
                py::arg("queue_id") = 0});

}

}  // namespace ttnn::operations::experimental::transformer::detail
