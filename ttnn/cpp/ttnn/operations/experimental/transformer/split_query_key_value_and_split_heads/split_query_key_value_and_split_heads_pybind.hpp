// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.hpp"


namespace ttnn::operations::experimental::transformer::detail {

namespace py = pybind11;

void bind_split_qkv(py::module& module) {
    using SplitOperationType = decltype(ttnn::experimental::split_query_key_value_and_split_heads);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::split_query_key_value_and_split_heads,
        R"doc(split_query_key_value_and_split_heads(input_tensor: ttnn.Tensor, compute_with_storage_grid_size: ttnn.CoreCoord: *, num_heads: int = 16, memory_config: Optional[MemoryConfig] = None) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]

            Splits [B, 1, 384, 3072] fused qkv matrix into 3 heads with shapes [B, 16, 384, 64], [B, 16, 64, 384], and [B, 16, 384, 64]. Supports both sharded and interleaved inputs.

            Args:
                * :attr:`input_tensor`: Input Tensor
                * :attr:`compute_with_storage_grid_size`: Compute Grid

            Keyword Args:
                * :attr:`num_heads`: Number of heads to split the tensor into
                * :attr:`memory_config`: Memory Config of the output tensor, if None then it gets set to input_tensor.memory_config()
                * :attr:`output_tensors`: preallocated output tensors
                * :attr:`queue_id`: command queue id
        )doc",
        ttnn::pybind_overload_t{
            [] (const SplitOperationType& self,
                const ttnn::Tensor& input_tensor,
                const CoreCoord& compute_with_storage_grid_size,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                const uint32_t num_heads,
                std::optional<std::vector<std::optional<ttnn::Tensor>>> optional_output_tensors,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, compute_with_storage_grid_size, memory_config, num_heads, optional_output_tensors);
                },
                py::arg("input_tensor").noconvert(),
                py::arg("compute_with_storage_grid_size").noconvert(),
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("num_heads") = 16,
                py::arg("output_tensors") = std::nullopt,
                py::arg("queue_id") = 0});

}

}  // namespace ttnn::operations::experimental::transformer::detail
