// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "split_query_key_value_and_split_heads.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_split_qkv(nb::module_& mod) {
    ttnn::bind_function<"split_query_key_value_and_split_heads", "ttnn.experimental.">(
        mod,
        R"doc(
            Splits [B, 1, 384, 3072] fused qkv matrix into 3 heads with shapes [B, 16, 384, 64], [B, 16, 64, 384], and [B, 16, 384, 64]. Supports both sharded and interleaved inputs.

            Args:
                * :attr:`input_tensor`: Input Tensor
                * :attr:`compute_with_storage_grid_size`: Compute Grid

            Keyword Args:
                * :attr:`num_heads`: Number of heads to split the tensor into
                * :attr:`memory_config`: Memory Config of the output tensor, if None then it gets set to input_tensor.memory_config()
                * :attr:`output_tensors`: preallocated output tensors
        )doc",
        &ttnn::experimental::split_query_key_value_and_split_heads,
        nb::arg("input_tensor").noconvert(),
        nb::arg("compute_with_storage_grid_size").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("num_heads") = 16,
        nb::arg("output_tensors") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer::detail
