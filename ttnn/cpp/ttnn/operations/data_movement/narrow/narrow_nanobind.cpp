// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "narrow_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/data_movement/narrow/narrow.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_narrow(nb::module_& mod) {
    const auto* doc = R"doc(
        This is a zero-cost narrow operation that returns the narrowed version of the tensor. The returned tensor shares the same data buffer as the input tensor, but dimension dim will have the specified length, starting from the start index.

        Note:
            * Input tensor must be stored on the device.
            * Currently supports only DRAM INTERLEAVED or L1 sharded tensors.
            * For DRAM INTERLEAVED tensors, narrow can only be performed on the first non-trivial dimension, with start pointing to the first bank.
            * For L1 sharded tensors, narrow is supported only in specific cases: when the narrowed region consists of complete full shards, or when the narrowed region spans multiple shards with the same page offset.
            * Supports all pairs of layout and data types that are supported by ttnn.

        Args:
            * input_tensor: Input Tensor.
            * dim: Dimension to narrow.
            * start: Starting index of the narrow operation.
            * length: Length of the narrow dimension.

        Returns:
            ttnn.Tensor: a reference to the narrowed tensor but with the new shape.

        Example:

            >>> tensor = ttnn.rand((32, 16, 16, 4), dtype=ttnn.bfloat16, device=device)
            >>> output = ttnn.narrow(tensor, 0, 12, 8)

        )doc";
    bind_registered_operation(
        mod,
        ttnn::narrow,
        doc,
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::narrow)& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               const int32_t start,
               const uint32_t length) -> ttnn::Tensor { return self(input_tensor, dim, start, length); },
            nb::arg("input_tensor"),
            nb::arg("dim"),
            nb::arg("start"),
            nb::arg("length"),
        });
}
}  // namespace ttnn::operations::data_movement
