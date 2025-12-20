// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "narrow_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/data_movement/narrow/narrow.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace {

template <typename data_movement_operation_t>
void bind_narrow_op(nb::module_& mod, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const data_movement_operation_t& self,
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

}  // namespace

void bind_narrow(nb::module_& mod) {
    bind_narrow_op(
        mod,
        ttnn::narrow,
        R"doc(
        This is a zero-cost narrow operation that returns the narrowed version of the tensor. The returned tensor shares the same data buffer as the input tensor, but dimension dim will have the specified length, starting from the start index.

        Args:
            * input_tensor: Input Tensor.
            * dim: Dimension to narrow.
            * start: Starting index of the narrow operation.
            * length: Length of the narrow dimension.

        Returns:
            ttnn.Tensor: a reference to the narrowed tensor but with the new shape.

        Example:

            >>> tensor = ttnn.from_torch(torch.rand((32, 16, 16, 4), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.narrow(tensor, 0, 16, 8)

        )doc");
}
}  // namespace ttnn::operations::data_movement
