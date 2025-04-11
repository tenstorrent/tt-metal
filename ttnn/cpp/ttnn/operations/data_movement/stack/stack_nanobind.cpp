// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stack_nanobind.hpp"

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/data_movement/stack/stack.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace {

template <typename data_movement_operation_t>
void bind_stack(nb::module_& mod, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const data_movement_operation_t& self, const std::vector<ttnn::Tensor>& input_tensors, const int dim)
                -> ttnn::Tensor { return self(input_tensors, dim); },
            nb::arg("input_tensors"),
            nb::arg("dim")});
}

}  // namespace

void bind_stack(nb::module_& mod) {
    bind_stack(
        mod,
        ttnn::stack,
        R"doc(stack(input_tensors: List[ttnn.Tensor], dim: int) -> ttnn.Tensor

        Stacks tensors along a new dimension.

        Args:
            * :attr:`input_tensors`: List of tensors to stack.
            * :attr:`dim`: Dimension along which to stack.

        Example:
           >>> input_tensor = ttnn.from_torch(torch.randn((2, 2), dtype=torch.bfloat16), device=device)
           >>> output = ttnn.stack((input_tensor,input_tensor), 1)

        )doc");
}

}  // namespace ttnn::operations::data_movement
