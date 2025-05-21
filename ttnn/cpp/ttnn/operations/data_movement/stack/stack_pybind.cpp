// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stack_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/data_movement/stack/stack.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_stack(pybind11::module& module, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self, const std::vector<ttnn::Tensor>& input_tensors, const int dim)
                -> ttnn::Tensor { return self(input_tensors, dim); },
            py::arg("input_tensors"),
            py::arg("dim")});
}

}  // namespace detail

void py_bind_stack(pybind11::module& module) {
    detail::bind_stack(
        module,
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
