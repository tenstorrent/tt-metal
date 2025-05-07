// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unsqueeze_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_unsqueeze(pybind11::module& module, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self, const ttnn::Tensor& input_tensor, const int dim) -> ttnn::Tensor {
                return self(input_tensor, dim);
            },
            py::arg("input_tensor"),
            py::arg("dim")});
}

}  // namespace detail

void py_bind_unsqueeze(pybind11::module& module) {
    detail::bind_unsqueeze(
        module,
        ttnn::unsqueeze,
        R"doc(unsqueeze(input_tensor: ttnn.Tensor,  dim: int) -> ttnn.Tensor

        Returns a tensor unsqueezed at the specified dimension

        Equivalent pytorch code:

        .. code-block:: python
            input_tensor = torch.rand((1,1,256), dtype=torch.bfloat16)
            output_tensor = torch.unsqueeze(input_tensor, 2) # tensor of shape (1,1,1,256), where at dimension 2 we added a new dim of size 1



        Args:
            * :attr:`input_tensor`: Input Tensor.
            * :attr:`dim`: Dim where we want to unsqueeze (add a new dimension of size 1)


        )doc");
}

}  // namespace ttnn::operations::data_movement
