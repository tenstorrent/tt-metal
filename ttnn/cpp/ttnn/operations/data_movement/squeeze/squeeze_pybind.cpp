// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "squeeze_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_squeeze(pybind11::module& module, const data_movement_operation_t& operation, const char* doc) {
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

void py_bind_squeeze(pybind11::module& module) {
    detail::bind_squeeze(module,
                         ttnn::squeeze,
                         R"doc(squeeze(input_tensor: ttnn.Tensor,  dim: int) -> ttnn.Tensor

        Returns a tensor squeezed at the specified dimension. Pytorch supports a tuple as well as a single scalar value for dim, currently our version only supports scalar values. We will address this in the future. If input_tensor.shape()[dim] is not 1, squeeze will be ignored for that shape.

        Equivalent pytorch code:

        .. code-block:: python
            input_tensor = torch.rand((1,1,1,256), dtype=torch.bfloat16)
            output_tensor = torch.squeeze(input_tensor, 2) # tensor of shape (1,1,256), where at dimension 2 we removed it



        Args:
            * :attr:`input_tensor`: Input Tensor.
            * :attr:`dim`: Dim where we want to squeeze


        )doc");
}

}  // namespace ttnn::operations::data_movement
