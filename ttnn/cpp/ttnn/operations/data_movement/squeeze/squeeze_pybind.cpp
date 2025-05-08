// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "squeeze_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_squeeze(pybind11::module& module, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self, const ttnn::Tensor& input_tensor, const pybind11::object& dim)
                -> ttnn::Tensor {
                if (dim.is_none()) {  // None
                    return self(input_tensor);
                } else if (pybind11::isinstance<pybind11::int_>(dim)) {  // int
                    return self(input_tensor, dim.cast<int>());
                } else if (pybind11::isinstance<pybind11::list>(dim)) {  // List[int]
                    auto dims = dim.cast<ttnn::SmallVector<int>>();
                    return self(input_tensor, dims);
                } else {
                    throw std::invalid_argument("dim must be an int, a list of ints, or None");
                }
            },
            py::arg("input_tensor"),
            py::arg("dim") = pybind11::none()  // Default value is None
        });
}

}  // namespace detail

void py_bind_squeeze(pybind11::module& module) {
    detail::bind_squeeze(
        module,
        ttnn::squeeze,
        R"doc(squeeze(input_tensor: ttnn.Tensor,  dim: int) -> ttnn.Tensor

        Returns a tensor with the specified dimensions squeezed. If `dim` is not provided, all dimensions of size 1 will be squeezed. If `dim` is an integer, only the specified dimension will be squeezed. If `dim` is a list of integers, all specified dimensions will be squeezed.

        If a specified dimension in `dim` does not have size 1, it will be ignored.

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
