// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/plusone/plusone.hpp"
#include "ttnn/operations/experimental/plusone/plusone_pybind.hpp"

namespace ttnn::operations::experimental::plusone::detail {
namespace py = pybind11;
void bind_experimental_plusone_operation(py::module& module) {
    auto doc =
        R"doc(plus_one(input_tensor: ttnn.Tensor) -> ttnn.Tensor

            Returns input tensor elements increased by 1.
            Input tensor must have UINT32 data type, ROW_MAJOR layout, and 1-D shape.
            This op only gives decent performance for small tensors (up to 100 elements).

            Equivalent pytorch code:

            .. code-block:: python

                return torch.add(input_tensor, 1)

            Args:
                * :attr:`input_tensor`: Input Tensor for plusone. 

        )doc";

    using OperationType = decltype(ttnn::plus_one);
    bind_registered_operation(module,
                              ttnn::plus_one,
                              doc,
                              ttnn::pybind_overload_t{[](const OperationType& self, const ttnn::Tensor& input_tensor) {
                                                          return self(input_tensor);
                                                      },
                                                      py::arg("input_tensor").noconvert()});
}

}  // namespace ttnn::operations::experimental::plusone::detail
