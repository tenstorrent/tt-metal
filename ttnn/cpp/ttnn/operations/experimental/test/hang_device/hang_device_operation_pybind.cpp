// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hang_device_operation_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn-pybind/decorators.hpp"
#include "hang_device_operation.hpp"
namespace ttnn::operations::experimental::test {
namespace py = pybind11;

void bind_test_hang_device_operation(py::module& module) {
    auto doc =
        R"doc(
            hang_device_operation(input_tensor: ttnn.Tensor) -> ttnn.Tensor

            Hangs the device, use for testing graph capture.
            Used for debugging purposes, please avoid to use in any production code
            Args:
                * :attr:`input_tensor`: Input Tensor.
        )doc";
    using OperationType = decltype(ttnn::prim::hang_device_operation);
    ttnn::bind_registered_operation(
        module,
        ttnn::prim::hang_device_operation,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self, const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                return self(input_tensor);
            },
            py::arg("input_tensor"),
        });
}
}  // namespace ttnn::operations::experimental::test
