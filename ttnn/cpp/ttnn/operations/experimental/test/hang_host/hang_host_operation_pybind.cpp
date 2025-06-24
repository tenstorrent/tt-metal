// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hang_host_operation_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn-pybind/decorators.hpp"
#include "hang_host_operation.hpp"

namespace ttnn::operations::experimental::test {
namespace py = pybind11;

void bind_test_hang_host_operation(py::module& module) {
    auto doc =
        R"doc(
            hang_host_operation(input_tensor: ttnn.Tensor) -> ttnn.Tensor

            Hangs the host, use for testing graph capture.
            Remember to compile with --ttnn-enable-operation-timeout to use this operation,
            otherwise it will just return the input tensor

            Args:
                * :attr:`input_tensor`: Input Tensor.
        )doc";

    using OperationType = decltype(ttnn::experimental::test::hang_host_operation);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::test::hang_host_operation,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self, const ttnn::Tensor& input_tensor) { return self(input_tensor); },
            py::arg("input_tensor"),
        });
}
}  // namespace ttnn::operations::experimental::test
