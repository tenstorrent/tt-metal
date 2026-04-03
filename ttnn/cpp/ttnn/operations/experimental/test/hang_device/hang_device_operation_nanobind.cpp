// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hang_device_operation_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "hang_device_operation.hpp"

namespace ttnn::operations::experimental::test {

void bind_test_hang_device_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Hangs the device, use for testing graph capture.
            Used for debugging purposes, please avoid to use in any production code
            Args:
                * :attr:`input_tensor`: Input Tensor.
        )doc";
    ttnn::bind_function<"test_hang_device_operation">(
        mod, doc, &ttnn::operations::experimental::test::test_hang_device_operation, nb::arg("input_tensor"));
}
}  // namespace ttnn::operations::experimental::test
