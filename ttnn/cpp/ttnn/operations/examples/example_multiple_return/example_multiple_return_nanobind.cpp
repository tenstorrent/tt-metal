// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "example_multiple_return_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/examples/example_multiple_return/example_multiple_return.hpp"

namespace ttnn::operations::examples {

void bind_example_multiple_return_operation(nb::module_& mod) {
    ttnn::bind_function<"composite_example_multiple_return">(
        mod,
        R"doc(
        Example operation that returns multiple tensors.

        Args:
            input_tensor (ttnn.Tensor): The input tensor.

        Returns:
            List[ttnn.Tensor]: List of output tensors.
        )doc",
        ttnn::overload_t(
            &ttnn::composite_example_multiple_return,
            nb::arg("input_tensor"),
            nb::arg("return_output1"),
            nb::arg("return_output2")));
}

}  // namespace ttnn::operations::examples
