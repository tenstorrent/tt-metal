// SPDX-License-Identifier: Apache-2.0
#include "my_matmul_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/my_matmul/my_matmul.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::my_matmul {

void bind_my_matmul_operation(nb::module_& mod) {
    ttnn::bind_function<"my_matmul">(
        mod,
        R"doc(my_matmul(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor) -> ttnn.Tensor

Naive single-core tiled matmul: C = A @ B.)doc",
        &ttnn::my_matmul,
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"));
}

}  // namespace ttnn::operations::my_matmul
