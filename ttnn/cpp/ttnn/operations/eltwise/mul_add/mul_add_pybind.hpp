// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"
#include "ttnn/operations/eltwise/mul_add/mul_add.hpp"

#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::mul_add {

void py_module(pybind11::module& module) {
    bind_registered_operation(
        module,
        ttnn::mul_add,
        R"doc(mul_add(input_tensor_a: ttnn.Tensor,  input_tensor_b: ttnn.Tensor, input_tensor_c: ttnn.Tensor) -> ttnn.Tensor

    Mul + add.

    Args:
        * :attr:`input_tensor_a`: Input Tensor.
        * :attr:`input_tensor_b`: Input Tensor.
        * :attr:`input_tensor_c`: Input Tensor.

    )doc",
        ttnn::pybind_arguments_t{py::arg("input_tensor_a"), py::arg("input_tensor_b"), py::arg("input_tensor_c")});
}

}  // namespace ttnn::operations::mul_add
