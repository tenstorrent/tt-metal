// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn/operations/data_movement.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace data_movement {
void py_module(py::module& module) {

    module.def("permute", &permute,
        py::arg("input_tensor"),
        py::arg("order"),
        R"doc(
Permutes :attr:`input_tensor` using :attr:`order`.

Args:
    * :attr:`input_tensor`: the input tensor
    * :attr:`order`: the desired ordering of dimensions.

Example::

    >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
    >>> output = ttnn.permute(tensor, (0, 1, 3, 2))
    >>> print(output.shape)
    [1, 1, 32, 64]

    )doc");
}

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
