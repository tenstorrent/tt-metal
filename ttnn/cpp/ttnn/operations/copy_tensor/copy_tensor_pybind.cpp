// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_tensor_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "copy_tensor/copy_tensor_pybind.hpp"

namespace ttnn::operations::copy_tensor {

namespace py = pybind11;

void py_module(py::module& module) { copy_tensor::detail::bind_copy_tensor(module); }

}  // namespace ttnn::operations::copy_tensor
