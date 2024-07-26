// SPDX-FileCopyrightText: © 2024 BOS
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "ttnn/operations/test_ops/test_concat/test_concat_pybind.hpp"

namespace ttnn::operations::test_ops {

void py_module(py::module& module) { 
    detail::bind_test_concat(module, ttnn::test_concat); 
}

}