// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"
#include <pybind11/pybind11.h>
#include "ttnn/operations/ccl/ccl_host_types.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace ttnn::operations::ccl {

void py_bind_reduce_scatter(pybind11::module& module);

}  // namespace ttnn::operations::ccl
