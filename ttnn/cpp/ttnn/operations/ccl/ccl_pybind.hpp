// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/all_gather/all_gather_pybind.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter_pybind.hpp"
#include "ttnn/operations/ccl/barrier/barrier_pybind.hpp"

// #include "ttnn/cpp/pybind11/decorators.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ttnn::operations::ccl {

void py_bind_common(pybind11::module& module);

void py_module(py::module& module) {
    ccl::py_bind_common(module);
    ccl::py_bind_all_gather(module);
    ccl::py_bind_reduce_scatter(module);
    ccl::py_bind_barrier(module);
}

}  // namespace ttnn::operations::ccl
