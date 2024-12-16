// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/ccl/ccl_experimental_pybind.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul/all_gather_matmul_pybind.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce/all_reduce_pybind.hpp"

namespace py = pybind11;

namespace ttnn::operations::experimental::ccl {

void py_module(py::module& module) {
    ccl::py_bind_all_gather_matmul(module);
    ccl::py_bind_all_reduce(module);
}

}  // namespace ttnn::operations::experimental::ccl
