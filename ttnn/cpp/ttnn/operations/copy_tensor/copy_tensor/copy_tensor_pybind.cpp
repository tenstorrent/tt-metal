// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_tensor_pybind.hpp"
#include "cpp/pybind11/decorators.hpp"
#include "copy_tensor.hpp"

namespace ttnn::operations::copy_tensor::detail {

namespace py = pybind11;

void bind_copy_tensor_operation(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::copy_tensor,
        R"doc(
        Copy tensor operation.
        )doc",

        ttnn::pybind_arguments_t{
            py::arg("src_tensor"),
            py::arg("dst_tensor"),
        });
}

void bind_copy_tensor(py::module& module) { bind_copy_tensor_operation(module); }

}  // namespace ttnn::operations::copy_tensor::detail
