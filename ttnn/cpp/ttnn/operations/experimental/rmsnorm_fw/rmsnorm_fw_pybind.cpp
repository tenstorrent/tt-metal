// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/pybind11/decorators.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/experimental/rmsnorm_fw/rmsnorm_fw.hpp"
#include "ttnn/operations/experimental/rmsnorm_fw/rmsnorm_fw_pybind.hpp"

namespace ttnn::operations::experimental::rmsnorm_fw::detail {

namespace py = pybind11;

void bind_experimental_rmsnorm_fw_operation(py::module& module) {
    std::string doc = "doc";
    using OperationType = decltype(ttnn::experimental::rmsnorm_fw);
    bind_registered_operation(
        module,
        ttnn::experimental::rmsnorm_fw,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& input,
               const Tensor& gamma,
               const bool return_intermediates,
               const float epsilon) { return self(input, gamma, return_intermediates, epsilon); },
            py::arg("input"),
            py::arg("gamma"),
            py::arg("return_intermediates"),
            py::arg("epsilon")});
}

}  // namespace ttnn::operations::experimental::rmsnorm_fw::detail
