// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_pybind.hpp"

#include "batch_norm.hpp"

#include "pybind11/decorators.hpp"
namespace py = pybind11;
namespace ttnn::operations::normalization::detail {
void bind_batch_norm_operation(pybind11::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::batch_norm,
        "batch_norm Operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::kw_only(),
            py::arg("running_mean") = std::nullopt,
            py::arg("running_var") = std::nullopt,
            py::arg("training") = false,
            py::arg("eps") = 1e-05,
            py::arg("weight") = std::nullopt,
            py::arg("bias") = std::nullopt,
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt

        });
}
}  // namespace ttnn::operations::normalization::detail
