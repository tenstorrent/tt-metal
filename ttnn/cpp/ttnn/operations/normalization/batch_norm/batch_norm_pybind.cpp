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
            py::arg("num_groups"),
            py::arg("eps") = 1e-5f,
            py::arg("gamma") = std::nullopt,
            py::arg("beta") = std::nullopt,
            py::kw_only(),
            py::arg("are_required_outputs") = std::vector<bool>{true, true, true},
            py::arg("output") = std::nullopt,
            py::arg("mean") = std::nullopt,
            py::arg("rstd") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("mean_memory_config") = std::nullopt,
            py::arg("rstd_memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt

        });
}
}  // namespace ttnn::operations::normalization::detail
