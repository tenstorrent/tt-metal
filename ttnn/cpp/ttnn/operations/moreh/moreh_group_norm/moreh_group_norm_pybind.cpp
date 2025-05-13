// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_pybind.hpp"

#include "moreh_group_norm.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::moreh::moreh_group_norm {
void bind_moreh_group_norm_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_group_norm,
        "Moreh Group Norm Operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("num_groups"),
            py::arg("eps") = 1e-5f,
            py::arg("gamma") = std::nullopt,
            py::arg("beta") = std::nullopt,
            py::kw_only(),
            py::arg("are_required_outputs") = std::vector<bool>{true, false, false},
            py::arg("output") = std::nullopt,
            py::arg("mean") = std::nullopt,
            py::arg("rstd") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("mean_memory_config") = std::nullopt,
            py::arg("rstd_memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt

        });
}
}  // namespace ttnn::operations::moreh::moreh_group_norm
