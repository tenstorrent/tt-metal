// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_backward_pybind.hpp"

#include "moreh_group_norm_backward.hpp"
#include "pybind11/decorators.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {
void bind_moreh_group_norm_backward_operation(py::module &module) {
    bind_registered_operation(
        module,
        ttnn::moreh_group_norm_backward,
        "Moreh Group Norm Backward Operation",
        ttnn::pybind_arguments_t{py::arg("output_grad"),
                                 py::arg("input"),
                                 py::arg("mean"),
                                 py::arg("rstd"),
                                 py::arg("num_groups"),
                                 py::kw_only(),
                                 py::arg("are_required_outputs") = std::vector<bool>{true, false, false},
                                 py::arg("gamma") = std::nullopt,
                                 py::arg("input_grad") = std::nullopt,
                                 py::arg("gamma_grad") = std::nullopt,
                                 py::arg("beta_grad") = std::nullopt,
                                 py::arg("input_grad_memory_config") = std::nullopt,
                                 py::arg("gamma_grad_memory_config") = std::nullopt,
                                 py::arg("beta_grad_memory_config") = std::nullopt,
                                 py::arg("compute_kernel_config") = std::nullopt

        });
}
}  // namespace ttnn::operations::moreh::moreh_group_norm_backward
