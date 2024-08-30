// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_adam_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_adam/moreh_adam.hpp"

namespace ttnn::operations::moreh::moreh_adam {
void bind_moreh_adam_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_adam,
        "Moreh Adam Operation",
        ttnn::pybind_arguments_t{
            py::arg("param_in"),
            py::arg("grad"),
            py::arg("exp_avg_in"),
            py::arg("exp_avg_sq_in"),

            py::arg("lr"),
            py::arg("beta1"),
            py::arg("beta2"),
            py::arg("eps"),
            py::arg("weight_decay"),
            py::arg("step"),
            py::arg("amsgrad"),

            py::arg("max_exp_avg_sq_in") = std::nullopt,
            py::arg("param_out") = std::nullopt,
            py::arg("exp_avg_out") = std::nullopt,
            py::arg("exp_avg_sq_out") = std::nullopt,
            py::arg("max_exp_avg_sq_out") = std::nullopt,

            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_adam
