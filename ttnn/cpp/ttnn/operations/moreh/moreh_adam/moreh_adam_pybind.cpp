// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_adam_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "pybind11/pytypes.h"
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

            py::kw_only(),
            py::arg("lr") = 0.001f,
            py::arg("beta1") = 0.9f,
            py::arg("beta2") = 0.999f,
            py::arg("eps") = 1e-8f,
            py::arg("weight_decay") = 0.0f,
            py::arg("step") = 0,
            py::arg("amsgrad") = false,

            py::arg("max_exp_avg_sq_in") = std::nullopt,
            py::arg("param_out") = std::nullopt,
            py::arg("exp_avg_out") = std::nullopt,
            py::arg("exp_avg_sq_out") = std::nullopt,
            py::arg("max_exp_avg_sq_out") = std::nullopt,

            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_adam
