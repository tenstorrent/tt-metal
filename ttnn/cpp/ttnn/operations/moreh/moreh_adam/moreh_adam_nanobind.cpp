// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_adam_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_adam/moreh_adam.hpp"

namespace ttnn::operations::moreh::moreh_adam {
void bind_moreh_adam_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_adam,
        "Moreh Adam Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("param_in"),
            nb::arg("grad"),
            nb::arg("exp_avg_in"),
            nb::arg("exp_avg_sq_in"),

            nb::kw_only(),
            nb::arg("lr") = 0.001f,
            nb::arg("beta1") = 0.9f,
            nb::arg("beta2") = 0.999f,
            nb::arg("eps") = 1e-8f,
            nb::arg("weight_decay") = 0.0f,
            nb::arg("step") = 0,
            nb::arg("amsgrad") = false,

            nb::arg("max_exp_avg_sq_in") = nb::none(),
            nb::arg("param_out") = nb::none(),
            nb::arg("exp_avg_out") = nb::none(),
            nb::arg("exp_avg_sq_out") = nb::none(),
            nb::arg("max_exp_avg_sq_out") = nb::none(),

            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_adam
