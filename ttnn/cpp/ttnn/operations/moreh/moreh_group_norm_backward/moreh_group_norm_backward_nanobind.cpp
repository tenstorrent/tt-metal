// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_backward_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "moreh_group_norm_backward.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {

void bind_moreh_group_norm_backward_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_group_norm_backward,
        "Moreh Group Norm Backward Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("output_grad"),
            nb::arg("input"),
            nb::arg("mean"),
            nb::arg("rstd"),
            nb::arg("num_groups"),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, false, false},
            nb::arg("gamma") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("gamma_grad") = nb::none(),
            nb::arg("beta_grad") = nb::none(),
            nb::arg("input_grad_memory_config") = nb::none(),
            nb::arg("gamma_grad_memory_config") = nb::none(),
            nb::arg("beta_grad_memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()

        });
}
}  // namespace ttnn::operations::moreh::moreh_group_norm_backward
