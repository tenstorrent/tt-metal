// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "moreh_group_norm.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::moreh::moreh_group_norm {
void bind_moreh_group_norm_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_group_norm,
        "Moreh Group Norm Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::arg("num_groups"),
            nb::arg("eps") = 1e-5f,
            nb::arg("gamma") = std::nullopt,
            nb::arg("beta") = std::nullopt,
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, false, false},
            nb::arg("output") = std::nullopt,
            nb::arg("mean") = std::nullopt,
            nb::arg("rstd") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("mean_memory_config") = std::nullopt,
            nb::arg("rstd_memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt

        });
}
}  // namespace ttnn::operations::moreh::moreh_group_norm
