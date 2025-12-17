// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm_nanobind.hpp"

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_bmm.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::moreh::moreh_bmm {

void bind_moreh_bmm_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_bmm,
        "Moreh BMM Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::arg("mat2"),
            nb::kw_only(),
            nb::arg("output") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_bmm
