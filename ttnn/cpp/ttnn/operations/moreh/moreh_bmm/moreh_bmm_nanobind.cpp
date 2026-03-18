// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm_nanobind.hpp"

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_bmm.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::moreh::moreh_bmm {

void bind_moreh_bmm_operation(nb::module_& mod) {
    ttnn::bind_function<"moreh_bmm">(
        mod,
        "Moreh BMM Operation",
        ttnn::overload_t(
            &ttnn::moreh_bmm,
            nb::arg("input"),
            nb::arg("mat2"),
            nb::kw_only(),
            nb::arg("output") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));
}
}  // namespace ttnn::operations::moreh::moreh_bmm
