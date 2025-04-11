// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_clip_grad_norm.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {

void bind_moreh_clip_grad_norm_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_clip_grad_norm,
        "moreh_clip_grad_norm",
        ttnn::nanobind_arguments_t{
            nb::arg("inputs"),
            nb::arg("max_norm"),
            nb::arg("norm_type") = 2.0f,
            nb::arg("error_if_nonfinite") = false,
            nb::kw_only(),
            nb::arg("total_norm") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm
