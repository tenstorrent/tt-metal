// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_pybind.hpp"

#include <pybind11/pybind11.h>

#include "moreh_clip_grad_norm.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {

void bind_moreh_clip_grad_norm_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_clip_grad_norm,
        "moreh_clip_grad_norm",
        ttnn::pybind_arguments_t{
            py::arg("inputs"),
            py::arg("max_norm"),
            py::arg("norm_type") = 2.0f,
            py::arg("error_if_nonfinite") = false,
            py::kw_only(),
            py::arg("total_norm") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm
