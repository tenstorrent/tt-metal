// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "pybind11/pytypes.h"
#include "ttnn/operations/moreh/moreh_layer_norm/moreh_layer_norm.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm {
void bind_moreh_layer_norm_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_layer_norm,
        "Moreh Layer Norm Operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("normalized_dims"),
            py::arg("eps") = 1e-5f,
            py::arg("gamma") = std::nullopt,
            py::arg("beta") = std::nullopt,
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("mean") = std::nullopt,
            py::arg("rstd") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm
