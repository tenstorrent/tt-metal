// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "pybind11/pytypes.h"
#include "ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward {
void bind_moreh_layer_norm_backward_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_layer_norm_backward,
        "Moreh Layer Norm Backward Operation",
        ttnn::pybind_arguments_t{
            py::arg("output_grad"),
            py::arg("input"),
            py::arg("mean"),
            py::arg("rstd"),
            py::arg("normalized_dims"),
            py::kw_only(),
            py::arg("gamma") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("gamma_grad") = std::nullopt,
            py::arg("beta_grad") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward
