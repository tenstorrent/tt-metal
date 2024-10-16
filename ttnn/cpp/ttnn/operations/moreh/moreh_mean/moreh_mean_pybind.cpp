// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_mean_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean.hpp"

namespace ttnn::operations::moreh::moreh_mean {
void bind_moreh_mean_operation(py::module& module) {
    bind_registered_operation(module,
                              ttnn::moreh_mean,
                              "Moreh Mean Operation",
                              ttnn::pybind_arguments_t{py::arg("input"),
                                                       py::kw_only(),
                                                       py::arg("dim"),
                                                       py::arg("keepdim") = false,
                                                       py::arg("divisor") = std::nullopt,
                                                       py::arg("output") = std::nullopt,
                                                       py::arg("memory_config") = std::nullopt,
                                                       py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_mean
