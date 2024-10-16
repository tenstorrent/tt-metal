// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_linear_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_linear/moreh_linear.hpp"

namespace ttnn::operations::moreh::moreh_linear {
void bind_moreh_linear_operation(py::module& module) {
    bind_registered_operation(module,
                              ttnn::moreh_linear,
                              "Moreh Linear Operation",
                              ttnn::pybind_arguments_t{py::arg("input"),
                                                       py::arg("weight"),
                                                       py::kw_only(),
                                                       py::arg("bias") = std::nullopt,
                                                       py::arg("output") = std::nullopt,
                                                       py::arg("memory_config") = std::nullopt,
                                                       py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_linear
