// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm_pybind.hpp"

#include "moreh_bmm.hpp"
#include "pybind11/decorators.hpp"

namespace ttnn::operations::moreh::moreh_bmm {
void bind_moreh_bmm_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_bmm,
        "Moreh BMM Operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("mat2"),
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_bmm
