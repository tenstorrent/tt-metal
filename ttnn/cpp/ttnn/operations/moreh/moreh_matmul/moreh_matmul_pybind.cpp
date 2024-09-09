// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_matmul_pybind.hpp"

#include "moreh_matmul.hpp"
#include "pybind11/cast.h"
#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_matmul/device/moreh_matmul_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_matmul {
void bind_moreh_matmul_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_matmul,
        "Moreh moreh_matmul Operation",
        ttnn::pybind_arguments_t{
            py::arg("input").noconvert(),
            py::arg("other").noconvert(),
            py::kw_only(),
            py::arg("transpose_input").noconvert() = false,
            py::arg("transpose_other").noconvert() = false,
            py::arg("output").noconvert() = std::nullopt,
            py::arg("bias").noconvert() = std::nullopt,
            py::arg("output_mem_config").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_matmul
