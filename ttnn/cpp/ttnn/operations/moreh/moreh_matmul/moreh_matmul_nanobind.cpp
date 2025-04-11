// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_matmul_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_matmul.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_matmul/device/moreh_matmul_device_operation.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh::moreh_matmul {
void bind_moreh_matmul_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_matmul,
        "Moreh Matmul Operation",
        ttnn::nanobind_arguments_t{
            nb::arg("input").noconvert(),
            nb::arg("other").noconvert(),
            nb::kw_only(),
            nb::arg("transpose_input").noconvert() = false,
            nb::arg("transpose_other").noconvert() = false,
            nb::arg("output").noconvert() = std::nullopt,
            nb::arg("bias").noconvert() = std::nullopt,
            nb::arg("memory_config").noconvert() = std::nullopt,
            nb::arg("compute_kernel_config").noconvert() = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_matmul
