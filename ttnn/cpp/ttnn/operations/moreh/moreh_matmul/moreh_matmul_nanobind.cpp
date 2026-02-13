// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_matmul_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_matmul.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/moreh/moreh_matmul/device/moreh_matmul_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_matmul {
void bind_moreh_matmul_operation(nb::module_& mod) {
    ttnn::bind_function<"moreh_matmul">(
        mod,
        "Moreh Matmul Operation",
        ttnn::overload_t(
            &ttnn::moreh_matmul,
            nb::arg("input").noconvert(),
            nb::arg("other").noconvert(),
            nb::kw_only(),
            nb::arg("transpose_input") = false,
            nb::arg("transpose_other") = false,
            nb::arg("output") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));
}
}  // namespace ttnn::operations::moreh::moreh_matmul
