// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_arange_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_arange.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh::moreh_arange {
void bind_moreh_arange_operation(nb::module_& mod) {
    mod.def(
        "moreh_arange",
        &ttnn::moreh_arange,
        "Moreh Arange Operation",
        nb::arg("start") = 0,
        nb::arg("end"),
        nb::arg("step") = 1,
        nb::arg("device"),
        nb::kw_only(),
        nb::arg("output") = nb::none(),
        nb::arg("untilize_out") = false,
        nb::arg("dtype") = DataType::BFLOAT16,
        nb::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG);
}
}  // namespace ttnn::operations::moreh::moreh_arange
