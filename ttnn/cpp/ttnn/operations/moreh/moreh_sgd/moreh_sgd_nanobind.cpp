// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sgd_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "moreh_sgd.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh::moreh_sgd {
void bind_moreh_sgd_operation(nb::module_& mod) {
    mod.def(
        "moreh_sgd",
        &ttnn::moreh_sgd,
        "Moreh SGD Operation",
        nb::arg("param_in"),
        nb::arg("grad"),
        nb::arg("momentum_buffer_in") = nb::none(),
        nb::arg("param_out") = nb::none(),
        nb::arg("momentum_buffer_out") = nb::none(),
        nb::arg("lr") = 1e-3,
        nb::arg("momentum") = 0,
        nb::arg("dampening") = 0,
        nb::arg("weight_decay") = 0,
        nb::arg("nesterov") = false,
        nb::kw_only(),
        nb::arg("momentum_initialized"),
        nb::arg("param_out_memory_config") = nb::none(),
        nb::arg("momentum_buffer_out_memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}
}  // namespace ttnn::operations::moreh::moreh_sgd
