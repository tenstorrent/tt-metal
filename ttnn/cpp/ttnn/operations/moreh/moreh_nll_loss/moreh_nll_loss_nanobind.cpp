// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "moreh_nll_loss.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss {

void bind_moreh_nll_loss_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Compute backward for nll_loss operation with reduction set to None
        )doc";

    ttnn::bind_function<"moreh_nll_loss">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::moreh_nll_loss,
            nb::arg("input_tensor"),
            nb::arg("target_tensor"),
            nb::arg("reduction"),
            nb::kw_only(),
            nb::arg("weight_tensor") = nb::none(),
            nb::arg("divisor_tensor") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("ignore_index") = -100,
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss
