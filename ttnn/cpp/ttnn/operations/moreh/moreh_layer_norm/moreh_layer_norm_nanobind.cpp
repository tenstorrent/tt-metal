// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "moreh_layer_norm.hpp"
#include "ttnn/types.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh::moreh_layer_norm {

void bind_moreh_layer_norm_operation(nb::module_& mod) {
    mod.def(
        "moreh_layer_norm",
        &ttnn::moreh_layer_norm,
        "Moreh Layer Norm Operation",
        nb::arg("input"),
        nb::arg("normalized_dims"),
        nb::arg("eps") = 1e-5f,
        nb::arg("gamma") = nb::none(),
        nb::arg("beta") = nb::none(),
        nb::kw_only(),
        nb::arg("output") = nb::none(),
        nb::arg("mean") = nb::none(),
        nb::arg("rstd") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::moreh::moreh_layer_norm
