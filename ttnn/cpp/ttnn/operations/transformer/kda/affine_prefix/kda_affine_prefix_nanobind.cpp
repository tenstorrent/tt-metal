// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_affine_prefix_nanobind.hpp"

#include "kda_affine_prefix.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>

namespace ttnn::operations::transformer {

void bind_kda_affine_prefix(nb::module_& mod) {
    ttnn::bind_function<"kda_affine_prefix", "ttnn.transformer.">(
        mod,
        R"doc(Compute the entry state for each grouped KDA affine summary.)doc",
        &ttnn::transformer::kda_affine_prefix,
        nb::arg("transform_a").noconvert(),
        nb::arg("transform_b").noconvert(),
        nb::arg("initial_state").noconvert(),
        nb::arg("groups_per_head"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
