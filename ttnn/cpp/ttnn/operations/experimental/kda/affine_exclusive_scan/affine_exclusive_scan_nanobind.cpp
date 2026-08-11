// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "affine_exclusive_scan_nanobind.hpp"

#include "affine_exclusive_scan.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>

namespace ttnn::operations::experimental::kda::affine_exclusive_scan::detail {

void bind_affine_exclusive_scan(nb::module_& mod) {
    ttnn::bind_function<"affine_exclusive_scan", "ttnn.experimental.kda.">(
        mod,
        R"doc(Compute the entry state for each grouped KDA affine summary.)doc",
        &ttnn::experimental::kda::affine_exclusive_scan,
        nb::arg("a").noconvert(),
        nb::arg("b").noconvert(),
        nb::arg("initial_state").noconvert(),
        nb::arg("groups_per_head"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::affine_exclusive_scan::detail
