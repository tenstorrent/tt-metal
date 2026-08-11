// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "reduce_affine_transforms_nanobind.hpp"

#include "reduce_affine_transforms.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

namespace ttnn::operations::experimental::kda::reduce_affine_transforms::detail {

void bind_reduce_affine_transforms(nb::module_& mod) {
    ttnn::bind_function<"reduce_affine_transforms", "ttnn.experimental.kda.">(
        mod,
        R"doc(Reduce ordered grouped affine transforms into one transform per head.)doc",
        &ttnn::experimental::kda::reduce_affine_transforms,
        nb::arg("a").noconvert(),
        nb::arg("b").noconvert(),
        nb::arg("groups_per_head"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::reduce_affine_transforms::detail
