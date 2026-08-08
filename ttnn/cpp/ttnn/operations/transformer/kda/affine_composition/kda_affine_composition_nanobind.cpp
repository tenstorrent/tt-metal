// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_affine_composition_nanobind.hpp"

#include "kda_affine_composition.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

namespace ttnn::operations::transformer {

void bind_kda_affine_composition(nb::module_& mod) {
    ttnn::bind_function<"kda_affine_compose", "ttnn.transformer.">(
        mod,
        R"doc(Compose grouped KDA affine summaries into one transform per head.)doc",
        &ttnn::transformer::kda_affine_compose,
        nb::arg("transform_a").noconvert(),
        nb::arg("transform_b").noconvert(),
        nb::arg("groups_per_head"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
