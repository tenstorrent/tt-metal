// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/kda/reduce_affine_transforms/reduce_affine_transforms_nanobind.hpp"

namespace ttnn::operations::experimental::kda::detail {

void bind_kda(nb::module_& mod) {
    auto kda_module = mod.def_submodule("kda", "Experimental KDA operations");
    reduce_affine_transforms::detail::bind_reduce_affine_transforms(kda_module);
}

}  // namespace ttnn::operations::experimental::kda::detail
