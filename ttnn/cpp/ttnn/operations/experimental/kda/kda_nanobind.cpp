// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/kda/prepare_chunk_recurrence/prepare_chunk_recurrence_nanobind.hpp"
#include "ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/sigmoid_gated_rms_norm_nanobind.hpp"

namespace ttnn::operations::experimental::kda::detail {

void bind_kda(nb::module_& mod) {
    auto kda_module = mod.def_submodule("kda", "Experimental KDA operations");
    prepare_chunk_recurrence::detail::bind_prepare_chunk_recurrence(kda_module);
    sigmoid_gated_rms_norm::detail::bind_sigmoid_gated_rms_norm(kda_module);
}

}  // namespace ttnn::operations::experimental::kda::detail
