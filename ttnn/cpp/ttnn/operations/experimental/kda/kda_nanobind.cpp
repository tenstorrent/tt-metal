// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/kda/recurrent_chunk_scan/recurrent_chunk_scan_nanobind.hpp"
#include "ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/sigmoid_gated_rms_norm_nanobind.hpp"

namespace ttnn::operations::experimental::kda::detail {

void bind_kda(nb::module_& mod) {
    auto kda_module = mod.def_submodule("kda", "Experimental KDA operations");
    recurrent_chunk_scan::detail::bind_recurrent_chunk_scan(kda_module);
    sigmoid_gated_rms_norm::detail::bind_sigmoid_gated_rms_norm(kda_module);
}

}  // namespace ttnn::operations::experimental::kda::detail
