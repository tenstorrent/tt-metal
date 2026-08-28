// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/kda/affine_exclusive_scan/affine_exclusive_scan_nanobind.hpp"
#include "ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/qkv_causal_conv1d_silu_nanobind.hpp"
#include "ttnn/operations/experimental/kda/recurrent_chunk_scan/recurrent_chunk_scan_nanobind.hpp"
#include "ttnn/operations/experimental/kda/reduce_affine_transforms/reduce_affine_transforms_nanobind.hpp"
#include "ttnn/operations/experimental/kda/prepare_chunk_recurrence/prepare_chunk_recurrence_nanobind.hpp"
#include "ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/sigmoid_gated_rms_norm_nanobind.hpp"

namespace ttnn::operations::experimental::kda::detail {

void bind_kda(nb::module_& mod) {
    auto kda_module = mod.def_submodule("kda", "Experimental KDA operations");
    affine_exclusive_scan::detail::bind_affine_exclusive_scan(kda_module);
    qkv_causal_conv1d_silu::detail::bind_qkv_causal_conv1d_silu(kda_module);
    recurrent_chunk_scan::detail::bind_recurrent_chunk_scan(kda_module);
    reduce_affine_transforms::detail::bind_reduce_affine_transforms(kda_module);
    prepare_chunk_recurrence::detail::bind_prepare_chunk_recurrence(kda_module);
    sigmoid_gated_rms_norm::detail::bind_sigmoid_gated_rms_norm(kda_module);
}

}  // namespace ttnn::operations::experimental::kda::detail
