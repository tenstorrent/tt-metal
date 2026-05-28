// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_rms_norm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/normalization/all_gather_rms_norm/all_gather_rms_norm.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::normalization::detail {

void bind_all_gather_rms_norm(nb::module_& mod) {
    ttnn::bind_function<"all_gather_rms_norm">(
        mod,
        R"doc(
            Generic fused all-gather RMSNorm.

            Fuses, into a single multi-device op: per-device partial stats (E[x^2] over the local shard
            of the reduction dim) -> cross-device all-gather of the stats over ``cluster_axis`` ->
            post-normalization ``x / sqrt(E[x^2] + epsilon)`` with optional ``weight`` (gamma), optional
            ``bias`` (beta) and optional fused residual add.

            Unlike ``ttnn.fused_rms_minimal`` (the LLaMA-decode "minimal" variant), this op is generic:
            it supports arbitrary M (long sequences), TILE layout and INTERLEAVED memory. It does NOT
            fuse RoPE or a head-split; run those as separate ops afterward.
        )doc",
        &ttnn::all_gather_rms_norm,
        nb::arg("input_tensor"),
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::arg("global_semaphore"),
        nb::kw_only(),
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("epsilon") = 1e-12,
        nb::arg("residual_input_tensor") = nb::none(),
        nb::arg("topology") = ttnn::ccl::Topology::Linear,
        nb::arg("num_links") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("persistent_stats_tensor") = nb::none());
}

}  // namespace ttnn::operations::normalization::detail
