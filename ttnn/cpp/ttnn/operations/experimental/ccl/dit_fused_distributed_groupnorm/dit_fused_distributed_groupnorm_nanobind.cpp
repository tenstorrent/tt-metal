// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_fused_distributed_groupnorm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/dit_fused_distributed_groupnorm/dit_fused_distributed_groupnorm.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_dit_fused_distributed_groupnorm(nb::module_& mod) {
    ttnn::bind_function<"dit_fused_distributed_groupnorm", "ttnn.experimental.">(
        mod,
        R"doc(
            Fused distributed GroupNorm for spatially sharded activations.

            Same contract as ``ttnn.group_norm``, plus fabric all-gather of
            per-group stats on ``cluster_axis`` (PRE → AG → POST). When mesh
            width on that axis is 1, runs local PRE+POST with no fabric.
        )doc",
        &ttnn::experimental::dit_fused_distributed_groupnorm,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("num_groups"),
        nb::arg("epsilon") = 1e-5,
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::arg("multi_device_global_semaphore"),
        nb::arg("topology") = ttnn::ccl::Topology::Ring,
        nb::arg("input_mask") = nb::none(),
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("use_welford") = false,
        nb::arg("persistent_output_buffer") = nb::none(),
        nb::arg("num_preferred_links") = nb::none(),
        nb::arg("subdevice_id") = nb::none());

    ttnn::bind_function<"dit_fused_distributed_groupnorm_create_stats_buffer", "ttnn.experimental.">(
        mod,
        R"doc(
            Allocate the persistent DRAM stats scratch buffer for
            `dit_fused_distributed_groupnorm`'s all-gather path (cluster width > 1).

            Returns None when cluster width is 1 (no AG). The caller must hold the
            tensor across launches and pass it via `persistent_output_buffer`.
        )doc",
        &ttnn::experimental::dit_fused_distributed_groupnorm_create_stats_buffer,
        nb::arg("input_tensor"),
        nb::arg("num_groups"),
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("num_links") = 1);
}

}  // namespace ttnn::operations::experimental::ccl
