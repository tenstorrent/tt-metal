// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subgroup_gather_histograms_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "subgroup_gather_histograms.hpp"

#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms::detail {

void bind_experimental_subgroup_gather_histograms_operation(nb::module_& mod) {
    ttnn::bind_function<"subgroup_gather_histograms", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Subgroup-scoped all-gather of per-chip expert histograms.

            Partitions the mesh along `cluster_axis` into `num_dispatch_subgroups` equal
            contiguous subgroups and, within each subgroup, performs an all-gather of the
            per-chip `[1, n_routed_experts]` UINT32 histogram into a
            `[dispatch_group_size, n_routed_experts]` tensor on every chip. Fabric traffic
            never crosses subgroup boundaries.

            Args:
                input_tensor: 1D UINT32 tensor of expert counts `[n_routed_experts]`
                    (or `[1, n_routed_experts]`), ROW_MAJOR, interleaved DRAM.
                cluster_axis: mesh axis to partition (currently only 0 is supported).
                num_dispatch_subgroups: number of equal-sized subgroups to split into.
                num_links: number of ethernet links for fabric (0 = auto).
                memory_config: output memory config.
                subdevice_id: optional subdevice for worker cores.
                topology: fabric topology (Linear or Ring).
        )doc",
        &subgroup_gather_histograms,
        nb::arg("input_tensor").noconvert(),
        nb::arg("cluster_axis"),
        nb::arg("num_dispatch_subgroups"),
        nb::arg("num_links"),
        nb::arg("memory_config"),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("topology") = nb::cast(tt::tt_fabric::Topology::Linear));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms::detail
