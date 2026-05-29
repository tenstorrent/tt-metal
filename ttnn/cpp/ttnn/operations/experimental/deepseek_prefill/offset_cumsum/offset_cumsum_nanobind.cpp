// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "offset_cumsum_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "offset_cumsum.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum::detail {
void bind_experimental_offset_cumsum_operation(nb::module_& mod) {
    ttnn::bind_function<"offset_cumsum", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Computes global dispatch offsets from per-device expert histograms gathered across a cluster axis.

            Takes a 1D UINT32 tensor of shape [n_routed_experts] (per-device expert histogram),
            performs all_gather along the specified cluster_axis to produce a 2D tensor
            [num_devices, n_routed_experts], then computes:
              1. Local offsets: shifted prefix sum across devices (row k = sum of rows 0..k-1)
              2. Expert region offsets: exclusive prefix sum of total counts within each
                 chip's expert group (experts_per_chip stride)
              3. Global offsets: local offsets + expert region offsets

            Returns a tuple of three UINT32 tensors:
              - global_dispatch_offsets: shape [1, n_routed_experts] combining local and
                expert region offsets for this device.
              - total_counts_per_expert: shape [1, n_routed_experts] containing the total count
                across all devices for each expert (i.e. the sum of all input rows).
              - expert_region_offsets: shape [1, n_routed_experts] containing only the expert
                region component (shared across all source devices in a dispatch group):
                exclusive prefix sum of tile-aligned total counts within each chip's expert
                group.

            Args:
                * :attr:`input_tensor`: 1D UINT32 tensor of expert counts [n_routed_experts].
                * :attr:`cluster_axis`: Axis along which to all_gather across devices.
                * :attr:`num_links`: Number of links for all_gather.
                * :attr:`experts_per_chip`: Number of experts per chip (for expert region grouping).
                * :attr:`memory_config`: Memory configuration for intermediate and output tensors.

        )doc",
        &offset_cumsum,
        nb::arg("input_tensor").noconvert(),
        nb::arg("cluster_axis"),
        nb::arg("num_links"),
        nb::arg("experts_per_chip"),
        nb::arg("memory_config"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum::detail
