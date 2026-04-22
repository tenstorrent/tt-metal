// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms {

// Subgroup-scoped all-gather of per-chip expert histograms along `cluster_axis`.
//
// Each chip contributes a [1, n_routed_experts] UINT32 histogram. The output on every
// chip is a [dispatch_group_size, n_routed_experts] tensor holding only the histograms
// of the chips in this chip's dispatch subgroup, ordered by subgroup-local linearized
// coord. Fabric traffic never crosses subgroup boundaries.
//
// This is an internal helper for `offset_cumsum` when `num_dispatch_subgroups > 1`. A
// Python binding is provided so it can be unit-tested in isolation.
ttnn::Tensor subgroup_gather_histograms(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    uint32_t num_dispatch_subgroups,
    uint32_t num_links,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    std::optional<tt::tt_fabric::Topology> topology = tt::tt_fabric::Topology::Linear);

}  // namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms

namespace ttnn {
using operations::experimental::deepseek_prefill::subgroup_gather_histograms::subgroup_gather_histograms;
}  // namespace ttnn
