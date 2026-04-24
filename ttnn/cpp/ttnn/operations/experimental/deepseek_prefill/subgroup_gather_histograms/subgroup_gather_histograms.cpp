// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subgroup_gather_histograms.hpp"
#include "device/subgroup_gather_histograms_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms {

ttnn::Tensor subgroup_gather_histograms(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    uint32_t num_dispatch_subgroups,
    uint32_t num_links,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    std::optional<tt::tt_fabric::Topology> topology) {
    TT_FATAL(cluster_axis == 0, "Only cluster_axis=0 is supported (got {})", cluster_axis);
    TT_FATAL(num_dispatch_subgroups >= 1, "num_dispatch_subgroups must be >= 1 (got {})", num_dispatch_subgroups);

    if (topology.has_value()) {
        TT_FATAL(
            topology.value() == tt::tt_fabric::Topology::Linear || topology.value() == tt::tt_fabric::Topology::Ring ||
                topology.value() == tt::tt_fabric::Topology::Mesh || topology.value() == tt::tt_fabric::Topology::Torus,
            "topology must be Linear, Ring, Mesh, or Torus");
    }

    auto* mesh_device = input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto worker_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    std::optional<uint32_t> axis_opt = cluster_axis;
    uint32_t num_links_ = num_links == 0 ? ccl::common::get_num_links(*mesh_device, axis_opt) : num_links;
    // Pass caller's topology through; if unset, get_usable_topology falls back to the actual fabric topology
    // (`tt::tt_fabric::get_fabric_topology()`), which picks Mesh on a 2D fabric automatically.
    auto usable_topology = ::ttnn::ccl::get_usable_topology(input_tensor, topology, axis_opt);

    // The prim itself is strictly axis-0. For a 2D mesh, replicate along axis 1 first via a standard
    // all_gather (mirrors the pattern in ttnn::all_gather, which recurses once per mesh axis). This
    // gives every chip in a row the same N = mesh_cols histograms; the axis-0 prim then gathers
    // subgroup_rows of those N-row blocks (redundantly from both columns — correctness-safe — but each
    // sender only fans out to same-column peers, so the redundancy is absorbed locally and fabric
    // traffic never crosses subgroups).
    const auto mesh_shape = mesh_device->shape();
    const uint32_t mesh_cols = mesh_shape.dims() > 1 ? mesh_shape[1] : 1;

    ttnn::Tensor prim_input = input_tensor;
    if (mesh_cols > 1) {
        // Pre-pass along axis 1. `ttnn::all_gather` internally converts Mesh→Linear, so this works on
        // both 1D and 2D fabric configs.
        prim_input = ttnn::all_gather(
            input_tensor,
            /*dim=*/0,
            /*cluster_axis=*/1,
            /*subdevice_id=*/std::nullopt,
            /*memory_config=*/memory_config,
            /*optional_output_tensor=*/std::nullopt,
            /*num_links=*/num_links_);
    }

    return ttnn::prim::subgroup_gather_histograms(
        prim_input,
        cluster_axis,
        num_dispatch_subgroups,
        num_links_,
        usable_topology,
        memory_config,
        worker_core_range_set);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms
