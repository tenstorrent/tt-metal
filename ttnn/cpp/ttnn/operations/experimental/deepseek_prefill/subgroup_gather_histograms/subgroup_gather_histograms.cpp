// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subgroup_gather_histograms.hpp"
#include "device/subgroup_gather_histograms_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
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

    auto topology_ = topology.value_or(tt::tt_fabric::Topology::Linear);
    TT_FATAL(
        topology_ == tt::tt_fabric::Topology::Linear || topology_ == tt::tt_fabric::Topology::Ring,
        "topology must be Linear or Ring");

    auto* mesh_device = input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto worker_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    std::optional<uint32_t> axis_opt = cluster_axis;
    uint32_t num_links_ = num_links == 0 ? ccl::common::get_num_links(*mesh_device, axis_opt) : num_links;
    auto usable_topology = ::ttnn::ccl::get_usable_topology(input_tensor, topology_, axis_opt);

    return ttnn::prim::subgroup_gather_histograms(
        input_tensor,
        cluster_axis,
        num_dispatch_subgroups,
        num_links_,
        usable_topology,
        memory_config,
        worker_core_range_set);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms
