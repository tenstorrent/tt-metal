// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_metadata.hpp"
#include "device/all_to_all_dispatch_metadata_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

namespace ttnn::operations::experimental::ccl {

std::array<ttnn::Tensor, 3> ExecuteAllToAllDispatchMetadata::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis,
    const std::optional<std::array<ttnn::Tensor, 3>>& optional_output_tensors,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<uint32_t>& output_concat_dim,
    const std::optional<CoreCoord>& drain_sync_tilizer_core,
    bool use_mux,
    const std::optional<CoreRangeSet>& worker_core_range_set,
    const std::optional<CoreRangeSet>& mux_core_range_set) {
    auto* mesh_device = input_tensor.device();

    uint32_t num_links_ = num_links.value_or(ttnn::operations::ccl::common::get_num_links(*mesh_device, axis));
    log_debug(tt::LogOp, "num_links: {}", num_links_);
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, axis);
    auto memory_config_ = memory_config.value_or(input_tensor.memory_config());
    uint32_t output_concat_dim_ = output_concat_dim.value_or(1);

    // Default drain_sync_tilizer_core to (0, 0) if not provided
    CoreCoord drain_core = drain_sync_tilizer_core.value_or(CoreCoord(0, 0));

    AllToAllDispatchMetadataDeviceOperation::AllToAllTransferType impl =
        AllToAllDispatchMetadataDeviceOperation::AllToAllTransferType::FullPacket;

    // Default worker cores: (0,0) to (0,7) - 8 cores for 4 links (2 workers per link)
    CoreRangeSet worker_cores =
        worker_core_range_set.value_or(CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 7))));

    // Default mux cores: (1,0) to (1,7) - 8 cores (2 per link × 4 links)
    CoreRangeSet mux_cores = mux_core_range_set.value_or(CoreRangeSet(CoreRange(CoreCoord(1, 0), CoreCoord(1, 7))));

    if (use_mux) {
        log_debug(tt::LogOp, "use_mux: true");
    } else {
        log_debug(tt::LogOp, "use_mux: false");
    }

    return ttnn::prim::all_to_all_dispatch_metadata(
        input_tensor,
        expert_indices_tensor,
        expert_scores_tensor,
        expert_mapping_tensor,
        axis,
        optional_output_tensors,
        num_links_,
        topology_,
        memory_config_,
        worker_cores,
        impl,
        output_concat_dim_,
        drain_core,
        use_mux,
        mux_cores);
}

}  // namespace ttnn::operations::experimental::ccl
