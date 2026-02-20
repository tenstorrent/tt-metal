// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_metadata.hpp"
#include "device/all_to_all_dispatch_metadata_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/work_split.hpp>
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
    const std::optional<CoreCoord>& drain_sync_tilizer_core,
    WorkerMode worker_mode,
    DispatchAlgorithm dispatch_algorithm,
    const std::optional<CoreRangeSet>& worker_core_range_set,
    const std::optional<CoreRangeSet>& mux_core_range_set,
    const std::optional<GlobalSemaphore>& cross_device_semaphore) {
    auto* mesh_device = input_tensor.device();

    uint32_t num_links_ = num_links.value_or(ttnn::operations::ccl::common::get_num_links(*mesh_device, axis));
    log_debug(tt::LogOp, "num_links: {}", num_links_);

    // Always derive topology from mesh - only RING topology is functional
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, std::nullopt, axis);

    // Resolve drain_sync_tilizer_core:
    // - If explicitly provided, use it
    // - If persistent output tensors are provided, extract from their shard spec (must be single core)
    // - Otherwise, error - one of the above must be provided
    std::optional<CoreCoord> resolved_drain_sync_tilizer_core = drain_sync_tilizer_core;
    if (!resolved_drain_sync_tilizer_core.has_value() && optional_output_tensors.has_value()) {
        // Extract drain core from the metadata tensor's shard spec (indices tensor is at index 1)
        const auto& indices_out_tensor = optional_output_tensors.value()[1];
        const auto& shard_spec = indices_out_tensor.memory_config().shard_spec();
        TT_FATAL(shard_spec.has_value(), "Persistent metadata tensor must have a shard spec");
        auto cores = tt::tt_metal::corerange_to_cores(shard_spec->grid);
        TT_FATAL(
            cores.size() == 1, "Persistent metadata tensor must be sharded on exactly 1 core, got {}", cores.size());
        resolved_drain_sync_tilizer_core = cores[0];
        log_debug(
            tt::LogOp,
            "Extracted drain_sync_tilizer_core from persistent tensor: ({}, {})",
            resolved_drain_sync_tilizer_core->x,
            resolved_drain_sync_tilizer_core->y);
    }
    TT_FATAL(
        resolved_drain_sync_tilizer_core.has_value(),
        "drain_sync_tilizer_core must be provided explicitly OR persistent output tensors must be provided "
        "(so drain core can be extracted from their shard spec)");

    // Default worker cores: (0,0) to (0,7) - 8 cores for 4 links (2 workers per link)
    CoreRangeSet worker_cores =
        worker_core_range_set.value_or(CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 7))));

    // Default mux cores: (1,0) to (1,7) - 8 cores (2 per link × 4 links)
    CoreRangeSet mux_cores = mux_core_range_set.value_or(CoreRangeSet(CoreRange(CoreCoord(1, 0), CoreCoord(1, 7))));

    log_debug(tt::LogOp, "worker_mode: {}", static_cast<int>(worker_mode));
    log_debug(tt::LogOp, "dispatch_algorithm: {}", static_cast<int>(dispatch_algorithm));

    return ttnn::prim::all_to_all_dispatch_metadata(
        input_tensor,
        expert_indices_tensor,
        expert_scores_tensor,
        expert_mapping_tensor,
        axis,
        optional_output_tensors,
        num_links_,
        topology_,
        worker_cores,
        resolved_drain_sync_tilizer_core,
        worker_mode,
        mux_cores,
        dispatch_algorithm,
        cross_device_semaphore);
}

}  // namespace ttnn::operations::experimental::ccl
