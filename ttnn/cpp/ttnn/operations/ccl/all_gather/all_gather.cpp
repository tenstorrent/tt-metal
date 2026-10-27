// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather.hpp"

#include "device/all_gather_device_operation.hpp"
#include "device/all_gather_plan.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

#include <string>

namespace ttnn {

ttnn::Tensor all_gather(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid,
    // The following args are deprecated and will be removed in a future update
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    bool use_l1_small_for_semaphores) {
    // Validate the gather dim before anything indexes the shape with it
    const int32_t input_rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    TT_FATAL(dim >= -input_rank && dim < input_rank, "Invalid gather dim {} for {}D input tensor", dim, input_rank);

    // Throw deprecation notice
    if (num_links.has_value() || topology.has_value() || chunks_per_sync.has_value() ||
        num_workers_per_link.has_value() || num_buffers_per_channel.has_value() || use_l1_small_for_semaphores) {
        log_warning(
            tt::LogOp,
            "The following ttnn.all_gather args are deprecated and will be removed in September-2026: num_links, "
            "topology, chunks_per_sync, num_workers_per_link, num_buffers_per_channel, use_l1_small_for_semaphores.");
    }

    namespace all_gather_plan = operations::ccl::all_gather_plan;
    auto run_composite = [&](const std::string& reason) {
        log_info(tt::LogOp, "Using slower composite all_gather: {}", reason);
        return composite_common::composite_all_gather(
            input_tensor,
            dim,
            std::nullopt,
            std::nullopt,
            memory_config,
            subdevice_id,
            cluster_axis,
            /*use_l1_small_for_semaphores=*/false,
            persistent_output_tensor,
            sub_core_grid);
    };

    // Checked before anything builds the output spec: that can TT_FATAL on a legacy output config that cannot hold
    // a gathered shape padded on the gather dim, and such an input goes to composite instead.
    const auto& logical_shape = input_tensor.logical_shape();
    const int32_t dim_from_end =
        static_cast<int32_t>(logical_shape.get_normalized_index(dim)) - static_cast<int32_t>(logical_shape.rank());
    if (auto reason = all_gather_plan::gather_dim_padding_reason(input_tensor.tensor_spec(), dim_from_end)) {
        return run_composite(*reason);
    }

    // Facts -> plan -> choice, once per call; the factories re-expand the hashed choice on a program-cache miss.
    auto [params, inputs] = operations::ccl::all_gather_build_operation_args(
        input_tensor, persistent_output_tensor, dim, memory_config, cluster_axis, subdevice_id, sub_core_grid);
    const auto plan = operations::ccl::plan_all_gather(params, inputs);
    if (plan.choice.algo == all_gather_plan::Algo::Composite) {
        return run_composite(plan.composite_reason);
    }
    return ttnn::device_operation::launch<operations::ccl::AllGatherDeviceOperation>(params, inputs);
}

}  // namespace ttnn
