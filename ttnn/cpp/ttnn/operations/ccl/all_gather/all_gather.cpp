// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather.hpp"

#include "device/all_gather_device_operation.hpp"
#include "device/all_gather_plan.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

#include <cstdlib>
#include <string>

namespace ttnn {

namespace {

using operations::ccl::AgAlgo;
using operations::ccl::AgForce;
namespace all_gather_plan = operations::ccl::all_gather_plan;

// Test-only override of the plan (TTNN_ALL_GATHER_FORCE, grammar in all_gather_plan.hpp). The op's only environment
// read; the parsed value is hashed in the params, so a changed override is a different cached program.
AgForce read_force() {
    const char* spec = std::getenv(all_gather_plan::kForceEnvVar);
    if (spec == nullptr) {
        return {};
    }
    const AgForce force = all_gather_plan::parse_force(spec);
    log_warning(tt::LogOp, "{}='{}' overrides the all_gather plan (test-only)", all_gather_plan::kForceEnvVar, spec);
    return force;
}

}  // namespace

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

    const AgForce force = read_force();
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
        TT_FATAL(
            force.algo == AgAlgo::Auto || force.algo == AgAlgo::Composite,
            "all_gather: {} forces a native algorithm, but {}",
            all_gather_plan::kForceEnvVar,
            *reason);
        return run_composite(*reason);
    }

    // Facts -> plan -> choice, once per call; the factories re-expand the hashed choice on a program-cache miss.
    auto [params, inputs] = operations::ccl::all_gather_build_operation_args(
        input_tensor, persistent_output_tensor, dim, memory_config, cluster_axis, subdevice_id, sub_core_grid, force);
    const auto plan = operations::ccl::plan_all_gather(params, inputs);
    if (plan.choice.algo == AgAlgo::Composite) {
        return run_composite(plan.composite_reason);
    }
    return ttnn::device_operation::launch<operations::ccl::AllGatherDeviceOperation>(params, inputs);
}

}  // namespace ttnn
