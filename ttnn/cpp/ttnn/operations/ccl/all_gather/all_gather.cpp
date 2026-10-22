// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather.hpp"

#include <string_view>

#include "device/all_gather_device_operation.hpp"
#include "device/all_gather_plan.hpp"
#include "device/all_gather_program.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

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

    // The persistent output tensor is the buffer that actually gets written, so it defines the
    // output config. Only buffer type is compared: full MemoryConfig equality would reject
    // nd-sharded configs, which compute_output_specs_helper() normalizes.
    if (memory_config.has_value() && persistent_output_tensor.has_value()) {
        TT_FATAL(
            memory_config->buffer_type() == persistent_output_tensor->memory_config().buffer_type(),
            "all_gather was given a memory_config ({}) in a different memory than the output_tensor's ({}). They "
            "must agree; omit memory_config to take it from the output tensor.",
            *memory_config,
            persistent_output_tensor->memory_config());
    }
    const std::optional<ttnn::MemoryConfig> output_memory_config =
        persistent_output_tensor.has_value()
            ? std::optional<ttnn::MemoryConfig>(persistent_output_tensor->memory_config())
            : memory_config;

    auto composite = [&](std::string_view reason) {
        log_info(tt::LogOp, "Using slower composite all_gather: {}", reason);
        return composite_common::composite_all_gather(
            input_tensor,
            dim,
            std::nullopt,
            std::nullopt,
            output_memory_config,
            subdevice_id,
            cluster_axis,
            false,
            persistent_output_tensor,
            sub_core_grid);
    };

    // Before the params: building them builds the output spec, which can TT_FATAL on a legacy output config that
    // cannot hold a gather-dim-padded shape.
    const auto& logical_shape = input_tensor.logical_shape();
    const int32_t dim_from_end =
        static_cast<int32_t>(logical_shape.get_normalized_index(dim)) - static_cast<int32_t>(logical_shape.rank());
    const auto axis_devices = operations::ccl::all_gather_axis_devices(input_tensor, cluster_axis);
    const uint32_t num_devices = axis_devices[0] * axis_devices[1];
    if (!operations::ccl::all_gather_plan::gather_shape_ok(input_tensor.tensor_spec(), dim_from_end, num_devices)) {
        return composite("the gather dim is padded or its gathered extent overflows uint32");
    }

    // The same plan select_program_factory makes from the same params, so the two cannot disagree.
    auto [params, inputs] = operations::ccl::all_gather_build_operation_args(
        input_tensor, persistent_output_tensor, dim, output_memory_config, cluster_axis, subdevice_id, sub_core_grid);
    // User errors (e.g. a wrong-sized output_tensor) must fail the same way whichever path runs, so validate before
    // routing: an illegal output spec would otherwise send the op to composite and fail there, after the gather.
    operations::ccl::AllGatherDeviceOperation::validate_on_program_cache_miss(params, inputs);
    const auto& output_spec =
        persistent_output_tensor.has_value() ? persistent_output_tensor->tensor_spec() : params.output_spec;
    const auto plan = operations::ccl::all_gather_program::make_device_plan(params, input_tensor, output_spec);
    if (plan.algo == operations::ccl::all_gather_plan::AgAlgo::Composite) {
        return composite(plan.composite_reason);
    }

    return ttnn::device_operation::launch<operations::ccl::AllGatherDeviceOperation>(params, inputs);
}

}  // namespace ttnn
