// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather.hpp"

#include "device/all_gather_device_operation.hpp"
#include "device/all_gather_geometry.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn {

namespace {

// The composite path (all_broadcast + concat), writing into `persistent_output_tensor` when one is given.
//
// The composite ops have no output-tensor parameter, so a persistent output is filled by one extra device copy
// of the gathered result. Cost, per device: an extra read + write of the whole gathered output (N x the input),
// one more op launch, and the composite's own N x input result buffer allocated next to the persistent output
// until this function returns. The copy runs as its own op on the default worker cores, not on sub_core_grid.
ttnn::Tensor all_gather_composite(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<ttnn::MemoryConfig>& output_memory_config,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid,
    const std::string& reason) {
    log_info(tt::LogOp, "Using slower composite all_gather: {}", reason);
    if (sub_core_grid.has_value()) {
        log_warning(
            tt::LogOp,
            "all_gather composite path does not take sub_core_grid; its all_broadcast / concat{} may use any worker "
            "core of the sub-device.",
            persistent_output_tensor.has_value() ? " / copy" : "");
    }
    ttnn::Tensor result = composite_common::composite_all_gather(
        input_tensor, dim, std::nullopt, std::nullopt, output_memory_config, subdevice_id, cluster_axis);
    if (!persistent_output_tensor.has_value()) {
        return result;
    }

    // The checks the native op's validate would have made, against what composite actually produced; ttnn::copy
    // would otherwise convert dtype (tile) or reject the copy with a less specific message.
    const auto& dst = persistent_output_tensor.value();
    TT_FATAL(dst.storage_type() == StorageType::DEVICE, "all_gather output_tensor must be on device");
    TT_FATAL(
        dst.dtype() == result.dtype() && dst.layout() == result.layout(),
        "all_gather output_tensor is {} / {} but the gathered result is {} / {}",
        dst.dtype(),
        dst.layout(),
        result.dtype(),
        result.layout());
    TT_FATAL(
        dst.logical_shape() == result.logical_shape() && dst.padded_shape() == result.padded_shape(),
        "all_gather output_tensor shape must be {} (padded {}), got {} (padded {})",
        result.logical_shape(),
        result.padded_shape(),
        dst.logical_shape(),
        dst.padded_shape());
    TT_FATAL(
        dst.memory_config().buffer_type() == result.memory_config().buffer_type(),
        "all_gather output_tensor is in {} but the composite result is in {}",
        dst.memory_config(),
        result.memory_config());
    // Writes into dst's buffer (prim::copy with dst as its preallocated output) and returns dst.
    return ttnn::copy(result, dst);
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
    auto composite = [&](const std::string& reason) {
        return all_gather_composite(
            input_tensor,
            dim,
            cluster_axis,
            output_memory_config,
            persistent_output_tensor,
            subdevice_id,
            sub_core_grid,
            reason);
    };

    // Keep the gather-dim padding check first: building the params builds the output spec, which can
    // TT_FATAL on a legacy output config that can't hold the gathered shape, and a gather-dim-padded input
    // should route to composite instead.
    const auto& logical_shape = input_tensor.logical_shape();
    const int32_t gather_dim_from_end =
        static_cast<int32_t>(logical_shape.get_normalized_index(dim)) - static_cast<int32_t>(logical_shape.rank());
    if (auto reason = operations::ccl::gather_dim_padding_reason(input_tensor, gather_dim_from_end)) {
        return composite(*reason);
    }

    // Build the op's params once, route with them, and launch with the same params, so the route taken here and
    // the factory select_program_factory picks come from one route_all_gather() decision.
    auto [params, inputs] = operations::ccl::all_gather_build_operation_args(
        input_tensor, persistent_output_tensor, dim, output_memory_config, cluster_axis, subdevice_id, sub_core_grid);
    const tt::tt_metal::TensorSpec& out_spec =
        persistent_output_tensor.has_value() ? persistent_output_tensor->tensor_spec() : params.output_spec;
    const auto decision = operations::ccl::route_all_gather(params, input_tensor, out_spec);
    if (decision.route == operations::ccl::GatherRoute::Composite) {
        return composite(decision.reason);
    }

    // Same as ttnn::prim::all_gather, minus rebuilding the params.
    return ttnn::device_operation::launch<operations::ccl::AllGatherDeviceOperation>(params, inputs);
}

}  // namespace ttnn
