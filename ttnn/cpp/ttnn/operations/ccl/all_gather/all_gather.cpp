// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather.hpp"

#include "device/all_gather_device_operation.hpp"
#include "device/all_gather_factory_common.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"

namespace ttnn {

// Native implementation only handles cases where every output write is an aligned NoC
// write into the output buffer, computed with index math alone -- the bytes are never
// rearranged on-device first. If it needs a transpose, untilize, re-pad, or
// re-shard ("massaged op"), it goes to composite.
std::pair<bool, std::string> use_composite_all_gather(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    std::optional<uint32_t> cluster_axis) {
    const auto& logical_shape = input_tensor.logical_shape();
    const int32_t gather_dim = static_cast<int32_t>(logical_shape.get_normalized_index(dim));
    // Indexes both logical_shape and padded_shape, identical to AllGatherParams::dim_from_end.
    const int32_t gather_dim_from_end = gather_dim - static_cast<int32_t>(logical_shape.rank());

    // Below is equivalent to: axis_num_devices[0] * axis_num_devices[1]
    const auto& mesh_shape = input_tensor.device()->shape();
    const uint32_t num_devices = cluster_axis.has_value() ? mesh_shape[*cluster_axis] : mesh_shape[0] * mesh_shape[1];

    // Gather-dim padding would need the output's gather-dim extent to be num_devices * the input's
    // padded extent, but it is num_devices * the logical one.
    auto gathered_padded_shape = input_tensor.padded_shape();
    if (logical_shape[gather_dim_from_end] != gathered_padded_shape[gather_dim_from_end]) {
        return {
            true,
            fmt::format(
                "gather dim {} is padded from {} to {}; size must be a multiple of the tile/shard extent",
                gather_dim,
                logical_shape[gather_dim_from_end],
                gathered_padded_shape[gather_dim_from_end])};
    }

    // Keep the padding check above this: building the spec can TT_FATAL on a legacy output config that
    // can't hold the gathered shape, and a gather-dim-padded input should route to composite instead.
    // The factories build for the output tensor itself, and a persistent one may live in another memory.
    const auto output_spec = persistent_output_tensor.has_value()
                                 ? persistent_output_tensor->tensor_spec()
                                 : operations::ccl::compute_output_specs_helper(
                                       input_tensor, gather_dim_from_end, num_devices, memory_config);

    // The kernel walks the output page grid as the input's with only the gather dim scaled, so any
    // other difference breaks it -- e.g. an output shard width that doesn't divide the row, which
    // pads the last dim.
    gathered_padded_shape[gather_dim_from_end] *= num_devices;
    if (output_spec.padded_shape() != gathered_padded_shape) {
        return {
            true,
            fmt::format(
                "output stored as {} instead of {}; its shard shape must divide the gathered shape evenly",
                output_spec.padded_shape(),
                gathered_padded_shape)};
    }

    // The native factories cut pages on the NOC write alignment and the read alignment of every buffer they read;
    // unicast's cut (it may also read the output) exists only where multicast's does.
    if (!operations::ccl::compute_walk_geometry(
             input_tensor,
             output_spec,
             gather_dim_from_end,
             num_devices,
             tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
             /*reads_output=*/false)
             .has_value()) {
        return {
            true,
            fmt::format(
                "input and output rows are {} B and {} B; the native kernels cannot cut them on the NOC write and read "
                "alignments",
                input_tensor.mesh_buffer().page_size(),
                output_spec.compute_page_size_bytes())};
    }

    return {false, {}};
}

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

    auto [use_composite, composite_reason] =
        use_composite_all_gather(input_tensor, dim, memory_config, persistent_output_tensor, cluster_axis);
    if (use_composite) {
        log_info(tt::LogOp, "Using slower composite all_gather: {}", composite_reason);
        // NOTE: sub_core_grid has no equivalent in the composite path and is ignored here for now.
        // The composite path allocates its own result, so a persistent output receives a copy of it.
        const auto composite_memory_config = persistent_output_tensor.has_value()
                                                 ? std::optional(persistent_output_tensor->memory_config())
                                                 : memory_config;
        auto output = composite_common::composite_all_gather(
            input_tensor, dim, std::nullopt, std::nullopt, composite_memory_config, subdevice_id, cluster_axis);
        if (!persistent_output_tensor.has_value()) {
            return output;
        }
        auto persistent_output = ttnn::copy(output, *persistent_output_tensor);
        persistent_output.update_tensor_topology(output.tensor_topology());
        return persistent_output;
    }

    return ttnn::prim::all_gather(
        input_tensor, persistent_output_tensor, dim, memory_config, cluster_axis, subdevice_id, sub_core_grid);
}

}  // namespace ttnn
