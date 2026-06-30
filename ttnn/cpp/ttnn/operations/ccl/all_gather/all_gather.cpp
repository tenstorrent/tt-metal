// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather.hpp"

#include "device/all_gather_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn {

// Native implementation only handles cases where every output write is an aligned NoC
// write into the output buffer, computed with index math alone -- the bytes are never
// rearranged on-device first. If it needs a transpose, untilize, re-pad, or
// re-shard ("massaged op"), it goes to composite.
std::pair<bool, std::string> use_composite_all_gather(
    const ttnn::Tensor& input_tensor, int32_t dim, const std::optional<ttnn::MemoryConfig>& memory_config) {
    const int32_t rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    const int32_t gather_dim = (dim < 0) ? rank + dim : dim;
    const bool is_last_dim = (gather_dim == rank - 1);

    // Row-major, last-dim gather, unaligned pages.
    if (input_tensor.layout() == ttnn::Layout::ROW_MAJOR && is_last_dim &&
        input_tensor.buffer()->page_size() != input_tensor.buffer()->aligned_page_size()) {
        return {true, "last-dim gather on unaligned row-major pages"};
    }

    // Tiled, padded on the gather dim.
    if (input_tensor.layout() == ttnn::Layout::TILE &&
        input_tensor.logical_shape()[gather_dim] != input_tensor.padded_shape()[gather_dim]) {
        return {true, "input tensor has tile padding on the gather dim"};
    }

    // Output memory_config forces an unrelated re-shard: happens when output shard width isn't
    // a whole multiple/divisor of the input shard width (no integer page-size ratio).
    if (memory_config.has_value() && memory_config->is_sharded() && input_tensor.memory_config().is_sharded()) {
        const uint32_t input_shard_width = input_tensor.memory_config().shard_spec()->shape[1];
        const uint32_t output_shard_width = memory_config->shard_spec()->shape[1];
        if (input_shard_width % output_shard_width != 0 && output_shard_width % input_shard_width != 0) {
            return {
                true,
                fmt::format(
                    "input and output shard widths ({} and {}) are not integer multiples/divisors of each other",
                    input_shard_width,
                    output_shard_width)};
        }
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
    // Throw deprecation notice
    if (num_links.has_value() || topology.has_value() || chunks_per_sync.has_value() ||
        num_workers_per_link.has_value() || num_buffers_per_channel.has_value() || use_l1_small_for_semaphores) {
        log_warning(
            tt::LogOp,
            "The following ttnn.all_gather args are deprecated and will be removed in a future update: num_links, "
            "topology, chunks_per_sync, num_workers_per_link, num_buffers_per_channel, use_l1_small_for_semaphores.");
    }

    auto [use_composite, composite_reason] = use_composite_all_gather(input_tensor, dim, memory_config);
    if (use_composite) {
        log_info(tt::LogOp, "Using slower composite all_gather: {}", composite_reason);

        // Query the Fabric setup
        auto* mesh_device = input_tensor.device();
        TT_FATAL(mesh_device != nullptr, "Input tensor should be on device for all_gather operation");
        uint32_t num_links_ = ttnn::operations::ccl::common::get_num_links(*mesh_device, cluster_axis);

        // NOTE: persistent_output_tensor and sub_core_grid have no equivalent in the composite
        // path and are ignored here for now.
        return composite_common::composite_all_gather(
            input_tensor, dim, num_links_, memory_config, subdevice_id, cluster_axis);
    }

    return ttnn::prim::all_gather(
        input_tensor, persistent_output_tensor, dim, memory_config, cluster_axis, subdevice_id, sub_core_grid);
}

}  // namespace ttnn
