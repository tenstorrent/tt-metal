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

    // A row-major gather whose output page differs in size from the input page (wider = concat,
    // narrower = split) moves data with aligned NoC writes. That requires both pages to be un-padded;
    // padded ones go to composite. Tile pages are always aligned, so this never fires for tile.
    if (input_tensor.layout() == ttnn::Layout::ROW_MAJOR) {
        const uint32_t element_size = input_tensor.element_size();
        const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
        const uint32_t input_unaligned_page_size = input_tensor.buffer()->page_size();
        const bool input_padded = input_unaligned_page_size != input_page_size;

        // Output page size. Interleaved output = full row (only grows vs input -> concat, never
        // split); sharded output page = one shard width.
        const ttnn::MemoryConfig output_mem_config = memory_config.value_or(input_tensor.memory_config());
        bool concat = false;
        bool split = false;
        uint32_t output_unaligned_page_size = 0;
        if (output_mem_config.is_sharded()) {
            output_unaligned_page_size = output_mem_config.shard_spec()->shape[1] * element_size;
            concat = output_unaligned_page_size > input_unaligned_page_size;
            split = input_unaligned_page_size > output_unaligned_page_size;
        } else {
            concat = input_tensor.memory_config().is_sharded() || is_last_dim;
        }

        if ((concat || split) && input_padded) {
            return {
                true,
                fmt::format(
                    "row-major input page ({} B) is not a multiple of the {} B page alignment",
                    input_unaligned_page_size,
                    input_tensor.buffer()->alignment())};
        }
        // NoC write alignment (NOC_{L1,DRAM}_WRITE_ALIGNMENT_BYTES = 16 B on Wormhole/Blackhole).
        constexpr uint32_t noc_write_alignment = 16;
        if (split && output_unaligned_page_size % noc_write_alignment != 0) {
            return {
                true,
                fmt::format(
                    "row-major output page ({} B) is not a multiple of the {} B NoC write alignment",
                    output_unaligned_page_size,
                    noc_write_alignment)};
        }
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
    bool use_l1_small_for_semaphores,
    std::optional<uint32_t> batch_slice_idx,
    std::optional<uint32_t> valid_gather_extent) {
    // Throw deprecation notice
    if (num_links.has_value() || topology.has_value() || chunks_per_sync.has_value() ||
        num_workers_per_link.has_value() || num_buffers_per_channel.has_value() || use_l1_small_for_semaphores) {
        log_warning(
            tt::LogOp,
            "The following ttnn.all_gather args are deprecated and will be removed in September-2026: num_links, "
            "topology, chunks_per_sync, num_workers_per_link, num_buffers_per_channel, use_l1_small_for_semaphores.");
    }

    auto [use_composite, composite_reason] = use_composite_all_gather(input_tensor, dim, memory_config);
    if (use_composite) {
        TT_FATAL(
            !batch_slice_idx.has_value() && !valid_gather_extent.has_value(),
            "batch_slice_idx / valid_gather_extent require the native all_gather path; composite fallback: {}",
            composite_reason);
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
        input_tensor,
        persistent_output_tensor,
        dim,
        memory_config,
        cluster_axis,
        subdevice_id,
        sub_core_grid,
        batch_slice_idx,
        valid_gather_extent);
}

}  // namespace ttnn
