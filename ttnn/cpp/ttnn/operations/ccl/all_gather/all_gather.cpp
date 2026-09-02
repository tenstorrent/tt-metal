// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather.hpp"

#include "device/all_gather_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>

#include <algorithm>

namespace ttnn {

// Native implementation only handles cases where every output write is an aligned NoC
// write into the output buffer, computed with index math alone -- the bytes are never
// rearranged on-device first. If it needs a transpose, untilize, re-pad, or
// re-shard ("massaged op"), it goes to composite.
std::pair<bool, std::string> use_composite_all_gather(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const std::optional<ttnn::MemoryConfig>& memory_config,
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
    const auto output_spec =
        operations::ccl::compute_output_specs_helper(input_tensor, gather_dim_from_end, num_devices, memory_config);

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

    // Page sizes are UNALIGNED (content) sizes, so a matched-but-padded gather reads as matched, not
    // concat. The page checks below are no-ops for tile, whose pages are always aligned and equally
    // sized on both sides.
    const uint32_t input_page_size = input_tensor.buffer()->page_size();
    const uint32_t input_aligned_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t output_page_size = output_spec.compute_page_size_bytes();

    // The factory derives chunks-per-page (concat) and split-factor (split) with a truncating
    // division, so a non-integer page ratio would silently mis-place every write.
    if (std::max(input_page_size, output_page_size) % std::min(input_page_size, output_page_size) != 0) {
        return {
            true,
            fmt::format(
                "input and output rows are {} B and {} B; one shard width must be a whole multiple of the other",
                input_page_size,
                output_page_size)};
    }
    // concat and split move data at content granularity, so the input page must have no padding.
    if (input_page_size != output_page_size && input_page_size != input_aligned_page_size) {
        return {
            true,
            fmt::format(
                "input rows ({} B) are padded to the {} B memory alignment; resharding needs unpadded rows",
                input_page_size,
                input_tensor.buffer()->alignment())};
    }
    // NoC write alignment (NOC_{L1,DRAM}_WRITE_ALIGNMENT_BYTES): 16 B on Wormhole/Blackhole, 1 B on Quasar.
    // Ideally this should be queried (Hal::get_write_alignment(HalMemType) is currently unreachable from TTNN).
    constexpr uint32_t noc_write_alignment = 16;
    // Split moves output-row sized chunks, written to the output and read at those offsets inside an input
    // page, so the row must suit the write alignment and the input's own (DRAM reads need 32/64 B).
    const uint32_t split_chunk_alignment = std::max(noc_write_alignment, input_tensor.buffer()->alignment());
    if (input_page_size > output_page_size && output_page_size % split_chunk_alignment != 0) {
        return {
            true,
            fmt::format(
                "output is sharded finer than the input; its rows ({} B) must be a multiple of {} B",
                output_page_size,
                split_chunk_alignment)};
    }
    // matched and concat write a whole *aligned* input page into each output page slot, so the slot
    // must be at least that large. They differ only across memories with unequal alignments.
    const uint32_t output_alignment = output_spec.memory_config().is_dram() ? tt::tt_metal::hal::get_dram_alignment()
                                                                            : tt::tt_metal::hal::get_l1_alignment();
    const uint32_t output_aligned_page_size = tt::align(output_page_size, output_alignment);
    if (input_page_size <= output_page_size && output_aligned_page_size < input_aligned_page_size) {
        return {
            true,
            fmt::format(
                "input rows pad to {} B but the output reserves {} B; use the same buffer type, or a row size "
                "that is a multiple of both DRAM and L1 alignments",
                input_aligned_page_size,
                output_aligned_page_size)};
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

    auto [use_composite, composite_reason] =
        use_composite_all_gather(input_tensor, dim, output_memory_config, cluster_axis);
    if (use_composite) {
        log_info(tt::LogOp, "Using slower composite all_gather: {}", composite_reason);
        // NOTE: persistent_output_tensor and sub_core_grid have no equivalent in the composite
        // path and are ignored here for now.
        return composite_common::composite_all_gather(
            input_tensor, dim, std::nullopt, std::nullopt, output_memory_config, subdevice_id, cluster_axis);
    }

    return ttnn::prim::all_gather(
        input_tensor, persistent_output_tensor, dim, output_memory_config, cluster_axis, subdevice_id, sub_core_grid);
}

}  // namespace ttnn
