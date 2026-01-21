// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Gather planning utilities for converting [B, C, HW] -> [C, B, HW].
 */

#include "gather.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <algorithm>
#include <cstring>
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

namespace ttnn::experimental::prim {

namespace {

// Try to extend the last transfer if it is adjacent and compatible; otherwise append a new one
inline void append_or_extend_transfer(
    std::vector<GatherTransfer>& transfers,
    uint32_t input_core_idx,
    uint32_t output_core_idx,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    uint32_t input_offset_within_shard,
    uint32_t output_offset_within_shard,
    uint32_t channel,
    uint32_t batch) {
    if (!transfers.empty()) {
        auto& last = transfers.back();
        const bool compatible = last.src_core_idx == input_core_idx && last.dst_core_idx == output_core_idx &&
                                last.channel == channel && last.batch == batch &&
                                last.src_offset + last.length == input_offset_within_shard &&
                                last.dst_offset + last.length == output_offset_within_shard;
        if (compatible) {
            last.length += 1;
            return;
        }
    }

    transfers.emplace_back(
        input_core_idx,
        output_core_idx,
        input_cores[input_core_idx],
        output_cores[output_core_idx],
        input_offset_within_shard,
        output_offset_within_shard,
        1,
        channel,
        batch);
}

// Shared implementation for gather transfer generation.
// If output_shard_width_override is provided, it is used verbatim; otherwise ceil(B*HW/num_output_cores).
std::vector<GatherTransfer> generate_gather_transfers(
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    std::optional<uint32_t> output_shard_width_override) {
    std::vector<GatherTransfer> transfers;
    const uint32_t num_input_cores = input_cores.size();
    const uint32_t num_output_cores = output_cores.size();

    TT_FATAL(HW % num_input_cores == 0, "HW={} must be divisible by num_input_cores={}", HW, num_input_cores);

    const uint32_t input_shard_width = HW / num_input_cores;
    const uint32_t output_shard_width = output_shard_width_override.has_value()
                                            ? output_shard_width_override.value()
                                            : (B * HW + num_output_cores - 1) / num_output_cores;
    if (output_shard_width_override.has_value()) {
        TT_FATAL(output_shard_width != 0, "output_shard_width_override must be non-zero");
    }

    // Iterate in destination order for better coalescing: rows=c, cols=b*HW + hw
    for (uint32_t c = 0; c < C; c++) {
        for (uint32_t b = 0; b < B; b++) {
            for (uint32_t hw = 0; hw < HW; hw++) {
                const uint32_t input_col = hw;
                const uint32_t input_core_idx = input_col / input_shard_width;
                const uint32_t input_offset_within_shard = input_col % input_shard_width;

                const uint32_t output_col = (b * HW) + hw;
                const uint32_t output_core_idx = output_col / output_shard_width;
                const uint32_t output_offset_within_shard = output_col % output_shard_width;

                append_or_extend_transfer(
                    transfers,
                    input_core_idx,
                    output_core_idx,
                    input_cores,
                    output_cores,
                    input_offset_within_shard,
                    output_offset_within_shard,
                    c,
                    b);
            }
        }
    }

    // Sort by (src_core, src_row=b*C+c, src_offset)
    std::sort(transfers.begin(), transfers.end(), [C](const GatherTransfer& a, const GatherTransfer& b) {
        if (a.src_core_idx != b.src_core_idx) {
            return a.src_core_idx < b.src_core_idx;
        }
        const uint32_t a_row = (a.batch * C) + a.channel;
        const uint32_t b_row = (b.batch * C) + b.channel;
        if (a_row != b_row) {
            return a_row < b_row;
        }
        return a.src_offset < b.src_offset;
    });

    return transfers;
}

}  // namespace

/**
 * Precompute all transfers needed for the gather operation
 *
 * This function analyzes the data movement pattern and generates an explicit
 * list of all transfers required.

 * - Converts logical layout [B, C, HW] to [C, B, HW]
 * - Input is width-sharded across the HW dimension: [B*C, HW/num_input_cores]
 * - Output is height-sharded across the B*HW dimension: [C, B*HW/num_output_cores]
 */
std::vector<GatherTransfer> precompute_gather_transfers(
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores) {
    return generate_gather_transfers(B, C, HW, input_cores, output_cores, std::nullopt);
}

// Variant with explicit output shard width override
std::vector<GatherTransfer> precompute_gather_transfers(
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    uint32_t output_shard_width_override) {
    return generate_gather_transfers(
        B, C, HW, input_cores, output_cores, std::optional<uint32_t>(output_shard_width_override));
}

std::vector<LowLevelGatherTransfer> lower_gather_transfers(
    const std::vector<GatherTransfer>& transfers,
    uint32_t /*B*/,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    uint32_t /*num_output_cores*/,
    uint32_t element_size_bytes,
    uint32_t output_shard_width) {
    std::vector<LowLevelGatherTransfer> low_level_transfers;

    const uint32_t num_input_cores = input_cores.size();

    // Calculate shard dimensions
    const uint32_t input_shard_width = HW / num_input_cores;
    // Always require explicit output shard width for consistent offsets
    TT_FATAL(output_shard_width != 0, "Output shard width must be provided");

    for (const auto& t : transfers) {
        // Calculate absolute offset in source shard
        // Input shard layout: [B*C, HW/num_input_cores] in row-major order
        uint32_t src_row = (t.batch * C) + t.channel;
        uint32_t src_absolute_offset = (src_row * input_shard_width) + t.src_offset;

        // Calculate absolute offset in destination shard
        // Output shard layout: [C, B*HW/num_output_cores] in row-major order
        uint32_t dst_row = t.channel;
        uint32_t dst_absolute_offset = (dst_row * output_shard_width) + t.dst_offset;

        // Extract NOC coordinates from source core
        uint32_t src_noc_x = input_cores[t.src_core_idx].x;
        uint32_t src_noc_y = input_cores[t.src_core_idx].y;

        // Convert element offsets to byte offsets
        uint32_t src_offset_bytes = src_absolute_offset * element_size_bytes;
        uint32_t dst_offset_bytes = dst_absolute_offset * element_size_bytes;
        uint32_t transfer_size_bytes = t.length * element_size_bytes;

        low_level_transfers.emplace_back(
            t.src_core_idx,
            src_absolute_offset,
            t.dst_core_idx,
            dst_absolute_offset,
            t.length,
            src_noc_x,
            src_noc_y,
            src_offset_bytes,
            dst_offset_bytes,
            transfer_size_bytes);
    }

    return low_level_transfers;
}

BlockedTransfersWithCount group_transfers_by_output_column_blocks(
    const std::vector<GatherTransfer>& transfers,
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    uint32_t num_output_cores,
    uint32_t element_size_bytes,
    uint32_t block_size,
    uint32_t output_shard_width) {
    // Dictionary to group transfers by (dst_shard_idx, column_block_idx)
    std::map<std::pair<uint32_t, uint32_t>, std::vector<LowLevelGatherTransfer>> groups;

    // Lower transfers to get flat offsets
    auto low_level_transfers = lower_gather_transfers(
        transfers, B, C, HW, input_cores, num_output_cores, element_size_bytes, output_shard_width);

    // Group transfers by which column blocks they write to, splitting transfers that cross boundaries
    for (size_t i = 0; i < transfers.size(); i++) {
        const auto& transfer = transfers[i];
        const auto& low_level = low_level_transfers[i];

        const uint32_t start_col = transfer.dst_offset;
        const uint32_t end_col = start_col + transfer.length - 1;
        const uint32_t start_block = start_col / block_size;
        const uint32_t end_block = end_col / block_size;

        // Split transfer across blocks if it spans multiple blocks
        for (uint32_t block_idx = start_block; block_idx <= end_block; block_idx++) {
            // Calculate the portion of this transfer that belongs to this block
            uint32_t block_start_col = block_idx * block_size;
            uint32_t block_end_col = std::min(block_start_col + block_size, output_shard_width);

            // Calculate the overlap between the transfer and this block
            uint32_t overlap_start = std::max(start_col, block_start_col);
            uint32_t overlap_end = std::min(end_col + 1, block_end_col);

            if (overlap_start >= overlap_end) {
                continue;  // No overlap
            }

            uint32_t overlap_length = overlap_end - overlap_start;

            // Calculate source offset: how many elements into the original transfer
            uint32_t src_offset_in_transfer = overlap_start - start_col;

            // Extract channel (row) from the original absolute destination offset
            // Original: dst_absolute_offset = (channel * output_shard_width) + dst_offset
            // So: channel = dst_absolute_offset / output_shard_width
            uint32_t channel = low_level.dst_offset / output_shard_width;

            // Calculate column offset within the block
            uint32_t block_local_col = overlap_start - block_start_col;

            // Calculate destination offset within cb_in_batch buffer
            // cb_in_batch layout: [C x block_size] in row-major order
            // Offset = (channel * block_size + block_local_col) * element_size_bytes
            // This offset is relative to the block buffer start and includes both channel and column
            uint32_t dst_offset_in_block = (channel * block_size + block_local_col) * element_size_bytes;

            // Create a split transfer for this block
            // dst_offset_bytes is calculated relative to block buffer start (channel * block_size + column)
            // The kernel uses this offset directly without modulo
            LowLevelGatherTransfer split_transfer(
                low_level.src_shard_idx,
                low_level.src_offset + src_offset_in_transfer,  // Adjust source offset
                low_level.dst_shard_idx,
                block_local_col,  // Destination offset in elements (column offset within block, for test compatibility)
                overlap_length,   // Length of this split
                low_level.src_noc_x,
                low_level.src_noc_y,
                (low_level.src_offset + src_offset_in_transfer) * element_size_bytes,  // Source offset in bytes
                dst_offset_in_block,  // Destination offset in bytes (relative to block buffer start: channel *
                                      // block_size + column)
                overlap_length * element_size_bytes);  // Transfer size in bytes

            auto key = std::make_pair(transfer.dst_core_idx, block_idx);
            groups[key].push_back(split_transfer);
        }
    }

    // Convert to vector of BlockedTransferGroup
    std::vector<BlockedTransferGroup> blocked_groups;
    for (const auto& [key, transfers_list] : groups) {
        BlockedTransferGroup group(key.first, key.second, block_size);
        group.transfers = transfers_list;
        blocked_groups.push_back(std::move(group));
    }

    // Sort by destination shard and block index
    std::sort(
        blocked_groups.begin(), blocked_groups.end(), [](const BlockedTransferGroup& a, const BlockedTransferGroup& b) {
            if (a.dst_shard_idx != b.dst_shard_idx) {
                return a.dst_shard_idx < b.dst_shard_idx;
            }
            return a.dst_block_idx < b.dst_block_idx;
        });

    // Calculate number of logical blocks per core
    // Each core processes blocks 0 through (output_shard_width / block_size - 1)
    // Since all cores have the same padded output_shard_width, they all have the same number of blocks
    const uint32_t num_logical_blocks_per_core = tt::div_up(output_shard_width, block_size);

    // Count unique column block indices for validation/debugging
    std::set<uint32_t> unique_block_indices;
    for (const auto& group : blocked_groups) {
        unique_block_indices.insert(group.dst_block_idx);
    }

    // Validate that unique block indices match expected per-core count
    if (unique_block_indices.size() != num_logical_blocks_per_core) {
        TT_FATAL(
            false,
            "Mismatch: expected {} blocks per core, but found {} unique block indices across all cores. "
            "This may indicate uneven block distribution.",
            num_logical_blocks_per_core,
            unique_block_indices.size());
    }

    return {std::move(blocked_groups), num_logical_blocks_per_core};
}

std::vector<BlockedTransferGroup> coalesce_contiguous_transfers(
    const std::vector<BlockedTransferGroup>& blocked_groups) {
    std::vector<BlockedTransferGroup> optimized_groups;
    optimized_groups.reserve(blocked_groups.size());

    for (const auto& group : blocked_groups) {
        if (group.transfers.empty()) {
            optimized_groups.push_back(group);
            continue;
        }

        // Create a new group with the same metadata
        BlockedTransferGroup optimized_group(group.dst_shard_idx, group.dst_block_idx, group.block_size);

        // Sort transfers by (src_shard_idx, src_offset_bytes, dst_offset_bytes) for coalescing
        auto sorted_transfers = group.transfers;
        std::sort(
            sorted_transfers.begin(),
            sorted_transfers.end(),
            [](const LowLevelGatherTransfer& a, const LowLevelGatherTransfer& b) {
                if (a.src_shard_idx != b.src_shard_idx) {
                    return a.src_shard_idx < b.src_shard_idx;
                }
                if (a.src_offset_bytes != b.src_offset_bytes) {
                    return a.src_offset_bytes < b.src_offset_bytes;
                }
                return a.dst_offset_bytes < b.dst_offset_bytes;
            });

        // Coalesce contiguous transfers
        for (size_t i = 0; i < sorted_transfers.size();) {
            auto current_transfer = sorted_transfers[i];
            size_t j = i + 1;

            // Traditional contiguous coalescing only
            while (j < sorted_transfers.size()) {
                const auto& next_transfer = sorted_transfers[j];

                bool same_core = (current_transfer.src_shard_idx == next_transfer.src_shard_idx) &&
                                 (current_transfer.src_noc_x == next_transfer.src_noc_x) &&
                                 (current_transfer.src_noc_y == next_transfer.src_noc_y);
                bool src_contiguous =
                    (current_transfer.src_offset_bytes + current_transfer.transfer_size_bytes ==
                     next_transfer.src_offset_bytes);
                bool dst_contiguous =
                    (current_transfer.dst_offset_bytes + current_transfer.transfer_size_bytes ==
                     next_transfer.dst_offset_bytes);

                if (same_core && src_contiguous && dst_contiguous) {
                    // Traditional contiguous coalescing
                    current_transfer.transfer_size_bytes += next_transfer.transfer_size_bytes;
                    current_transfer.length += next_transfer.length;
                    j++;
                } else {
                    break;
                }
            }

            // Add the (possibly coalesced) transfer
            optimized_group.transfers.push_back(current_transfer);
            i = j;
        }

        // Log the optimization results
        if (optimized_group.transfers.size() != group.transfers.size()) {
            log_debug(
                tt::LogType::LogOp,
                "Coalesced block[{}:{}]: {} transfers -> {} transfers (saved {} NOC ops)",
                group.dst_shard_idx,
                group.dst_block_idx,
                group.transfers.size(),
                optimized_group.transfers.size(),
                group.transfers.size() - optimized_group.transfers.size());
        }

        optimized_groups.push_back(std::move(optimized_group));
    }

    return optimized_groups;
}

std::vector<GatherTransfer> precompute_gather_transfers(
    const Tensor& input, const std::vector<CoreCoord>& output_cores) {
    // Extract tensor properties
    const auto& input_shape = input.logical_shape();
    uint32_t B = input_shape[1];   // Batch size
    uint32_t C = input_shape[2];   // Channels
    uint32_t HW = input_shape[3];  // Spatial dimension

    // Get input core coordinates from tensor's shard spec
    TT_FATAL(input.is_sharded(), "Input tensor must be sharded for gather operation");

    const auto& shard_spec = input.shard_spec().value();
    const auto& core_grid = shard_spec.grid;
    std::vector<CoreCoord> input_cores = corerange_to_cores(
        core_grid, std::nullopt, shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    // Call the original implementation
    return precompute_gather_transfers(B, C, HW, input_cores, output_cores);
}

std::vector<BlockedTransferGroup> group_transfers_by_output_column_blocks(
    const Tensor& input, const Tensor& output, const std::vector<GatherTransfer>& transfers, uint32_t block_size) {
    // Extract tensor properties
    const auto& input_shape = input.logical_shape();
    uint32_t B = input_shape[1];
    uint32_t C = input_shape[2];
    uint32_t HW = input_shape[3];

    // Ensure both tensors are sharded
    TT_FATAL(input.is_sharded(), "Input tensor must be sharded for gather operation");
    TT_FATAL(output.is_sharded(), "Output tensor must be sharded for gather operation");

    // Get element size in bytes from tensor data type
    uint32_t element_size_bytes = input.element_size();

    // Get core coordinates from tensors
    const auto& input_shard_spec = input.shard_spec().value();
    const auto& input_core_grid = input_shard_spec.grid;
    auto input_cores_vec = corerange_to_cores(
        input_core_grid, std::nullopt, input_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    const auto& output_shard_spec = output.shard_spec().value();
    const auto& output_core_grid = output_shard_spec.grid;
    auto output_cores_vec = corerange_to_cores(
        output_core_grid, std::nullopt, output_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    uint32_t num_output_cores = output_cores_vec.size();
    // Compute per-core output row width (number of columns in each [C x (B*HW_per_core)] shard)
    TT_FATAL(output.is_sharded(), "Output tensor must be sharded for gather operation");
    const auto& output_shard_spec2 = output.shard_spec().value();
    uint32_t output_shard_width =
        static_cast<uint32_t>(output_shard_spec2.shape[0]);  // columns per output core = B*HW_per_core

    // Call the unified implementation and return only the blocked transfers
    auto result = group_transfers_by_output_column_blocks(
        transfers, B, C, HW, input_cores_vec, num_output_cores, element_size_bytes, block_size, output_shard_width);
    return result.blocked_transfers;
}

/**
 * @brief Split blocked transfer groups by destination core for per-core processing
 *
 * This function reorganizes BlockedTransferGroup objects by their destination core index,
 * creating a vector of vectors where each index corresponds to an output core and contains
 * all the BlockedTransferGroup objects that write to that core.
 */
std::vector<std::vector<BlockedTransferGroup>> split_by_destination_core(
    const std::vector<BlockedTransferGroup>& blocked_groups, uint32_t num_output_cores) {
    // Initialize result with empty vectors for each output core
    std::vector<std::vector<BlockedTransferGroup>> result(num_output_cores);

    // Distribute each BlockedTransferGroup to the appropriate core's vector
    for (const auto& group : blocked_groups) {
        uint32_t dst_core_idx = group.dst_shard_idx;

        // Validate core index bounds
        TT_FATAL(
            dst_core_idx < num_output_cores,
            "Destination core index {} is out of bounds (num_output_cores={})",
            dst_core_idx,
            num_output_cores);

        result[dst_core_idx].push_back(group);
    }

    return result;
}

}  // namespace ttnn::experimental::prim

// fmt formatter implementations
namespace fmt {

auto formatter<ttnn::experimental::prim::GatherTransfer>::format(
    const ttnn::experimental::prim::GatherTransfer& t, format_context& ctx) const -> format_context::iterator {
    std::string str = fmt::format(
        "GatherTransfer(B={}, C={}: Core{}[{},{}][row={}·C+{}, cols={}:{}] → Core{}[{},{}][row={}, cols={}:{}], "
        "len={})",
        t.batch,
        t.channel,
        t.src_core_idx,
        t.src_core_coord.x,
        t.src_core_coord.y,
        t.batch,
        t.channel,
        t.src_offset,
        t.src_offset + t.length,
        t.dst_core_idx,
        t.dst_core_coord.x,
        t.dst_core_coord.y,
        t.channel,
        t.dst_offset,
        t.dst_offset + t.length,
        t.length);
    return fmt::format_to(ctx.out(), "{}", str);
}

auto formatter<ttnn::experimental::prim::LowLevelGatherTransfer>::format(
    const ttnn::experimental::prim::LowLevelGatherTransfer& t, format_context& ctx) const -> format_context::iterator {
    std::string str = fmt::format(
        "LowLevelGatherTransfer(src_shard{}[{}:{}] (offset={} B) @ NOC({},{}) => dst_shard{}[{}:{}] (offset={} B), "
        "len={}, size={} B)",
        t.src_shard_idx,
        t.src_offset,
        t.src_offset + t.length,
        t.src_offset_bytes,
        t.src_noc_x,
        t.src_noc_y,
        t.dst_shard_idx,
        t.dst_offset,
        t.dst_offset + t.length,
        t.dst_offset_bytes,
        t.length,
        t.transfer_size_bytes);
    return fmt::format_to(ctx.out(), "{}", str);
}

auto formatter<ttnn::experimental::prim::BlockedTransferGroup>::format(
    const ttnn::experimental::prim::BlockedTransferGroup& t, format_context& ctx) const -> format_context::iterator {
    uint32_t col_start = t.dst_block_idx * t.block_size;
    uint32_t col_end = col_start + t.block_size;

    std::string str = fmt::format(
        "BlockedTransferGroup(shard={}, cols=[{}:{}], {} transfers)",
        t.dst_shard_idx,
        col_start,
        col_end,
        t.transfers.size());
    return fmt::format_to(ctx.out(), "{}", str);
}

}  // namespace fmt
