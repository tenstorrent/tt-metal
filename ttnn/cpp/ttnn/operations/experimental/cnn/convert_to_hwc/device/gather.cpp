// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Gather Operation: Hardware-Optimized Tensor Reordering from [B, C, HW] to [C, B, HW]
 * ===================================================================================
 *
 * This implementation performs efficient data movement for tensor layout transformation,
 * specifically designed for neural network accelerators when transitioning between
 * different tensor layouts (e.g., NCHW to NHWC formats).
 *
 * Key Features:
 * 1. Transfer-based approach: Models distributed memory with explicit DMA transfers
 * 2. Hardware-oriented design: Maps directly to TT-Metal's distributed memory architecture
 * 3. Memory efficiency: Blocked implementation reduces on-chip memory requirements by 8x+
 * 4. Optimized data movement: Transfer coalescing and sorted accesses for bandwidth
 *
 * Data Sharding Model:
 * - Input sharding: Width-sharded across HW dimension [B*C, HW/num_input_cores]
 * - Output sharding: Height-sharded across B*HW dimension [C, B*HW/num_output_cores]
 * - Each core holds a contiguous slice of the tensor in its local memory
 *
 * Hardware Mapping:
 * - Each "core" represents a separate processing element with local SRAM
 * - Transfers model DMA operations between cores
 * - Blocking reduces on-chip SRAM requirements (typically 64KB-256KB per core)
 * - Sorted transfers improve DRAM access patterns and enable prefetching
 */

#include "gather.hpp"
#include <tt-metalium/host_api.hpp>
#include <algorithm>
#include <cstring>
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

namespace ttnn::operations::experimental::cnn::convert_to_hwc::detail {

/**
 * Precompute all transfers needed for the gather operation
 *
 * This function analyzes the data movement pattern and generates an explicit
 * list of all transfers required. This is critical for hardware implementation
 * where data movements must be scheduled and optimized.
 *
 * Key algorithm insights:
 * - For each (batch, channel) pair, we trace through spatial positions
 * - At each position, we determine source core/offset and dest core/offset
 * - We coalesce contiguous transfers to minimize DMA operations
 * - Transfers are sorted by source for optimal cache utilization
 *
 * The transfer generation handles edge cases:
 * - Input shard boundaries (when data spans multiple input cores)
 * - Output shard boundaries (when destination spans multiple output cores)
 * - Both boundaries can be hit in a single transfer (requiring a split)
 *
 * Hardware implications:
 * - Each Transfer maps to a DMA descriptor in hardware
 * - Coalescing reduces DMA overhead and improves bandwidth
 * - Sorting enables prefetching and reduces DRAM page misses
 */
std::vector<GatherTransfer> precompute_gather_transfers(
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores) {
    std::vector<GatherTransfer> transfers;

    uint32_t num_input_cores = input_cores.size();
    uint32_t num_output_cores = output_cores.size();

    // Input validation
    TT_FATAL(HW % num_input_cores == 0, "HW={} must be divisible by num_input_cores={}", HW, num_input_cores);
    TT_FATAL(
        (B * HW) % num_output_cores == 0, "B*HW={} must be divisible by num_output_cores={}", B * HW, num_output_cores);

    // Calculate sizes
    uint32_t input_shard_width = HW / num_input_cores;
    uint32_t output_shard_width = B * HW / num_output_cores;

    // Variables preserved for documentation and future use:
    // uint32_t input_shard_height = B * C;  // Height of each input shard
    // uint32_t output_shard_height = C;      // Height of each output shard

    // For each batch and channel, compute transfers
    // This loops through the logical tensor in output order to ensure
    // we can coalesce transfers that are contiguous in the output
    for (uint32_t c = 0; c < C; c++) {
        for (uint32_t b = 0; b < B; b++) {
            for (uint32_t hw = 0; hw < HW; hw++) {
                // Source location in original tensor [B, C, HW]
                // This element is at position (b, c, hw)

                // In input sharding: width-sharded [B * C, HW / num_input_cores]
                // Row in input: b * C + c
                // Column in input: hw
                uint32_t input_col = hw;
                uint32_t input_core_idx = input_col / input_shard_width;
                uint32_t input_offset_within_shard = input_col % input_shard_width;

                // Variables preserved for clarity:
                // uint32_t input_row = b * C + c;  // Row index in the input shard

                // In output sharding: width-sharded [C, BHW / num_output_cores]
                // Row in output: c
                // Column in output: b * HW + hw
                uint32_t output_col = b * HW + hw;
                uint32_t output_core_idx = output_col / output_shard_width;
                uint32_t output_offset_within_shard = output_col % output_shard_width;

                // Check if we need to add a new transfer or can extend an existing one
                if (!transfers.empty()) {
                    auto& last_transfer = transfers.back();
                    if (last_transfer.src_core_idx == input_core_idx && last_transfer.dst_core_idx == output_core_idx &&
                        last_transfer.channel == c && last_transfer.batch == b &&
                        last_transfer.src_offset + last_transfer.length == input_offset_within_shard &&
                        last_transfer.dst_offset + last_transfer.length == output_offset_within_shard) {
                        // Extend the previous transfer (coalescing)
                        last_transfer.length += 1;
                        continue;
                    }
                }

                // Create a new transfer
                transfers.emplace_back(
                    input_core_idx,
                    output_core_idx,
                    input_cores[input_core_idx],
                    output_cores[output_core_idx],
                    input_offset_within_shard,
                    output_offset_within_shard,
                    1,  // length
                    c,
                    b);
            }
        }
    }

    // Sort transfers by source core, then by source row, then by source offset
    // This improves cache locality when reading from input shards
    std::sort(transfers.begin(), transfers.end(), [C](const GatherTransfer& a, const GatherTransfer& b) {
        if (a.src_core_idx != b.src_core_idx) {
            return a.src_core_idx < b.src_core_idx;
        }
        uint32_t a_row = a.batch * C + a.channel;
        uint32_t b_row = b.batch * C + b.channel;
        if (a_row != b_row) {
            return a_row < b_row;
        }
        return a.src_offset < b.src_offset;
    });

    return transfers;
}

std::vector<LowLevelGatherTransfer> lower_gather_transfers(
    const std::vector<GatherTransfer>& transfers,
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    uint32_t num_output_cores,
    uint32_t element_size_bytes) {
    std::vector<LowLevelGatherTransfer> low_level_transfers;

    uint32_t num_input_cores = input_cores.size();

    // Calculate shard dimensions
    uint32_t input_shard_width = HW / num_input_cores;
    uint32_t output_shard_width = B * HW / num_output_cores;

    for (const auto& t : transfers) {
        // Calculate absolute offset in source shard
        // Input shard layout: [B*C, HW/num_input_cores] in row-major order
        uint32_t src_row = t.batch * C + t.channel;
        uint32_t src_absolute_offset = src_row * input_shard_width + t.src_offset;

        // Calculate absolute offset in destination shard
        // Output shard layout: [C, B*HW/num_output_cores] in row-major order
        uint32_t dst_row = t.channel;
        uint32_t dst_absolute_offset = dst_row * output_shard_width + t.dst_offset;

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

std::vector<BlockedTransferGroup> group_transfers_by_output_column_blocks(
    const std::vector<GatherTransfer>& transfers,
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    uint32_t num_output_cores,
    uint32_t element_size_bytes,
    uint32_t block_size) {
    // Dictionary to group transfers by (dst_shard_idx, column_block_idx)
    std::map<std::pair<uint32_t, uint32_t>, std::vector<LowLevelGatherTransfer>> groups;

    // Lower transfers to get flat offsets
    auto low_level_transfers =
        lower_gather_transfers(transfers, B, C, HW, input_cores, num_output_cores, element_size_bytes);

    // Group transfers by which column blocks they write to
    for (size_t i = 0; i < transfers.size(); i++) {
        const auto& transfer = transfers[i];
        const auto& low_level = low_level_transfers[i];

        // Determine which column blocks this transfer writes to
        // A transfer may span multiple column blocks
        uint32_t start_col = transfer.dst_offset;
        uint32_t end_col = transfer.dst_offset + transfer.length - 1;

        uint32_t start_block = start_col / block_size;
        uint32_t end_block = end_col / block_size;

        // Add this transfer to all column blocks it touches
        for (uint32_t block_idx = start_block; block_idx <= end_block; block_idx++) {
            auto key = std::make_pair(transfer.dst_core_idx, block_idx);
            groups[key].push_back(low_level);
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

    // Count unique column block indices to determine actual number of logical blocks
    std::set<uint32_t> unique_block_indices;
    for (const auto& group : blocked_groups) {
        unique_block_indices.insert(group.dst_block_idx);
    }

    // Log the actual number of logical blocks vs transfer groups
    log_info(
        tt::LogType::LogAlways,
        "group_transfers_by_output_column_blocks: {} transfer groups, {} logical blocks",
        blocked_groups.size(),
        unique_block_indices.size());

    return blocked_groups;
}

BlockedTransfersWithCount group_transfers_by_output_column_blocks_with_count(
    const std::vector<GatherTransfer>& transfers,
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    uint32_t num_output_cores,
    uint32_t element_size_bytes,
    uint32_t block_size) {
    // Get the blocked transfers using existing function
    auto blocked_transfers = group_transfers_by_output_column_blocks(
        transfers, B, C, HW, input_cores, num_output_cores, element_size_bytes, block_size);

    // Count unique column block indices to determine actual number of logical blocks
    std::set<uint32_t> unique_block_indices;
    for (const auto& group : blocked_transfers) {
        unique_block_indices.insert(group.dst_block_idx);
    }

    return {std::move(blocked_transfers), static_cast<uint32_t>(unique_block_indices.size())};
}

std::vector<BlockedTransferGroup> coalesce_contiguous_transfers(
    const std::vector<BlockedTransferGroup>& blocked_groups) {
    std::vector<BlockedTransferGroup> optimized_groups;
    optimized_groups.reserve(blocked_groups.size());

    // Count total transfers before optimization
    size_t total_transfers_before = 0;
    for (const auto& group : blocked_groups) {
        total_transfers_before += group.transfers.size();
    }
    log_info(
        tt::LogType::LogAlways,
        "Coalescing: {} transfer groups with {} total transfers",
        blocked_groups.size(),
        total_transfers_before);

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

        // Coalesce contiguous and stride-aware transfers
        for (size_t i = 0; i < sorted_transfers.size();) {
            auto current_transfer = sorted_transfers[i];
            size_t j = i + 1;
            uint32_t stride_pattern = 0;
            bool has_stride_pattern = false;
            std::vector<size_t> stride_group_indices;  // Track which transfers are part of stride group
            stride_group_indices.push_back(i);

            // Phase 1: Traditional contiguous coalescing
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

                if (!same_core || !src_contiguous) {
                    break;
                }

                if (dst_contiguous) {
                    // Traditional contiguous coalescing
                    current_transfer.transfer_size_bytes += next_transfer.transfer_size_bytes;
                    current_transfer.length += next_transfer.length;
                    j++;
                } else {
                    break;  // Switch to stride-aware coalescing
                }
            }

            // Phase 2: Stride-aware coalescing (if no contiguous coalescing happened)
            if (j == i + 1 && j < sorted_transfers.size()) {
                const auto& next_transfer = sorted_transfers[j];
                uint32_t dst_stride = next_transfer.dst_offset_bytes - current_transfer.dst_offset_bytes;
                uint32_t src_stride = next_transfer.src_offset_bytes - current_transfer.src_offset_bytes;

                // Check if this could be a stride pattern
                bool same_core = (current_transfer.src_shard_idx == next_transfer.src_shard_idx) &&
                                 (current_transfer.src_noc_x == next_transfer.src_noc_x) &&
                                 (current_transfer.src_noc_y == next_transfer.src_noc_y);
                bool valid_stride = (dst_stride > current_transfer.transfer_size_bytes) &&
                                    (dst_stride <= 8 * current_transfer.transfer_size_bytes) &&
                                    (src_stride == current_transfer.transfer_size_bytes);  // Source must be contiguous

                if (same_core && valid_stride &&
                    next_transfer.transfer_size_bytes == current_transfer.transfer_size_bytes) {
                    stride_pattern = dst_stride;
                    has_stride_pattern = true;
                    stride_group_indices.push_back(j);

                    // Look for more transfers that match this stride pattern
                    size_t k = j + 1;
                    while (k < sorted_transfers.size()) {
                        const auto& candidate = sorted_transfers[k];
                        const auto& prev = sorted_transfers[k - 1];

                        uint32_t expected_src = prev.src_offset_bytes + prev.transfer_size_bytes;
                        uint32_t expected_dst = prev.dst_offset_bytes + stride_pattern;

                        bool matches_stride = (candidate.src_shard_idx == current_transfer.src_shard_idx) &&
                                              (candidate.src_noc_x == current_transfer.src_noc_x) &&
                                              (candidate.src_noc_y == current_transfer.src_noc_y) &&
                                              (candidate.src_offset_bytes == expected_src) &&
                                              (candidate.dst_offset_bytes == expected_dst) &&
                                              (candidate.transfer_size_bytes == current_transfer.transfer_size_bytes);

                        if (matches_stride) {
                            stride_group_indices.push_back(k);
                            k++;
                        } else {
                            break;
                        }
                    }

                    // Only coalesce if we have at least 2 transfers in the stride pattern
                    if (stride_group_indices.size() >= 2) {
                        // Create a coalesced transfer representing the stride pattern
                        // Note: This is conceptual - the actual kernel would need to handle strided transfers
                        size_t total_transfers = stride_group_indices.size();
                        current_transfer.transfer_size_bytes =
                            current_transfer.transfer_size_bytes * total_transfers;  // Total bytes
                        current_transfer.length = current_transfer.length * total_transfers;

                        // Store stride information (would need to extend LowLevelGatherTransfer to support this)
                        // For now, we'll log the optimization
                        log_info(
                            tt::LogType::LogAlways,
                            "STRIDE COALESCED: {} transfers into 1 strided transfer (stride={}, segment_size={}, "
                            "total_bytes={})",
                            total_transfers,
                            stride_pattern,
                            sorted_transfers[i].transfer_size_bytes,
                            current_transfer.transfer_size_bytes);

                        j = stride_group_indices.back() + 1;  // Skip all coalesced transfers
                    } else {
                        // Not enough for stride coalescing
                        log_info(
                            tt::LogType::LogAlways,
                            "Detected stride pattern but insufficient transfers: src[{}→{}] dst[{}→{}] stride={}",
                            current_transfer.src_offset_bytes,
                            current_transfer.src_offset_bytes + current_transfer.transfer_size_bytes,
                            current_transfer.dst_offset_bytes,
                            next_transfer.dst_offset_bytes,
                            stride_pattern);
                        j = i + 1;
                    }
                } else {
                    j = i + 1;  // No coalescing possible
                }
            }

            // Add the (possibly coalesced) transfer
            optimized_group.transfers.push_back(current_transfer);
            i = j;
        }

        // Log detailed transfer information for debugging
        if (group.transfers.size() > 1) {
            log_info(
                tt::LogType::LogAlways,
                "Block[{}:{}] has {} transfers:",
                group.dst_shard_idx,
                group.dst_block_idx,
                group.transfers.size());
            for (size_t i = 0; i < std::min(group.transfers.size(), size_t(4)); ++i) {
                const auto& t = sorted_transfers[i];
                log_info(
                    tt::LogType::LogAlways,
                    "  [{}]: src_shard={} src_off={} dst_off={} size={} noc=({},{})",
                    i,
                    t.src_shard_idx,
                    t.src_offset_bytes,
                    t.dst_offset_bytes,
                    t.transfer_size_bytes,
                    t.src_noc_x,
                    t.src_noc_y);
            }
            if (group.transfers.size() > 4) {
                log_info(tt::LogType::LogAlways, "  ... and {} more transfers", group.transfers.size() - 4);
            }
        }

        // Log the optimization results
        if (optimized_group.transfers.size() != group.transfers.size()) {
            log_info(
                tt::LogType::LogAlways,
                "Coalesced block[{}:{}]: {} transfers -> {} transfers (saved {} NOC ops)",
                group.dst_shard_idx,
                group.dst_block_idx,
                group.transfers.size(),
                optimized_group.transfers.size(),
                group.transfers.size() - optimized_group.transfers.size());
        } else if (group.transfers.size() > 1) {
            log_info(
                tt::LogType::LogAlways,
                "Block[{}:{}]: No coalescing possible ({} transfers remain separate)",
                group.dst_shard_idx,
                group.dst_block_idx,
                group.transfers.size());
        }

        optimized_groups.push_back(std::move(optimized_group));
    }

    // Count total transfers after optimization
    size_t total_transfers_after = 0;
    for (const auto& group : optimized_groups) {
        total_transfers_after += group.transfers.size();
    }

    log_info(
        tt::LogType::LogAlways,
        "Coalescing complete: {} total transfers -> {} total transfers (saved {} NOC ops)",
        total_transfers_before,
        total_transfers_after,
        total_transfers_before - total_transfers_after);

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

    // Call the original implementation
    return group_transfers_by_output_column_blocks(
        transfers, B, C, HW, input_cores_vec, num_output_cores, element_size_bytes, block_size);
}

/**
 * @brief Split blocked transfer groups by destination core for per-core processing
 *
 * This function reorganizes BlockedTransferGroup objects by their destination core index,
 * creating a vector of vectors where each index corresponds to an output core and contains
 * all the BlockedTransferGroup objects that write to that core.
 *
 * This is useful for distributing work to individual cores, as each core only needs to
 * process the transfers that are relevant to its local memory.
 *
 * @param blocked_groups Vector of BlockedTransferGroup objects to split
 * @param num_output_cores Total number of output cores (determines the size of outer vector)
 * @return Vector of vectors where index i contains all BlockedTransferGroup objects
 *         that write to output core i. Some inner vectors may be empty if a core
 *         has no transfers assigned to it.
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

}  // namespace ttnn::operations::experimental::cnn::convert_to_hwc::detail

// fmt formatter implementations
namespace fmt {

auto formatter<ttnn::operations::experimental::cnn::convert_to_hwc::detail::GatherTransfer>::format(
    const ttnn::operations::experimental::cnn::convert_to_hwc::detail::GatherTransfer& t, format_context& ctx) const
    -> format_context::iterator {
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

auto formatter<ttnn::operations::experimental::cnn::convert_to_hwc::detail::LowLevelGatherTransfer>::format(
    const ttnn::operations::experimental::cnn::convert_to_hwc::detail::LowLevelGatherTransfer& t,
    format_context& ctx) const -> format_context::iterator {
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

auto formatter<ttnn::operations::experimental::cnn::convert_to_hwc::detail::BlockedTransferGroup>::format(
    const ttnn::operations::experimental::cnn::convert_to_hwc::detail::BlockedTransferGroup& t,
    format_context& ctx) const -> format_context::iterator {
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
