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

namespace ttnn::operations::experimental::cnn::convert_to_chw::detail {

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
    uint32_t num_input_cores,
    uint32_t num_output_cores) {
    std::vector<LowLevelGatherTransfer> low_level_transfers;

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

        low_level_transfers.emplace_back(
            t.src_core_idx, src_absolute_offset, t.dst_core_idx, dst_absolute_offset, t.length);
    }

    return low_level_transfers;
}

std::vector<BlockedTransferGroup> group_transfers_by_output_column_blocks(
    const std::vector<GatherTransfer>& transfers,
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    uint32_t num_input_cores,
    uint32_t num_output_cores,
    uint32_t block_size) {
    // Dictionary to group transfers by (dst_shard_idx, column_block_idx)
    std::map<std::pair<uint32_t, uint32_t>, std::vector<LowLevelGatherTransfer>> groups;

    // Lower transfers to get flat offsets
    auto low_level_transfers = lower_gather_transfers(transfers, B, C, HW, num_input_cores, num_output_cores);

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

    return blocked_groups;
}

/**
 * Float-specific wrapper for testing and demonstration
 *
 * This function shows how the gather operation works with float32 data.
 * For production use with different data types (e.g., bfloat16 on device),
 * you can call gather_with_blocked_transfers_generic directly:
 *
 * Example with bfloat16:
 * ```cpp
 * std::vector<uint16_t> bf16_input(...);  // Your bfloat16 data
 * std::vector<uint16_t> bf16_output(...);
 *
 * gather_with_blocked_transfers_generic(
 *     bf16_input.data(),      // Input as void*
 *     bf16_output.data(),     // Output as void*
 *     sizeof(uint16_t),       // element_size = 2 for bfloat16
 *     B, C, HW,
 *     input_cores, output_cores,
 *     block_size
 * );
 * ```
 *
 * The element_size parameter makes this design more aligned with hardware
 * DMA operations, which work with byte counts rather than type information.
 */
std::vector<std::vector<float>> gather_with_blocked_transfers(
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    const std::vector<std::vector<float>>& input_shards,
    uint32_t block_size) {
    uint32_t num_input_cores = input_cores.size();
    uint32_t num_output_cores = output_cores.size();

    // Input validation
    TT_FATAL(HW % num_input_cores == 0, "HW={} must be divisible by num_input_cores={}", HW, num_input_cores);
    TT_FATAL(
        (B * HW) % num_output_cores == 0, "B*HW={} must be divisible by num_output_cores={}", B * HW, num_output_cores);

    // First, precompute the high-level transfer list
    auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);

    // Calculate output dimensions
    uint32_t output_shard_height = C;
    uint32_t output_shard_width = B * HW / num_output_cores;

    // Group transfers by output column blocks
    auto blocked_groups =
        group_transfers_by_output_column_blocks(transfers, B, C, HW, num_input_cores, num_output_cores, block_size);

    // Flatten input shards for C-style access
    std::vector<std::vector<float>> input_shards_flat;
    for (const auto& shard : input_shards) {
        input_shards_flat.push_back(shard);  // Already flattened in this implementation
    }

    // Initialize output shards
    std::vector<std::vector<float>> output_shards(num_output_cores);
    for (uint32_t i = 0; i < num_output_cores; i++) {
        output_shards[i].resize(output_shard_height * output_shard_width, -1.0f);
    }

    // Process each column block
    for (const auto& group : blocked_groups) {
        // Calculate column range for this block
        uint32_t col_start = group.dst_block_idx * block_size;
        uint32_t col_end = std::min(col_start + block_size, output_shard_width);
        uint32_t actual_block_width = col_end - col_start;

        // Allocate temporary buffer for this column block (all rows, block_size columns)
        // In real hardware, this could be in local memory
        std::vector<float> block_buffer(output_shard_height * actual_block_width, 0.0f);

        // Execute all transfers for this column block
        for (const auto& transfer : group.transfers) {
            // Get source data
            const auto& src_flat = input_shards_flat[transfer.src_shard_idx];

            // For each element in the transfer
            for (uint32_t i = 0; i < transfer.length; i++) {
                uint32_t src_idx = transfer.src_offset + i;
                uint32_t dst_idx = transfer.dst_offset + i;

                // Calculate 2D position in output shard
                uint32_t dst_row = dst_idx / output_shard_width;
                uint32_t dst_col = dst_idx % output_shard_width;

                // Skip if this element is outside the current column block
                if (dst_col < col_start || dst_col >= col_end) {
                    continue;
                }

                // Calculate position within the column block
                uint32_t block_col = dst_col - col_start;

                // Write to block buffer
                block_buffer[dst_row * actual_block_width + block_col] = src_flat[src_idx];
            }
        }

        // Write the complete column block to the output shard
        auto& output_shard = output_shards[group.dst_shard_idx];
        for (uint32_t row = 0; row < output_shard_height; row++) {
            std::memcpy(
                &output_shard[row * output_shard_width + col_start],
                &block_buffer[row * actual_block_width],
                actual_block_width * sizeof(float));
        }
    }

    return output_shards;
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

    // Get core counts from tensors
    const auto& input_shard_spec = input.shard_spec().value();
    const auto& input_core_grid = input_shard_spec.grid;
    auto input_cores_vec = corerange_to_cores(
        input_core_grid, std::nullopt, input_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    uint32_t num_input_cores = input_cores_vec.size();

    const auto& output_shard_spec = output.shard_spec().value();
    const auto& output_core_grid = output_shard_spec.grid;
    auto output_cores_vec = corerange_to_cores(
        output_core_grid, std::nullopt, output_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    uint32_t num_output_cores = output_cores_vec.size();

    // Call the original implementation
    return group_transfers_by_output_column_blocks(transfers, B, C, HW, num_input_cores, num_output_cores, block_size);
}

/**
 * @brief Implementation of blocked gather with configurable element size
 *
 * This implementation keeps all offset calculations in elements and uses
 * element_size only for memory operations. This design is more aligned
 * with hardware DMA operations which work with byte counts.
 *
 * Memory efficiency:
 * - Block buffer size = C × block_size × element_size bytes
 * - For C=16, block_size=4:
 *   - float32 (element_size=4): 16 × 4 × 4 = 256 bytes
 *   - bfloat16 (element_size=2): 16 × 4 × 2 = 128 bytes
 */
void gather_with_blocked_transfers_generic(
    const void* input_data,
    void* output_data,
    uint32_t element_size,
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    uint32_t block_size) {
    // First, precompute transfers (element-based, size-agnostic)
    auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);

    // Group transfers by output column blocks
    auto blocked_groups = group_transfers_by_output_column_blocks(
        transfers, B, C, HW, input_cores.size(), output_cores.size(), block_size);

    uint32_t input_shard_width = HW / input_cores.size();
    uint32_t output_shard_width = B * HW / output_cores.size();
    uint32_t output_shard_height = C;

    // Cast to byte pointers for arithmetic
    const uint8_t* input_bytes = static_cast<const uint8_t*>(input_data);
    uint8_t* output_bytes = static_cast<uint8_t*>(output_data);

    // Process each block group
    for (const auto& group : blocked_groups) {
        uint32_t col_start = group.dst_block_idx * block_size;
        uint32_t col_end = std::min(col_start + block_size, output_shard_width);
        uint32_t actual_block_width = col_end - col_start;

        // Allocate block buffer in bytes
        uint32_t block_buffer_elements = output_shard_height * actual_block_width;
        std::vector<uint8_t> block_buffer(block_buffer_elements * element_size);

        // Execute transfers into block buffer
        for (const auto& transfer : group.transfers) {
            // Calculate source pointer for this shard
            const uint8_t* src_shard_base =
                input_bytes + (transfer.src_shard_idx * B * C * input_shard_width * element_size);

            const uint8_t* src_ptr = src_shard_base + (transfer.src_offset * element_size);

            // Process the transfer - copy elements that belong to this block
            for (uint32_t i = 0; i < transfer.length; i++) {
                uint32_t dst_idx = transfer.dst_offset + i;
                uint32_t dst_row = dst_idx / output_shard_width;
                uint32_t dst_col = dst_idx % output_shard_width;

                // Check if this element belongs in the current block
                if (dst_col >= col_start && dst_col < col_end) {
                    uint32_t block_col = dst_col - col_start;
                    uint32_t block_idx = dst_row * actual_block_width + block_col;

                    // Copy element_size bytes
                    std::memcpy(&block_buffer[block_idx * element_size], &src_ptr[i * element_size], element_size);
                }
            }
        }

        // Copy block buffer to output shard
        uint8_t* output_shard_base =
            output_bytes + (group.dst_shard_idx * output_shard_height * output_shard_width * element_size);

        for (uint32_t row = 0; row < output_shard_height; row++) {
            // Copy entire row of the block
            std::memcpy(
                &output_shard_base[(row * output_shard_width + col_start) * element_size],
                &block_buffer[row * actual_block_width * element_size],
                actual_block_width * element_size);
        }
    }
}

}  // namespace ttnn::operations::experimental::cnn::convert_to_chw::detail

// fmt formatter implementations
namespace fmt {

auto formatter<ttnn::operations::experimental::cnn::convert_to_chw::detail::GatherTransfer>::format(
    const ttnn::operations::experimental::cnn::convert_to_chw::detail::GatherTransfer& t, format_context& ctx) const
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

auto formatter<ttnn::operations::experimental::cnn::convert_to_chw::detail::LowLevelGatherTransfer>::format(
    const ttnn::operations::experimental::cnn::convert_to_chw::detail::LowLevelGatherTransfer& t,
    format_context& ctx) const -> format_context::iterator {
    std::string str = fmt::format(
        "LowLevelGatherTransfer(src_shard{}[{}:{}] → dst_shard{}[{}:{}], len={})",
        t.src_shard_idx,
        t.src_offset,
        t.src_offset + t.length,
        t.dst_shard_idx,
        t.dst_offset,
        t.dst_offset + t.length,
        t.length);
    return fmt::format_to(ctx.out(), "{}", str);
}

auto formatter<ttnn::operations::experimental::cnn::convert_to_chw::detail::BlockedTransferGroup>::format(
    const ttnn::operations::experimental::cnn::convert_to_chw::detail::BlockedTransferGroup& t,
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
