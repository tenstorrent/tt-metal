// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <map>
#include <algorithm>
#include <fmt/core.h>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::cnn::convert_to_chw::detail {

/**
 * @brief High-level transfer representation with semantic information
 *
 * Represents a data transfer from input core to output core with batch/channel context.
 * Used during transfer precomputation phase to track data movement patterns.
 *
 * Note: All offsets and lengths are in elements, not bytes. The actual
 * byte size depends on the data type (e.g., 2 bytes for bfloat16, 4 for float32).
 */
struct GatherTransfer {
    uint32_t src_core_idx;
    uint32_t dst_core_idx;
    CoreCoord src_core_coord;
    CoreCoord dst_core_coord;
    uint32_t src_offset;  // Offset within source shard (in elements)
    uint32_t dst_offset;  // Offset within destination shard (in elements)
    uint32_t length;      // Number of elements to transfer
    uint32_t channel;     // Channel index for this transfer
    uint32_t batch;       // Batch index for this transfer

    GatherTransfer(
        uint32_t sc_idx,
        uint32_t dc_idx,
        const CoreCoord& sc_coord,
        const CoreCoord& dc_coord,
        uint32_t s_off,
        uint32_t d_off,
        uint32_t len,
        uint32_t ch,
        uint32_t b) :
        src_core_idx(sc_idx),
        dst_core_idx(dc_idx),
        src_core_coord(sc_coord),
        dst_core_coord(dc_coord),
        src_offset(s_off),
        dst_offset(d_off),
        length(len),
        channel(ch),
        batch(b) {}
};

/**
 * @brief Low-level transfer representation using absolute offsets
 *
 * Suitable for hardware implementation with raw memory addresses.
 * All offsets are absolute within the flattened shard arrays.
 */
struct LowLevelGatherTransfer {
    uint32_t src_shard_idx;  // Which input shard (0 to num_input_cores-1)
    uint32_t src_offset;     // Absolute offset within the source shard (in elements)
    uint32_t dst_shard_idx;  // Which output shard (0 to num_output_cores-1)
    uint32_t dst_offset;     // Absolute offset within the destination shard (in elements)
    uint32_t length;         // Number of elements to transfer

    LowLevelGatherTransfer(uint32_t ssi, uint32_t so, uint32_t dsi, uint32_t doff, uint32_t len) :
        src_shard_idx(ssi), src_offset(so), dst_shard_idx(dsi), dst_offset(doff), length(len) {}
};

/**
 * @brief Group of transfers that write to the same column block of the output
 *
 * All transfers in this group write to columns [block_idx*block_size : (block_idx+1)*block_size].
 * Used for memory-efficient blocked processing.
 */
struct BlockedTransferGroup {
    uint32_t dst_shard_idx;                         // Which output shard
    uint32_t dst_block_idx;                         // Which column block (0, 1, 2, ...)
    uint32_t block_size;                            // Width of the column block
    std::vector<LowLevelGatherTransfer> transfers;  // All transfers writing to this column block

    BlockedTransferGroup(uint32_t dsi, uint32_t dbi, uint32_t bs) :
        dst_shard_idx(dsi), dst_block_idx(dbi), block_size(bs) {}
};

/**
 * @brief Precompute all transfers needed for the gather operation
 *
 * Analyzes the data layout transformation from [B, C, HW] to [C, B, HW] and generates
 * an explicit list of all data movements required between distributed cores.
 *
 * @param B Batch size
 * @param C Number of channels
 * @param HW Total spatial dimension (height * width)
 * @param input_cores Vector of input core coordinates
 * @param output_cores Vector of output core coordinates
 * @return Vector of GatherTransfer objects sorted by (src_core, batch*C + channel, src_offset)
 */
std::vector<GatherTransfer> precompute_gather_transfers(
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores);

/**
 * @brief Convert high-level transfers to low-level transfers with absolute offsets
 *
 * This function bridges the gap between semantic transfers and raw memory operations,
 * converting logical coordinates into absolute memory offsets suitable for DMA.
 *
 * @param transfers High-level transfer list
 * @param B Batch size
 * @param C Number of channels
 * @param HW Total spatial dimension
 * @param num_input_cores Number of input cores
 * @param num_output_cores Number of output cores
 * @return Vector of LowLevelGatherTransfer objects
 */
std::vector<LowLevelGatherTransfer> lower_gather_transfers(
    const std::vector<GatherTransfer>& transfers,
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    uint32_t num_input_cores,
    uint32_t num_output_cores);

/**
 * @brief Group transfers by output column blocks for memory-efficient processing
 *
 * Analyzes all transfers and groups them by which column block of the output they write to,
 * enabling the output to be generated in small, cache-friendly chunks.
 *
 * @param transfers High-level transfer list
 * @param B Batch size
 * @param C Number of channels
 * @param HW Total spatial dimension
 * @param num_input_cores Number of input cores
 * @param num_output_cores Number of output cores
 * @param block_size Width of each column block (default 4)
 * @return Vector of BlockedTransferGroup objects sorted by (dst_shard_idx, dst_block_idx)
 */
std::vector<BlockedTransferGroup> group_transfers_by_output_column_blocks(
    const std::vector<GatherTransfer>& transfers,
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    uint32_t num_input_cores,
    uint32_t num_output_cores,
    uint32_t block_size = 4);

/**
 * @brief Tensor-based interface for gather transfers precomputation
 *
 * This function analyzes the input tensor's sharding and generates transfers
 * needed to reorganize data from [B, C, HW] to [C, B, HW] format.
 *
 * @param input Input tensor with [B*C, HW] sharding across cores
 * @param output_cores Vector of output core coordinates
 * @return Vector of GatherTransfer objects sorted by source for optimal cache access
 */
std::vector<GatherTransfer> precompute_gather_transfers(
    const Tensor& input, const std::vector<CoreCoord>& output_cores);

/**
 * @brief Tensor-based interface for blocked transfer grouping
 *
 * Groups transfers by output column blocks for memory-efficient processing
 * using the tensor's metadata to determine sharding information.
 *
 * @param input Input tensor with sharding information
 * @param output Output tensor with sharding information
 * @param transfers High-level transfer list
 * @param block_size Width of each column block (default 4)
 * @return Vector of BlockedTransferGroup objects
 */
std::vector<BlockedTransferGroup> group_transfers_by_output_column_blocks(
    const Tensor& input, const Tensor& output, const std::vector<GatherTransfer>& transfers, uint32_t block_size = 4);

/**
 * @brief Convert element count to byte count
 *
 * @param element_count Number of elements
 * @param element_size Size of each element in bytes
 * @return Number of bytes
 */
inline uint32_t elements_to_bytes(uint32_t element_count, uint32_t element_size) {
    return element_count * element_size;
}

}  // namespace ttnn::operations::experimental::cnn::convert_to_chw::detail

// fmt formatter template specializations for pretty printing
template <>
struct fmt::formatter<ttnn::operations::experimental::cnn::convert_to_chw::detail::GatherTransfer>
    : formatter<string_view> {
    auto format(
        const ttnn::operations::experimental::cnn::convert_to_chw::detail::GatherTransfer& t,
        fmt::format_context& ctx) const -> format_context::iterator;
};

template <>
struct fmt::formatter<ttnn::operations::experimental::cnn::convert_to_chw::detail::LowLevelGatherTransfer>
    : formatter<string_view> {
    auto format(
        const ttnn::operations::experimental::cnn::convert_to_chw::detail::LowLevelGatherTransfer& t,
        fmt::format_context& ctx) const -> format_context::iterator;
};

template <>
struct fmt::formatter<ttnn::operations::experimental::cnn::convert_to_chw::detail::BlockedTransferGroup>
    : formatter<string_view> {
    auto format(
        const ttnn::operations::experimental::cnn::convert_to_chw::detail::BlockedTransferGroup& t,
        fmt::format_context& ctx) const -> format_context::iterator;
};
