// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/core.h>
#include <functional>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

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
 *
 * Note: For blocked transfers (used in group_transfers_by_output_column_blocks),
 * dst_offset and dst_offset_bytes are relative to the block start, not absolute.
 * This semantic difference exists because the kernel expects block-relative offsets
 * for blocked transfers.
 */
struct LowLevelGatherTransfer {
    uint32_t src_shard_idx;        // Which input shard (0 to num_input_cores-1)
    uint32_t src_offset;           // Absolute offset within the source shard (in elements)
    uint32_t dst_shard_idx;        // Which output shard (0 to num_output_cores-1)
    uint32_t dst_offset;           // Absolute offset within the destination shard (in elements)
                                   // For blocked transfers: relative to block start (see note above)
    uint32_t length;               // Number of elements to transfer
    uint32_t src_noc_x;            // Source NOC X coordinate
    uint32_t src_noc_y;            // Source NOC Y coordinate
    uint32_t src_offset_bytes;     // Absolute offset within the source shard (in bytes)
    uint32_t dst_offset_bytes;     // Absolute offset within the destination shard (in bytes)
    uint32_t transfer_size_bytes;  // Number of bytes to transfer
    uint32_t bank_id;              // DRAM bank id (0 for L1)

    LowLevelGatherTransfer(uint32_t ssi, uint32_t so, uint32_t dsi, uint32_t doff, uint32_t len) :
        src_shard_idx(ssi),
        src_offset(so),
        dst_shard_idx(dsi),
        dst_offset(doff),
        length(len),
        src_noc_x(0),
        src_noc_y(0),
        src_offset_bytes(0),
        dst_offset_bytes(0),
        transfer_size_bytes(0),
        bank_id(0) {}

    LowLevelGatherTransfer(
        uint32_t ssi,
        uint32_t so,
        uint32_t dsi,
        uint32_t doff,
        uint32_t len,
        uint32_t snx,
        uint32_t sny,
        uint32_t sob,
        uint32_t dob,
        uint32_t tsb) :
        src_shard_idx(ssi),
        src_offset(so),
        dst_shard_idx(dsi),
        dst_offset(doff),
        length(len),
        src_noc_x(snx),
        src_noc_y(sny),
        src_offset_bytes(sob),
        dst_offset_bytes(dob),
        transfer_size_bytes(tsb),
        bank_id(0) {}
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

// Variant that allows overriding the per-output-core width (useful for uneven output sharding)
std::vector<GatherTransfer> precompute_gather_transfers(
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    uint32_t output_shard_width_override);

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
 * @param input_cores Vector of input core coordinates
 * @param num_output_cores Number of output cores
 * @param element_size_bytes Size of each element in bytes
 * @return Vector of LowLevelGatherTransfer objects
 */
std::vector<LowLevelGatherTransfer> lower_gather_transfers(
    const std::vector<GatherTransfer>& transfers,
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    uint32_t num_output_cores,
    uint32_t element_size_bytes,
    uint32_t output_shard_width);

// Forward declaration for API that returns both groups and count
struct BlockedTransfersWithCount;

/**
 * @brief Group transfers by output column blocks for memory-efficient processing
 *
 * Returns both the grouped transfers and the number of unique logical blocks.
 *
 * @param transfers High-level transfer list
 * @param B Batch size
 * @param C Number of channels
 * @param HW Total spatial dimension
 * @param input_cores Vector of input core coordinates
 * @param num_output_cores Number of output cores
 * @param element_size_bytes Size of each element in bytes
 * @param block_size Width of each column block (default 4)
 * @param output_shard_width Explicit width of each destination shard row
 * @return BlockedTransfersWithCount {blocked_transfers, num_logical_blocks}
 */
BlockedTransfersWithCount group_transfers_by_output_column_blocks(
    const std::vector<GatherTransfer>& transfers,
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    uint32_t num_output_cores,
    uint32_t element_size_bytes,
    uint32_t block_size,
    uint32_t output_shard_width);

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
    const Tensor& input,
    const Tensor& output,
    const std::vector<GatherTransfer>& transfers,
    uint32_t block_size,
    uint32_t output_shard_width);

/**
 * @brief Wrapper struct that returns both blocked transfers and the actual number of logical blocks
 */
struct BlockedTransfersWithCount {
    std::vector<BlockedTransferGroup> blocked_transfers;
    uint32_t num_logical_blocks;
};

// Note: group_transfers_by_output_column_blocks returns both groups and logical block count.

/**
 * @brief Coalesce contiguous transfers within blocked transfer groups to reduce NOC operations
 *
 * Analyzes transfers within each BlockedTransferGroup and merges contiguous transfers
 * that read from the same source core and write to adjacent destination addresses.
 * This optimization reduces the number of NOC operations and improves performance.
 *
 * @param blocked_groups Vector of BlockedTransferGroup objects to optimize
 * @return Vector of optimized BlockedTransferGroup objects with coalesced transfers
 */
std::vector<BlockedTransferGroup> coalesce_contiguous_transfers(
    const std::vector<BlockedTransferGroup>& blocked_groups);

/**
 * @brief Split blocked transfer groups by destination core for per-core processing
 *
 * This function reorganizes BlockedTransferGroup objects by their destination core index,
 * creating a vector of vectors where each index corresponds to an output core and contains
 * all the BlockedTransferGroup objects that write to that core.
 *
 * @param blocked_groups Vector of BlockedTransferGroup objects to split
 * @param num_output_cores Total number of output cores (determines the size of outer vector)
 * @return Vector of vectors where index i contains all BlockedTransferGroup objects
 *         that write to output core i. Some inner vectors may be empty if a core
 *         has no transfers assigned to it.
 */
std::vector<std::vector<BlockedTransferGroup>> split_by_destination_core(
    const std::vector<BlockedTransferGroup>& blocked_groups, uint32_t num_output_cores);

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

inline void serialize_low_level_transfer(
    const LowLevelGatherTransfer& transfer,
    std::vector<uint32_t>& output,
    const std::vector<CoreCoord>& input_cores,
    const std::function<CoreCoord(const CoreCoord&)>& logical_to_worker_core) {
    // Convert logical core coordinates to worker core coordinates
    CoreCoord logical_core = input_cores[transfer.src_shard_idx];
    auto worker_core = logical_to_worker_core(logical_core);

    // Kernel pulls data from input so only source core x/y is required
    output.push_back(worker_core.x);
    output.push_back(worker_core.y);
    output.push_back(transfer.src_offset_bytes);
    output.push_back(transfer.dst_offset_bytes);
    output.push_back(transfer.transfer_size_bytes);
    // Always serialize bank_id as the 6th value per transfer.
    // For DRAM, logical_to_worker_core encodes the bank_id in x; for L1, this value is ignored by the kernel.
    output.push_back(worker_core.x);
}

inline std::vector<uint32_t> serialize_blocked_transfer_groups(
    const std::vector<BlockedTransferGroup>& groups,
    const std::vector<CoreCoord>& input_cores,
    const std::function<CoreCoord(const CoreCoord&)>& logical_to_worker_core) {
    std::vector<uint32_t> output;
    const uint32_t number_of_blocks = groups.size();
    output.push_back(number_of_blocks);
    for (const auto& group : groups) {
        const uint32_t group_size = group.transfers.size();
        output.push_back(group_size);
        for (const auto& transfer : group.transfers) {
            serialize_low_level_transfer(transfer, output, input_cores, logical_to_worker_core);
        }
    }
    return output;
}

}  // namespace ttnn::experimental::prim

// fmt formatter template specializations for pretty printing
template <>
struct fmt::formatter<ttnn::experimental::prim::GatherTransfer> : formatter<string_view> {
    auto format(const ttnn::experimental::prim::GatherTransfer& t, fmt::format_context& ctx) const
        -> format_context::iterator;
};

template <>
struct fmt::formatter<ttnn::experimental::prim::LowLevelGatherTransfer> : formatter<string_view> {
    auto format(const ttnn::experimental::prim::LowLevelGatherTransfer& t, fmt::format_context& ctx) const
        -> format_context::iterator;
};

template <>
struct fmt::formatter<ttnn::experimental::prim::BlockedTransferGroup> : formatter<string_view> {
    auto format(const ttnn::experimental::prim::BlockedTransferGroup& t, fmt::format_context& ctx) const
        -> format_context::iterator;
};
