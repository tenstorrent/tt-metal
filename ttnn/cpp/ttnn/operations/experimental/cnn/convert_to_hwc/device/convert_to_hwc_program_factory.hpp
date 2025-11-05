// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::cnn::detail {

// Named constants for circular buffer indices
namespace CBIndex {
constexpr uint32_t CB_IN = tt::CBIndex::c_0;
constexpr uint32_t CB_IN_BATCH = tt::CBIndex::c_1;
constexpr uint32_t CB_IN_TILED = tt::CBIndex::c_2;
constexpr uint32_t CB_IN_TRANSPOSE_0 = tt::CBIndex::c_3;
constexpr uint32_t CB_IN_TRANSPOSE_1 = tt::CBIndex::c_4;
constexpr uint32_t CB_OUT = tt::CBIndex::c_5;
}  // namespace CBIndex

// Configuration class to encapsulate operation parameters
struct ConvertToHwcConfig {
    // Input tensor properties
    uint32_t batch_size;
    uint32_t input_channels;
    uint32_t hw_total;
    uint32_t element_size_bytes;
    tt::DataFormat input_format;

    // Shard specifications
    uint32_t l1_input_shard_height;
    uint32_t l1_input_shard_width;
    uint32_t output_shard_height;
    uint32_t output_shard_width;

    // Core information
    std::vector<CoreCoord> l1_input_cores;
    std::vector<CoreCoord> dram_input_cores;
    CoreRangeSet l1_input_core_grid;

    // DRAM/L1 configuration
    bool is_input_in_dram;
    uint32_t remote_address;
    tt::tt_metal::BufferType remote_buffer_type;
    tt::CoreType remote_core_type;

    // Alignment requirements
    uint32_t alignment_elements;

    static ConvertToHwcConfig create_from_tensors(const Tensor& input, const Tensor& output);
    void validate() const;
};

struct BatchTransferInstruction {
    uint32_t src_core_idx;
    uint32_t dst_core_idx;
    CoreCoord src_core_coord;
    CoreCoord dst_core_coord;
    uint32_t src_offset;
    uint32_t dst_offset;
    uint32_t transfer_size;
    uint32_t bank_id;  // 0 for L1 transfers, actual bank_id for DRAM transfers
};

struct TransferData {
    uint32_t src_offset;
    uint32_t dst_offset;
    uint32_t size;
};

// Generate individual transfers for a single destination core
std::map<uint32_t, std::vector<TransferData>> generate_transfers_for_output_core(
    uint32_t dst_core,
    uint32_t batch_size,
    uint32_t channels,
    uint32_t hw_total,
    uint32_t input_num_cores,
    uint32_t output_num_cores,
    uint32_t element_size_bytes);

// Optimize transfers using batch-aware grouping
std::vector<BatchTransferInstruction> optimize_transfers_batch_aware(
    const std::map<uint32_t, std::vector<TransferData>>& transfers_by_src,
    uint32_t dst_core,
    uint32_t batch_size,
    uint32_t channels,
    uint32_t hw_total,
    uint32_t input_num_cores,
    uint32_t element_size_bytes,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores);

// Log transfer generation parameters and results
void log_transfer_generation_info(
    uint32_t batch_size,
    uint32_t channels,
    uint32_t hw_total,
    uint32_t input_num_cores,
    uint32_t output_num_cores,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    const std::vector<BatchTransferInstruction>& instructions);

std::vector<BatchTransferInstruction> generate_batch_redistribution_transfers(
    uint32_t batch_size,
    uint32_t channels,
    uint32_t hw_total,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    uint32_t element_size_bytes);

template <typename T>
std::vector<std::vector<T>> group_by_destination_core(const std::vector<T>& transfers, int num_output_cores);

void populate_dram_bank_ids(
    std::vector<BatchTransferInstruction>& transfers,
    const std::vector<CoreCoord>& dram_cores,
    const tt::tt_metal::BufferType& dram_buffer_type,
    tt::tt_metal::IDevice* device);

uint32_t compute_alignment_requirement_in_elements(const Tensor& input_tensor);

tt::tt_metal::operation::ProgramWithCallbacks multi_core_convert_to_hwc(const Tensor& a, Tensor& output);

}  // namespace ttnn::operations::experimental::cnn::detail
