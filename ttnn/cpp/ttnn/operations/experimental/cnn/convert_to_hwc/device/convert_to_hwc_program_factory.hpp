// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::cnn::detail {

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
