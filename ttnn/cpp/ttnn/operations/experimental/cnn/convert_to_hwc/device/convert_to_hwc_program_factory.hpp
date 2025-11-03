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

    BatchTransferInstruction(
        uint32_t src_idx,
        uint32_t dst_idx,
        CoreCoord src_coord,
        CoreCoord dst_coord,
        uint32_t src_off,
        uint32_t dst_off,
        uint32_t size) :
        src_core_idx(src_idx),
        dst_core_idx(dst_idx),
        src_core_coord(src_coord),
        dst_core_coord(dst_coord),
        src_offset(src_off),
        dst_offset(dst_off),
        transfer_size(size) {}
};

struct TransferData {
    uint32_t src_offset;
    uint32_t dst_offset;
    uint32_t size;

    TransferData(uint32_t src_off, uint32_t dst_off, uint32_t transfer_size) :
        src_offset(src_off), dst_offset(dst_off), size(transfer_size) {}
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

uint32_t compute_alignment_requirement_in_elements(const Tensor& input_tensor);

tt::tt_metal::operation::ProgramWithCallbacks multi_core_convert_to_hwc(const Tensor& a, Tensor& output);

}  // namespace ttnn::operations::experimental::cnn::detail
