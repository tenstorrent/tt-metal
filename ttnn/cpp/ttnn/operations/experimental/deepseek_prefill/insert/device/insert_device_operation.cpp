// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "insert_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::insert {

namespace {

bool is_dram_interleaved(const ttnn::Tensor& tensor) {
    const auto& mem_cfg = tensor.memory_config();
    return mem_cfg.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mem_cfg.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

void validate_index_tensor(const ttnn::Tensor& tensor, const std::string& name) {
    TT_FATAL(tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a buffer", name);
    TT_FATAL(tensor.dtype() == tt::tt_metal::DataType::UINT32, "{} must be UINT32, got {}", name, tensor.dtype());
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "{} must be ROW_MAJOR layout, got {}", name, tensor.layout());
    TT_FATAL(is_dram_interleaved(tensor), "{} must be DRAM interleaved", name);

    const auto& shape = tensor.logical_shape();
    const auto rank = shape.rank();
    const bool valid_1d = rank == 1;
    const bool valid_2d = rank == 2 && shape[0] == 1;
    TT_FATAL(valid_1d || valid_2d, "{} must be 1D or 2D with first dimension == 1, got shape {}", name, shape);
}

void validate_data_tensor(const ttnn::Tensor& tensor, const std::string& name) {
    TT_FATAL(tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a buffer", name);
    TT_FATAL(tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "{} must be BFLOAT8_B, got {}", name, tensor.dtype());
    TT_FATAL(tensor.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout, got {}", name, tensor.layout());
    TT_FATAL(is_dram_interleaved(tensor), "{} must be DRAM interleaved", name);
    TT_FATAL(tensor.logical_shape().rank() == 2, "{} must be 2D, got rank {}", name, tensor.logical_shape().rank());
}

}  // namespace

// Host-side validation covers only static invariants (dtypes, layouts, shape
// relationships, tile alignment of host-known scalars). Two *data-dependent*
// bounds are checked on-device at runtime inside the reader/writer kernels:
//   * start[id] + ceil_tile(counts[id]) <= global_rows — slice stays inside
//     global_tensor.
//   * ceil_tile(counts[id]) <= local_rows — kernel doesn't over-read local_tensor.
// Host can't check these here without reading device-resident start/counts,
// which we deliberately avoid (op must be device-local for multi-device mesh
// use — each device may have its own start/counts values).
//
// NOT CHECKED anywhere (caller's contract):
//   * start[id] + counts[id] <= start[id + 1] — this expert's slice does not
//     overlap the next expert's slice. Enforcing this would require reading
//     the adjacent expert's metadata, which breaks the "single global_expert_id,
//     no cross-expert state" contract. Upstream code laying out the dispatch
//     buffer is responsible for honoring this invariant.
void InsertDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& global_tensor = tensor_args.global_tensor;
    const auto& local_tensor = tensor_args.local_tensor;
    const auto& start = tensor_args.start;
    const auto& counts = tensor_args.counts;
    const auto& global_expert_idx_table = tensor_args.global_expert_idx_table;

    // global_tensor / local_tensor validation: 2D, BFLOAT8_B, TILE, DRAM interleaved.
    validate_data_tensor(global_tensor, "global_tensor");
    validate_data_tensor(local_tensor, "local_tensor");

    // Hidden dim must match between the two data tensors.
    const auto global_hidden_dim = global_tensor.logical_shape()[-1];
    const auto local_hidden_dim = local_tensor.logical_shape()[-1];
    TT_FATAL(
        global_hidden_dim == local_hidden_dim,
        "local_tensor hidden_dim ({}) must match global_tensor hidden_dim ({})",
        local_hidden_dim,
        global_hidden_dim);

    // start, counts, and global_expert_idx_table share the same static invariants.
    validate_index_tensor(start, "start");
    validate_index_tensor(counts, "counts");
    validate_index_tensor(global_expert_idx_table, "global_expert_idx_table");

    // Last dimension of start and counts must match.
    const auto start_last_dim = start.logical_shape()[-1];
    const auto counts_last_dim = counts.logical_shape()[-1];
    TT_FATAL(
        start_last_dim == counts_last_dim,
        "start and counts must have the same last dimension, got start={} and counts={}",
        start_last_dim,
        counts_last_dim);

    // local_expert_id must index into global_expert_idx_table's last dimension.
    // Validity of global_expert_idx_table[local_expert_id] as an index into start/counts
    // is checked in-kernel at runtime (the value is device-resident).
    const auto idx_table_last_dim = global_expert_idx_table.logical_shape()[-1];
    TT_FATAL(
        operation_attributes.local_expert_id < idx_table_last_dim,
        "local_expert_id ({}) must be in the range [0, {}] (global_expert_idx_table last dimension - 1)",
        operation_attributes.local_expert_id,
        idx_table_last_dim - 1);

    // Tile-alignment checks.
    const uint32_t tile_height = tt::constants::TILE_HEIGHT;
    const uint32_t tile_width = tt::constants::TILE_WIDTH;
    const auto global_rows = global_tensor.logical_shape()[0];
    const auto local_rows = local_tensor.logical_shape()[0];
    TT_FATAL(
        global_rows % tile_height == 0,
        "global_tensor rows/tokens ({}) must be a multiple of TILE_HEIGHT ({})",
        global_rows,
        tile_height);
    TT_FATAL(
        global_hidden_dim % tile_width == 0,
        "global_tensor hidden_dim ({}) must be a multiple of TILE_WIDTH ({})",
        global_hidden_dim,
        tile_width);
    TT_FATAL(
        local_rows % tile_height == 0,
        "local_tensor rows/tokens ({}) must be a multiple of TILE_HEIGHT ({})",
        local_rows,
        tile_height);
    TT_FATAL(local_rows > 0, "local_tensor must have at least one tile row/token");
}

void InsertDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // Empty for now.
}

InsertDeviceOperation::spec_return_value_t InsertDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // The op is in-place: the output is global_tensor.
    return tensor_args.global_tensor.tensor_spec();
}

InsertDeviceOperation::tensor_return_value_t InsertDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // In-place: reuse global_tensor. No new DRAM allocation.
    return tensor_args.global_tensor;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::insert

namespace ttnn::prim {

ttnn::Tensor prefill_insert(
    const ttnn::Tensor& global_tensor,
    const ttnn::Tensor& local_tensor,
    const ttnn::Tensor& start,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t local_expert_id) {
    using OperationType = ttnn::operations::experimental::deepseek_prefill::insert::InsertDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.local_expert_id = local_expert_id},
        OperationType::tensor_args_t{
            .global_tensor = global_tensor,
            .local_tensor = local_tensor,
            .start = start,
            .counts = counts,
            .global_expert_idx_table = global_expert_idx_table});
}

}  // namespace ttnn::prim
