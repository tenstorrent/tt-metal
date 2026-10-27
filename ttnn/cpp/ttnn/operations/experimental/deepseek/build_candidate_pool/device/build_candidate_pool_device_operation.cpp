// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "build_candidate_pool_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::build_candidate_pool {

using namespace tt::tt_metal;

namespace {

void require_int32_rm_interleaved(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "build_candidate_pool: {} must be on device", name);
    TT_FATAL(
        tensor.dtype() == DataType::INT32 && tensor.layout() == Layout::ROW_MAJOR &&
            tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "build_candidate_pool: {} must be INT32 row-major interleaved",
        name);
}

void require_output(const Tensor& out, const TensorSpec& spec, const char* name) {
    TT_FATAL(out.storage_type() == StorageType::DEVICE, "build_candidate_pool: {} must be on device", name);
    TT_FATAL(
        out.logical_shape() == spec.logical_shape() && out.dtype() == spec.data_type() && out.layout() == spec.layout(),
        "build_candidate_pool: {} must be {} {} {}, got {} {} {}",
        name,
        spec.logical_shape(),
        spec.data_type(),
        spec.layout(),
        out.logical_shape(),
        out.dtype(),
        out.layout());
    TT_FATAL(
        out.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "build_candidate_pool: {} must be interleaved",
        name);
}

}  // namespace

BuildCandidatePoolDeviceOperation::program_factory_t BuildCandidatePoolDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void BuildCandidatePoolDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& blocks = tensor_args.candidate_blocks;
    require_int32_rm_interleaved(blocks, "candidate_blocks");
    TT_FATAL(
        blocks.logical_shape().rank() == 2 && blocks.logical_shape()[0] == 1,
        "build_candidate_pool: candidate_blocks must be [1, candidate_topk_blocks], got {}",
        blocks.logical_shape());
    require_int32_rm_interleaved(tensor_args.pool_count, "pool_count");
    TT_FATAL(
        tensor_args.pool_count.logical_shape().volume() == 1,
        "build_candidate_pool: pool_count must hold one count, got {}",
        tensor_args.pool_count.logical_shape());
    const auto& page_table = tensor_args.page_table_tensor;
    require_int32_rm_interleaved(page_table, "page_table_tensor");
    TT_FATAL(
        page_table.logical_shape().rank() == 2 &&
            (page_table.logical_shape()[0] == 1 || tensor_args.page_table_row_tensor.has_value()),
        "build_candidate_pool: page_table_tensor must be [1, max_blocks_per_user] (or [num_sessions, "
        "max_blocks_per_user] with page_table_row_tensor), got {}",
        page_table.logical_shape());
    if (tensor_args.page_table_row_tensor.has_value()) {
        require_int32_rm_interleaved(tensor_args.page_table_row_tensor.value(), "page_table_row_tensor");
        TT_FATAL(
            tensor_args.page_table_row_tensor->logical_shape().volume() == 1,
            "build_candidate_pool: page_table_row_tensor must hold one row, got {}",
            tensor_args.page_table_row_tensor->logical_shape());
    }

    const auto& key_cache = tensor_args.key_cache;
    TT_FATAL(
        key_cache.storage_type() == StorageType::DEVICE && key_cache.logical_shape().rank() == 4,
        "build_candidate_pool: key_cache must be a rank-4 device tensor");
    TT_FATAL(
        key_cache.layout() == Layout::TILE && key_cache.dtype() == DataType::BFLOAT16 &&
            key_cache.memory_config().is_dram(),
        "build_candidate_pool: key_cache must be bfloat16 TILE in DRAM, got {} {} {}",
        key_cache.dtype(),
        key_cache.layout(),
        key_cache.memory_config());
    const auto tile_shape = key_cache.tensor_spec().tile().get_tile_shape();
    TT_FATAL(
        tile_shape[0] == tt::constants::TILE_HEIGHT && tile_shape[1] == tt::constants::TILE_WIDTH,
        "build_candidate_pool: key_cache must use 32x32 tiles");

    const auto& key_shape = key_cache.logical_shape();
    const uint32_t page_block_size = key_shape[-2];
    TT_FATAL(key_shape[1] == 1, "build_candidate_pool: key_cache must have one head, got {}", key_shape);
    TT_FATAL(
        page_block_size % tt::constants::TILE_HEIGHT == 0 && key_shape[-1] % tt::constants::TILE_WIDTH == 0,
        "build_candidate_pool: key_cache blocks {} must be whole tiles",
        key_shape);

    // A candidate block is cbs rows of a 16-row tile face: cbs * 32 B contiguous per face.
    const uint32_t cbs = args.candidate_block_size;
    TT_FATAL(cbs >= 2 && 16 % cbs == 0, "build_candidate_pool: candidate_block_size {} must be 2, 4, 8 or 16", cbs);
    const uint32_t topk_blocks = blocks.logical_shape()[-1];
    TT_FATAL(
        (topk_blocks * cbs) % page_block_size == 0,
        "build_candidate_pool: the pool ({} blocks of {}) must fill whole page blocks of {}",
        topk_blocks,
        cbs,
        page_block_size);

    const auto specs = compute_output_specs(args, tensor_args);
    if (tensor_args.pool_keys_output.has_value()) {
        require_output(tensor_args.pool_keys_output.value(), specs[0], "pool_keys_output");
    }
    if (tensor_args.pool_rows_output.has_value()) {
        require_output(tensor_args.pool_rows_output.value(), specs[1], "pool_rows_output");
    }
}

BuildCandidatePoolDeviceOperation::spec_return_value_t BuildCandidatePoolDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& key_shape = tensor_args.key_cache.logical_shape();
    const uint32_t page_block_size = key_shape[-2];
    const uint32_t pool_keys = tensor_args.candidate_blocks.logical_shape()[-1] * args.candidate_block_size;
    return {
        TensorSpec(
            ttnn::Shape({pool_keys / page_block_size, 1, page_block_size, key_shape[-1]}),
            TensorLayout(tensor_args.key_cache.dtype(), PageConfig(Layout::TILE), MemoryConfig{})),
        TensorSpec(
            ttnn::Shape({1, pool_keys}), TensorLayout(DataType::INT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{})),
    };
}

BuildCandidatePoolDeviceOperation::tensor_return_value_t BuildCandidatePoolDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.key_cache.device();
    return {
        tensor_args.pool_keys_output.has_value() ? tensor_args.pool_keys_output.value()
                                                 : ttnn::create_device_tensor(specs[0], device),
        tensor_args.pool_rows_output.has_value() ? tensor_args.pool_rows_output.value()
                                                 : ttnn::create_device_tensor(specs[1], device),
    };
}

}  // namespace ttnn::operations::experimental::deepseek::build_candidate_pool

namespace ttnn::prim {

std::vector<ttnn::Tensor> build_candidate_pool(
    const ttnn::Tensor& candidate_blocks,
    const ttnn::Tensor& pool_count,
    const ttnn::Tensor& key_cache,
    const ttnn::Tensor& page_table_tensor,
    uint32_t candidate_block_size,
    const std::optional<ttnn::Tensor>& pool_keys_output,
    const std::optional<ttnn::Tensor>& pool_rows_output,
    const std::optional<ttnn::Tensor>& page_table_row_tensor) {
    using OperationType =
        ttnn::operations::experimental::deepseek::build_candidate_pool::BuildCandidatePoolDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{.candidate_block_size = candidate_block_size};
    auto tensor_args = OperationType::tensor_args_t{
        .candidate_blocks = candidate_blocks,
        .pool_count = pool_count,
        .key_cache = key_cache,
        .page_table_tensor = page_table_tensor,
        .pool_keys_output = pool_keys_output,
        .pool_rows_output = pool_rows_output,
        .page_table_row_tensor = page_table_row_tensor,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
