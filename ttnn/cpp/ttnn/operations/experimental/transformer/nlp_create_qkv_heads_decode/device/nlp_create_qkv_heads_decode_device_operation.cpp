// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

NLPCreateQKVHeadsDecodeDeviceOperation::program_factory_t
NLPCreateQKVHeadsDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    bool is_input_sharded = input_tensor.is_sharded();

    if (is_input_sharded) {
        if (operation_attributes.input_on_subcoregrids) {
            return NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory{};
        }
        return NLPCreateQKVHeadsDecodeShardedProgramFactory{};
    }
    return NLPCreateQKVHeadsDecodeInterleavedProgramFactory{};
}

void NLPCreateQKVHeadsDecodeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void NLPCreateQKVHeadsDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();
    const auto& batch_offset = tensor_args.batch_offset;

    // TODO: Rewrite validation for this decode case
    // NOTE: Checks for head_dim and shape[3] is done in nlp_create_qkv_heads because it's needed to infer head_dim
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "Unsupported data format");
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Only tile layout is supported for input tensor");

    // input
    const uint32_t num_users_supported = 32;
    uint32_t num_users = input_shape[2];
    TT_FATAL(
        input_shape[3] % TILE_WIDTH == 0,
        "Unsupported input shape = {}",
        input_shape);  // head_dim must be multiple of TILE_WIDTH
    TT_FATAL(num_users <= num_users_supported, "Unsupported input shape = {}", input_shape);  // 32 users
    TT_FATAL(input_shape[1] == 1, "Unsupported input shape = {}", input_shape);
    TT_FATAL(input_shape[0] == 1, "Unsupported input shape = {}", input_shape);
    const auto& QKV_memcfg = input_tensor.memory_config();
    if (input_tensor.is_sharded()) {
        TT_FATAL(
            QKV_memcfg.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
            "Current input memory layout is {}. It must be width sharded",
            QKV_memcfg.memory_layout());
        TT_FATAL(
            input_tensor.shard_spec().value().shape[0] ==
                input_tensor.physical_volume() / input_tensor.padded_shape()[-1],
            "Shard shape must be correct");
        TT_FATAL(
            input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
            "Shard orientation must be ROW_MAJOR");

        if (!operation_attributes.overlap_qk_coregrid) {
            // Validate if each shard is a multiple of head_dim and doesn't contain partial heads
            TT_FATAL(
                operation_attributes.head_dim % input_tensor.shard_spec().value().shape[1] == 0,
                "We don't support partial heads in shards when q and k heads are not overlapping coregrid");
        }
        TT_FATAL(
            !(batch_offset.has_value() ^ operation_attributes.slice_size.has_value()),
            "Both batch_offset and slice_size must be provided or neither");
        if (batch_offset.has_value() && operation_attributes.slice_size.has_value()) {
            TT_FATAL(batch_offset.value().logical_shape()[0] == 1, "batch_offset must be unary tensor");
            num_users = operation_attributes.slice_size.value();
        }

    } else {
        TT_FATAL(operation_attributes.overlap_qk_coregrid, "Overlap_qk_coregrid must be true for non-sharded input");
    }

    // output
    TT_FATAL(
        operation_attributes.output_mem_config.is_sharded() &&
            operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Output tensor must be height sharded");

    // Support maximum 32 heads for now
    TT_FATAL(
        operation_attributes.num_q_heads <= 32,
        "There are {} q heads only 32 are supported",
        operation_attributes.num_q_heads);
    TT_FATAL(
        operation_attributes.num_q_heads >= operation_attributes.num_kv_heads,
        "num_q_heads={} must be greater than or equal to num_kv_heads={}",
        operation_attributes.num_q_heads,
        operation_attributes.num_kv_heads);

    uint32_t num_cores = operation_attributes.output_mem_config.shard_spec().value().grid.num_cores();

    // 1 User Per Core Max and 32 users for now
    if (operation_attributes.overlap_qk_coregrid) {
        TT_FATAL(num_cores >= num_users, "Grid Size is {}. Need at least 32 cores for decode", num_cores);
    } else {
        TT_FATAL(
            num_cores >= 2 * num_users,
            "Input coregrid size is {}. Need cores atleast double of num_users for decode when q and k heads are not "
            "overlapping "
            "coregrid",
            num_cores);
    }
}

std::vector<ttnn::TensorSpec> NLPCreateQKVHeadsDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();

    auto batch = input_shape[2];
    if (operation_attributes.slice_size.has_value()) {
        batch = operation_attributes.slice_size.value();
    }

    auto head_dim = operation_attributes.head_dim;

    const Shape q_output_shape({input_shape[0], batch, operation_attributes.num_q_heads, head_dim});
    const Shape v_output_shape({input_shape[0], batch, operation_attributes.num_kv_heads, head_dim});
    const Shape& k_output_shape = v_output_shape;

    auto num_q_heads_padded = ((operation_attributes.num_q_heads - 1) / TILE_HEIGHT + 1) * TILE_HEIGHT;
    auto num_kv_heads_padded = ((operation_attributes.num_q_heads - 1) / TILE_HEIGHT + 1) * TILE_HEIGHT;

    CoreRangeSet output_core_grid = operation_attributes.output_mem_config.shard_spec().value().grid;
    CoreRangeSet q_shard_grid, k_shard_grid, v_shard_grid;
    auto start_core_coord = output_core_grid.ranges().front().start_coord;

    q_shard_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, batch, output_core_grid, true);
    if (operation_attributes.overlap_qk_coregrid) {
        k_shard_grid = q_shard_grid;
    } else {
        CoreRangeSet q_plus_one_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
            start_core_coord, batch + 1, output_core_grid, true);
        CoreCoord k_start_core_coord;
        if (!q_plus_one_grid.ranges().empty()) {
            k_start_core_coord = q_plus_one_grid.ranges().back().end_coord;
        }
        k_shard_grid =
            tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(k_start_core_coord, batch, output_core_grid, true);
    }
    v_shard_grid = q_shard_grid;

    tt::tt_metal::ShardSpec q_shard_spec{q_shard_grid, {num_q_heads_padded, operation_attributes.head_dim}};
    tt::tt_metal::ShardSpec k_shard_spec{k_shard_grid, {num_kv_heads_padded, operation_attributes.head_dim}};
    tt::tt_metal::ShardSpec v_shard_spec{v_shard_grid, {num_kv_heads_padded, operation_attributes.head_dim}};
    tt::tt_metal::MemoryConfig q_mem_config = operation_attributes.output_mem_config.with_shard_spec(q_shard_spec);
    tt::tt_metal::MemoryConfig k_mem_config = operation_attributes.output_mem_config.with_shard_spec(k_shard_spec);
    tt::tt_metal::MemoryConfig v_mem_config = operation_attributes.output_mem_config.with_shard_spec(v_shard_spec);

    return {
        TensorSpec(
            q_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), q_mem_config)),
        TensorSpec(
            k_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), k_mem_config)),
        TensorSpec(
            v_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), v_mem_config))};
}

std::vector<Tensor> NLPCreateQKVHeadsDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    return {
        create_device_tensor(output_specs[0], input_tensor.device()),
        create_device_tensor(output_specs[1], input_tensor.device()),
        create_device_tensor(output_specs[2], input_tensor.device())};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> nlp_create_qkv_heads_decode(
    const Tensor& input_tensor,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    bool overlap_qk_coregrid,
    bool input_on_subcoregrids,
    const std::optional<const Tensor>& batch_offset,
    std::optional<uint32_t> slice_size,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::experimental::prim::NLPCreateQKVHeadsDecodeDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .overlap_qk_coregrid = overlap_qk_coregrid,
        .input_on_subcoregrids = input_on_subcoregrids,
        .slice_size = slice_size,
        .output_mem_config = output_mem_config};
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor, .batch_offset = batch_offset};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
