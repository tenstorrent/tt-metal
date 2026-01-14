// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "create_qkv_heads_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::create_qkv_heads {

CreateQKVHeadsDeviceOperation::program_factory_t CreateQKVHeadsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return ttnn::operations::experimental::create_qkv_heads::program::CreateQKVHeadsProgramFactory{};
}

void CreateQKVHeadsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void CreateQKVHeadsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(
        input_tensor.layout() == Layout::TILE, "Input tensor layout must be TILE but got {}", input_tensor.layout());
    TT_FATAL(input_tensor.is_sharded(), "Operands to TM must be sharded");
    const auto& input_shape = input_tensor.padded_shape();
    TT_FATAL(input_shape[1] == 1, "Unsupported input shape");

    auto bbox = input_tensor.shard_spec().value().grid.bounding_box();
    TT_FATAL(
        (bbox.end_coord.x < input_tensor.device()->compute_with_storage_grid_size().x &&
         bbox.end_coord.y < input_tensor.device()->compute_with_storage_grid_size().y),
        "Error");
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
        "Input tensor memory layout must be BLOCK_SHARDED but got {}",
        input_tensor.memory_config().memory_layout());
    ShardOrientation shard_orientation = input_tensor.shard_spec().value().orientation;
    bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
    uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
    uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;

    TT_FATAL(
        args.num_q_heads % args.num_kv_heads == 0,
        "Number of q heads {} must fit evenly into number of kv heads {}",
        args.num_q_heads,
        args.num_kv_heads);
    TT_FATAL(
        input_shape[3] % (num_w_cores * tt::constants::TILE_WIDTH) == 0,
        "Flattened hidden dimension {} must be a multiple of width cores {} * tile width {} to ensure that each core "
        "gets an even amount of tiles",
        input_shape[3],
        num_w_cores,
        tt::constants::TILE_WIDTH);

    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Output memory config layout must be HEIGHT_SHARDED but got {}",
        args.output_mem_config.memory_layout());
    TT_FATAL(input_shape[0] == num_h_cores, "Batch size {} must be equal to num cores {}", input_shape[0], num_h_cores);
}

spec_return_value_t CreateQKVHeadsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_outputs.has_value()) {
        const auto& [q_tensor, k_tensor, v_tensor] = tensor_args.preallocated_outputs.value();
        return {q_tensor.tensor_spec(), k_tensor.tensor_spec(), v_tensor.tensor_spec()};
    }

    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();

    const auto q_shape = ttnn::Shape{input_shape[0], args.num_q_heads, input_shape[2], args.head_dim};
    const auto v_shape = ttnn::Shape{input_shape[0], args.num_kv_heads, input_shape[2], args.head_dim};
    const auto k_shape = args.transpose_k_heads
                             ? ttnn::Shape{input_shape[0], args.num_kv_heads, args.head_dim, input_shape[2]}
                             : v_shape;

    CoreRangeSet all_cores = input_tensor.shard_spec().value().grid;
    ShardOrientation shard_orientation = input_tensor.shard_spec().value().orientation;
    auto bbox = all_cores.bounding_box();
    uint32_t num_cores = bbox.size();

    uint32_t q_shard_h =
        q_shape[0] * q_shape[1] * q_shape[2] / num_cores;  // want the API to work for different sequence lengths
    uint32_t k_shard_h =
        k_shape[0] * k_shape[1] * k_shape[2] / num_cores;  // want the API to work for different sequence lengths
    uint32_t v_shard_h =
        v_shape[0] * v_shape[1] * v_shape[2] / num_cores;  // want the API to work for different sequence lengths

    auto q_spec = tt::tt_metal::ShardSpec(all_cores, {q_shard_h, q_shape[-1]}, shard_orientation);
    auto k_spec = tt::tt_metal::ShardSpec(all_cores, {k_shard_h, k_shape[-1]}, shard_orientation);
    auto v_spec = tt::tt_metal::ShardSpec(all_cores, {v_shard_h, v_shape[-1]}, shard_orientation);
    // create sharded tensors
    auto mem_config_q = args.output_mem_config.with_shard_spec(q_spec);
    auto mem_config_k = args.output_mem_config.with_shard_spec(k_spec);
    auto mem_config_v = args.output_mem_config.with_shard_spec(v_spec);

    TensorSpec out_tensor_q(
        q_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), mem_config_q));
    TensorSpec out_tensor_k(
        k_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), mem_config_k));
    TensorSpec out_tensor_v(
        v_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), mem_config_v));
    return {out_tensor_q, out_tensor_k, out_tensor_v};
}

tensor_return_value_t CreateQKVHeadsDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_outputs.has_value()) {
        return tensor_args.preallocated_outputs.value();
    }

    auto specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(std::get<0>(specs), tensor_args.input.device()),
        create_device_tensor(std::get<1>(specs), tensor_args.input.device()),
        create_device_tensor(std::get<2>(specs), tensor_args.input.device()),
    };
}

tt::stl::hash::hash_t CreateQKVHeadsDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    operation::Hash hash = operation::hash_operation<CreateQKVHeadsDeviceOperation>(
        args.num_q_heads,
        args.num_kv_heads,
        args.head_dim,
        args.transpose_k_heads,
        args.output_mem_config,
        input_tensor,
        input_tensor.device()->compute_with_storage_grid_size());

    return hash;
}

}  // namespace ttnn::operations::experimental::create_qkv_heads

namespace ttnn::prim {

ttnn::operations::experimental::create_qkv_heads::tensor_return_value_t create_qkv_heads(
    const Tensor& input_tensor,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::tuple<Tensor, Tensor, Tensor>>& preallocated_outputs) {
    using OperationType = ttnn::operations::experimental::create_qkv_heads::CreateQKVHeadsDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .transpose_k_heads = transpose_k_heads,
        .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
    };
    auto tensor_args =
        OperationType::tensor_args_t{.input = input_tensor, .preallocated_outputs = preallocated_outputs};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
