// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_device_operation.hpp"

#include "ttnn/operations/core/work_split/work_split.hpp"

namespace ttnn::operations::experimental::transformer {

// Generic NLP CreateHeads op
void NlpCreateHeadsDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input_tensor_q;
    const auto input_shape = input_tensor.get_legacy_shape();

    // NOTE: Checks for head_dim and shape[3] is done in nlp_create_qkv_heads because it's needed to infer head_dim
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, fmt::format("Operands to TM need to be on device! {}", input_tensor.storage_type()));
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE);

    TT_FATAL(input_shape[2] % TILE_HEIGHT == 0, fmt::format("Unsupported input height {} is not tile aligned", input_shape[2]));
    TT_FATAL(input_shape[1] == 1, fmt::format("Unsupported input sequence length {} is not equal to 1", input_shape[1]));
    if (input_tensor.is_sharded()) {
        TT_FATAL(input_tensor.shard_spec().value().shape[0] == input_tensor.volume() / input_tensor.get_legacy_shape()[-1]);
        TT_FATAL(operation_attributes.output_mem_config.is_sharded() && operation_attributes.output_mem_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
        TT_FATAL(input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
        auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
        uint32_t num_cores = core_grid.x * core_grid.y;
        // 1 Head Per Core Max for now
        TT_FATAL(operation_attributes.num_q_heads <= num_cores);
        TT_FATAL(operation_attributes.num_kv_heads <= num_cores);
        TT_FATAL(operation_attributes.num_q_heads >= operation_attributes.num_kv_heads);
        TT_FATAL(operation_attributes.num_q_heads % input_tensor.shard_spec().value().num_cores() == 0);
        if (tensor_args.input_tensor_kv.has_value()) {
            TT_FATAL(tensor_args.input_tensor_kv.value().is_sharded());
            TT_FATAL(input_tensor.shard_spec().value().grid == tensor_args.input_tensor_kv.value().shard_spec().value().grid);
            TT_FATAL(input_tensor.shard_spec().value().orientation == tensor_args.input_tensor_kv.value().shard_spec().value().orientation);
            TT_FATAL(input_tensor.shard_spec().value().shape[1] == (operation_attributes.num_q_heads / operation_attributes.num_kv_heads) * operation_attributes.head_dim);
        } else {
            TT_FATAL(operation_attributes.num_kv_heads % input_tensor.shard_spec().value().num_cores() == 0);
            TT_FATAL(input_tensor.shard_spec().value().shape[1] == (operation_attributes.num_q_heads / operation_attributes.num_kv_heads + 2) * operation_attributes.head_dim);
        }
        TT_FATAL(!operation_attributes.transpose_k_heads);
    } else {
        TT_FATAL(operation_attributes.output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }

    if (tensor_args.input_tensor_kv.has_value()) {
        const auto& input_tensor_kv = tensor_args.input_tensor_kv.value();
        const auto input_shape_kv = input_tensor_kv.get_legacy_shape();

        TT_FATAL(input_tensor_kv.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
        TT_FATAL(input_tensor_kv.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
        TT_FATAL(input_tensor_kv.get_dtype() == input_tensor.get_dtype(), "KV tensor dtype must be same as Q tensor dtype!");
        TT_FATAL(input_tensor_kv.get_layout() == Layout::TILE);

        TT_FATAL(input_shape_kv[0] == input_shape[0], "KV tensor batch dim must be same as Q tensor batch!");
        TT_FATAL(input_shape_kv[1] == 1, fmt::format("Unsupported input shape {} is not equal to 1", input_shape_kv[1]));
        TT_FATAL(input_shape_kv[2] == input_shape[2], "KV tensor seq_len dim must be same as Q tensor seq_len!");
        if (input_tensor_kv.is_sharded()) {
            TT_FATAL(input_tensor.is_sharded());
            TT_FATAL(input_tensor_kv.shard_spec().value().shape[0] == input_tensor_kv.volume() / input_tensor_kv.get_legacy_shape()[-1]);
            TT_FATAL(input_tensor_kv.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
            TT_FATAL(input_tensor_kv.shard_spec().value().shape[1] == 2 * operation_attributes.head_dim);
            TT_FATAL(operation_attributes.num_kv_heads % input_tensor_kv.shard_spec().value().num_cores() == 0);
        }
    }
}

void NlpCreateHeadsDeviceOperation::validate_on_program_cache_hit(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return;
}

NlpCreateHeadsDeviceOperation::shape_return_value_t NlpCreateHeadsDeviceOperation::compute_output_shapes(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input_tensor_q;
    const auto input_shape = input_tensor.get_legacy_shape();

    auto sequence_length = input_shape[2];
    auto head_dim = operation_attributes.head_dim;
    if (sequence_length % TILE_HEIGHT != 0) {
        sequence_length = (sequence_length / TILE_HEIGHT + 1) * TILE_HEIGHT;
    }
    if (head_dim % TILE_WIDTH != 0) {
        head_dim = (head_dim / TILE_WIDTH + 1) * TILE_WIDTH;
    }

    const tt::tt_metal::Shape q_output_shape = {input_shape[0], operation_attributes.num_q_heads, sequence_length, head_dim};
    const tt::tt_metal::Shape v_output_shape = {input_shape[0], operation_attributes.num_kv_heads, sequence_length, head_dim};
    const tt::tt_metal::Shape k_output_shape = operation_attributes.transpose_k_heads
                                     ? (tt::tt_metal::Shape){input_shape[0], operation_attributes.num_kv_heads, head_dim, sequence_length}
                                     : v_output_shape;
    return {ttnn::Shape(q_output_shape), ttnn::Shape(k_output_shape), ttnn::Shape(v_output_shape)};

}

NlpCreateHeadsDeviceOperation::tensor_return_value_t NlpCreateHeadsDeviceOperation::create_output_tensors(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input_tensor_q;
    if (tensor_args.optional_output_tensors.size() == 3) {
        const auto& output_tensors = tensor_args.optional_output_tensors;
        return {output_tensors.at(0).value(), output_tensors.at(1).value(), output_tensors.at(2).value()};
    }
    if (operation_attributes.output_mem_config.is_sharded()) {
        auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
        auto q_shard_grid = ttnn::num_cores_to_corerange_set(operation_attributes.num_q_heads, core_grid, true);
        ShardSpec q_shard_spec{q_shard_grid, {TILE_HEIGHT, operation_attributes.head_dim}};
        auto q_mem_config = operation_attributes.output_mem_config;
        q_mem_config.shard_spec = q_shard_spec;
        auto kv_shard_grid = ttnn::num_cores_to_corerange_set(operation_attributes.num_kv_heads, core_grid, true);
        ShardSpec kv_shard_spec{kv_shard_grid, {TILE_HEIGHT, operation_attributes.head_dim}};
        auto kv_mem_config = operation_attributes.output_mem_config;
        kv_mem_config.shard_spec = kv_shard_spec;
        auto output_shapes = compute_output_shapes(operation_attributes, tensor_args);
        return {
            create_device_tensor(std::get<0>(output_shapes), input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), q_mem_config),
            create_device_tensor(std::get<1>(output_shapes), input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), kv_mem_config),
            create_device_tensor(std::get<2>(output_shapes), input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), kv_mem_config)
        };

    } else {
        auto output_shapes = compute_output_shapes(operation_attributes, tensor_args);
        return {
            create_device_tensor(std::get<0>(output_shapes), input_tensor.get_dtype(),Layout::TILE, input_tensor.device(), operation_attributes.output_mem_config),
            create_device_tensor(std::get<1>(output_shapes), input_tensor.get_dtype(),Layout::TILE, input_tensor.device(), operation_attributes.output_mem_config),
            create_device_tensor(std::get<2>(output_shapes), input_tensor.get_dtype(),Layout::TILE, input_tensor.device(), operation_attributes.output_mem_config)
        };
    }
}

NlpCreateHeadsDeviceOperation::program_factory_t NlpCreateHeadsDeviceOperation::select_program_factory(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor_q;
    if (input_tensor.is_sharded()) {
        return Sharded{};
    } else {
        return Interleaved{};
    }
}


std::tuple<NlpCreateHeadsDeviceOperation::operation_attributes_t, NlpCreateHeadsDeviceOperation::tensor_args_t>
NlpCreateHeadsDeviceOperation::invoke(
        const Tensor& input_tensor_q,
        const std::optional<Tensor>& input_tensor_kv,
        const uint32_t num_q_heads,
        const std::optional<uint32_t> num_kv_heads,
        uint32_t head_dim,
        const bool transpose_k_heads,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::vector<std::optional<Tensor>>> optional_output_tensors) {

    return {operation_attributes_t{.num_q_heads = num_q_heads,
                                   .num_kv_heads = num_kv_heads.value_or(num_q_heads),
                                   .head_dim = head_dim,
                                   .transpose_k_heads = transpose_k_heads,
                                   .output_mem_config = memory_config.value_or(input_tensor_q.memory_config())},
            tensor_args_t{.input_tensor_q = input_tensor_q,
                          .input_tensor_kv = input_tensor_kv,
                          .optional_output_tensors = optional_output_tensors.value_or(std::vector<std::optional<Tensor>>{})}};
}

}  // namespace ttnn::operations::experimental::transformer
