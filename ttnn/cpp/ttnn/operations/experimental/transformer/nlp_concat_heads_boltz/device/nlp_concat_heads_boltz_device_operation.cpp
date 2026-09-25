// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_boltz_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {
namespace {

// Head occupies padded[1]*padded[2] rows. Also called from compute_output_specs (before validate).
uint32_t boltz_heads_per_shard(const Tensor& input_tensor) {
    const auto shard_spec = input_tensor.shard_spec().value();
    const auto& padded_shape = input_tensor.padded_shape();
    const uint32_t rows_per_head = padded_shape[1] * padded_shape[2];
    TT_FATAL(
        shard_spec.shape[1] == padded_shape[-1],
        "Input tensor shard width ({}) must equal padded width ({})",
        shard_spec.shape[1],
        padded_shape[-1]);
    TT_FATAL(
        rows_per_head > 0 && shard_spec.shape[0] % rows_per_head == 0,
        "Input tensor shard height ({}) must be divisible by rows per head (padded[1] * padded[2] = {} * {} = {})",
        shard_spec.shape[0],
        padded_shape[1],
        padded_shape[2],
        rows_per_head);
    const uint32_t heads_per_shard = shard_spec.shape[0] / rows_per_head;
    TT_FATAL(
        heads_per_shard > 0 && padded_shape[0] % heads_per_shard == 0,
        "Input tensor num_heads ({}) must be divisible by heads per shard (shard height / rows per head = {} / "
        "{} = {})",
        padded_shape[0],
        shard_spec.shape[0],
        rows_per_head,
        heads_per_shard);
    return heads_per_shard;
}

}  // namespace

void NLPConcatHeadsBoltzDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    TT_FATAL(input_tensor.storage_type() == ttnn::StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::TILE,
        "Input tensor layout must be TILE but got {}",
        input_tensor.layout());
    if (input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() != tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
            "Input tensor memory layout must not be WIDTH_SHARDED but got {}",
            input_tensor.memory_config().memory_layout());
        boltz_heads_per_shard(input_tensor);
        TT_FATAL(
            args.output_mem_config.memory_layout() != tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            "Output memory config layout must not be HEIGHT_SHARDED but got {}",
            args.output_mem_config.memory_layout());
        // The sharded kernel writes directly into the output shard; an interleaved output has no
        // shard to write to (legacy silently produced garbage through an unconfigured CB here).
        TT_FATAL(
            args.output_mem_config.is_sharded(),
            "Sharded input requires a sharded output memory config but got {}",
            args.output_mem_config.memory_layout());
    } else {
        TT_FATAL(
            args.output_mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "Output memory config layout must be INTERLEAVED but got {}",
            args.output_mem_config.memory_layout());
    }

    if (tensor_args.preallocated_output.has_value()) {
        const auto computed_output_spec = compute_output_specs(args, tensor_args);
        const auto& preallocated_output = tensor_args.preallocated_output.value();
        TT_FATAL(
            preallocated_output.logical_shape() == computed_output_spec.logical_shape(),
            "Preallocated output shape must match computed output shape");
    }
}

NLPConcatHeadsBoltzDeviceOperation::spec_return_value_t NLPConcatHeadsBoltzDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.logical_shape();

    auto num_heads = input_shape[0];
    auto head_dim = input_shape[3];

    auto hidden_dim = num_heads * head_dim;

    Shape output_shape({1, input_shape[1], input_shape[2], hidden_dim});

    if (args.output_mem_config.is_sharded()) {
        tt::tt_metal::ShardSpec shard_spec = input_tensor.shard_spec().value();
        const uint32_t heads_per_shard = boltz_heads_per_shard(input_tensor);
        shard_spec.shape = {shard_spec.shape[0] / heads_per_shard, shard_spec.shape[1] * heads_per_shard};
        auto mem_config = tt::tt_metal::MemoryConfig(
            args.output_mem_config.memory_layout(), args.output_mem_config.buffer_type(), shard_spec);
        return tt::tt_metal::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));
    }

    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), args.output_mem_config));
}

NLPConcatHeadsBoltzDeviceOperation::tensor_return_value_t NLPConcatHeadsBoltzDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor nlp_concat_heads_boltz(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    using OperationType = ttnn::experimental::prim::NLPConcatHeadsBoltzDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{memory_config};
    auto tensor_args = OperationType::tensor_args_t{input_tensor, std::move(optional_output_tensor)};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
