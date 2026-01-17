// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

using namespace tt::constants;

namespace ttnn::operations::experimental::transformer::rotary_embedding {

RotaryEmbeddingDeviceOperation::program_factory_t RotaryEmbeddingDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return rotary_embedding::program::RotaryEmbeddingProgramFactory{};
}

void RotaryEmbeddingDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void RotaryEmbeddingDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;

    auto* ref_device = input_tensor.device();
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to rotary embedding need to be on device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr, "Operands to rotary embedding need to be allocated in buffers on device!");
    TT_FATAL(cos.storage_type() == StorageType::DEVICE, "Operands to rotary embedding need to be on device!");
    TT_FATAL(cos.buffer() != nullptr, "Operands to rotary embedding need to be allocated in buffers on device!");
    TT_FATAL(sin.storage_type() == StorageType::DEVICE, "Operands to rotary embedding need to be on device!");
    TT_FATAL(sin.buffer() != nullptr, "Operands to rotary embedding need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.device() == ref_device, "Operands to rotary embedding need to be on same device!");
    TT_FATAL(cos.device() == ref_device, "Operands to rotary embedding need to be on same device!");
    TT_FATAL(sin.device() == ref_device, "Operands to rotary embedding need to be on same device!");
    TT_FATAL((input_tensor.layout() == Layout::TILE), "Inputs to rotary embedding must be tilized");
    TT_FATAL((cos.layout() == Layout::TILE), "Inputs to rotary embedding must be tilized");
    TT_FATAL((sin.layout() == Layout::TILE), "Inputs to rotary embedding must be tilized");

    TT_FATAL(input_tensor.padded_shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible into tiles");
    uint32_t seq_len = input_tensor.padded_shape()[-2];
    uint32_t X = input_tensor.padded_shape()[-1];
    TT_FATAL(cos.dtype() == sin.dtype(), "Cos and Sin dtypes must match");
    TT_FATAL(cos.padded_shape() == sin.padded_shape(), "Cos and Sin dims must match");
    TT_FATAL(
        cos.padded_shape()[0] == 1 && cos.padded_shape()[1] == 1 && cos.padded_shape()[-1] == X,
        "Cos dims must match input dims");
    if (args.token_idx.has_value()) {
        TT_FATAL(cos.padded_shape()[-2] >= args.token_idx.value(), "Cos dims must match input dims");
    } else {
        TT_FATAL(cos.padded_shape()[-2] >= seq_len, "Cos dims must match input dims");
    }
    if (input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
            "Input tensor memory layout must not be WIDTH_SHARDED but got {}",
            input_tensor.memory_config().memory_layout());
        TT_FATAL(
            input_tensor.shard_spec().value().shape[1] == input_tensor.padded_shape()[-1],
            "Input tensor shard width ({}) must equal padded width ({})",
            input_tensor.shard_spec().value().shape[1],
            input_tensor.padded_shape()[-1]);
        // Require even work division for now
        TT_FATAL(
            (input_tensor.physical_volume() / input_tensor.padded_shape()[-1]) %
                    input_tensor.shard_spec().value().shape[0] ==
                0,
            "Error");
        if (args.output_mem_config.is_sharded()) {
            TT_FATAL(
                args.output_mem_config.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                "Output memory config layout must not be WIDTH_SHARDED but got {}",
                args.output_mem_config.memory_layout());
        }
    } else if (args.output_mem_config.is_sharded()) {
        TT_FATAL(
            args.output_mem_config.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
            "Output memory config layout must not be WIDTH_SHARDED but got {}",
            args.output_mem_config.memory_layout());
    } else {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Input tensor memory layout must be INTERLEAVED but got {}",
            input_tensor.memory_config().memory_layout());
        TT_FATAL(
            args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Output memory config layout must be INTERLEAVED but got {}",
            args.output_mem_config.memory_layout());
    }
}

spec_return_value_t RotaryEmbeddingDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto shape = input_tensor.padded_shape();
    if (!args.token_idx.has_value()) {
        shape[-2] = tt::round_up(args.seq_len, TILE_HEIGHT);
    }

    if (args.output_mem_config.is_sharded()) {
        tt::tt_metal::ShardSpec shard_spec{CoreRangeSet(), {0, 0}};
        if (input_tensor.is_sharded()) {
            shard_spec = input_tensor.shard_spec().value();
        } else {
            uint32_t num_blocks = input_tensor.physical_volume() / input_tensor.padded_shape()[-1] / TILE_HEIGHT;
            auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
            uint32_t num_grid_cores = core_grid.x * core_grid.y;
            uint32_t num_cores = 0;
            for (uint32_t i = num_grid_cores; i > 0; --i) {
                if (num_blocks % i == 0) {
                    num_cores = i;
                    break;
                }
            }
            uint32_t Ht = tt::div_up(num_blocks, num_cores);
            shard_spec.grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, core_grid, true);
            shard_spec.shape = {Ht * TILE_HEIGHT, input_tensor.padded_shape()[-1]};
            shard_spec.orientation = ShardOrientation::ROW_MAJOR;
        }
        auto mem_config = args.output_mem_config.with_shard_spec(shard_spec);
        return TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), mem_config));
    }

    return TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), args.output_mem_config));
}

tensor_return_value_t RotaryEmbeddingDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t RotaryEmbeddingDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto program_factory = select_program_factory(args, tensor_args);
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<RotaryEmbeddingDeviceOperation>(
        args.seq_len,
        args.output_mem_config,
        program_factory.index(),
        tensor_args.input,
        tensor_args.cos,
        tensor_args.sin);
    return hash;
}

}  // namespace ttnn::operations::experimental::transformer::rotary_embedding

namespace ttnn::prim {

ttnn::operations::experimental::transformer::rotary_embedding::tensor_return_value_t rotary_embedding(
    const Tensor& input,
    const Tensor& cos,
    const Tensor& sin,
    uint32_t seq_len,
    std::optional<uint32_t> token_idx,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::transformer::rotary_embedding::RotaryEmbeddingDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .seq_len = seq_len,
        .token_idx = token_idx,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input, .cos = cos, .sin = sin};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
