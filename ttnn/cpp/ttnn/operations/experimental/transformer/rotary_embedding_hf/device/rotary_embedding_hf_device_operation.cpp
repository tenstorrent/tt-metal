// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_hf_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

RotaryEmbeddingHfDeviceOperation::program_factory_t RotaryEmbeddingHfDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t&) {
    if (args.is_decode_mode) {
        return RotaryEmbeddingHfMultiCoreSharded{};
    }
    return RotaryEmbeddingHfMultiCore{};
}

void RotaryEmbeddingHfDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;

    auto* ref_device = input_tensor.device();
    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Input must be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input must be allocated in buffer on device!");
    TT_FATAL(cos.storage_type() == tt::tt_metal::StorageType::DEVICE, "Cos must be on device!");
    TT_FATAL(cos.buffer() != nullptr, "Cos must be allocated in buffer on device!");
    TT_FATAL(sin.storage_type() == tt::tt_metal::StorageType::DEVICE, "Sin must be on device!");
    TT_FATAL(sin.buffer() != nullptr, "Sin must be allocated in buffer on device!");
    TT_FATAL(input_tensor.device() == ref_device, "All tensors must be on same device!");
    TT_FATAL(cos.device() == ref_device, "All tensors must be on same device!");
    TT_FATAL(sin.device() == ref_device, "All tensors must be on same device!");
    TT_FATAL(input_tensor.layout() == tt::tt_metal::Layout::TILE, "Input must be tilized");
    TT_FATAL(cos.layout() == tt::tt_metal::Layout::TILE, "Cos must be tilized");
    TT_FATAL(sin.layout() == tt::tt_metal::Layout::TILE, "Sin must be tilized");

    TT_FATAL(input_tensor.padded_shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible by 64");
    uint32_t X = input_tensor.padded_shape()[-1];
    TT_FATAL(cos.dtype() == sin.dtype(), "Cos and Sin dtypes must match");
    TT_FATAL(cos.padded_shape() == sin.padded_shape(), "Cos and Sin shapes must match");
    TT_FATAL(cos.padded_shape()[0] == 1 && cos.padded_shape()[-1] == X, "Cos dims must match input dims");

    if (args.is_decode_mode) {
        // Decode mode: input [1, batch, num_heads, head_dim], cos/sin [1, batch, 1, head_dim]
        TT_FATAL(input_tensor.is_sharded(), "Input must be sharded in decode mode");
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            "Input must be HEIGHT_SHARDED in decode mode");
        TT_FATAL(cos.is_sharded(), "Cos must be sharded in decode mode");
        TT_FATAL(sin.is_sharded(), "Sin must be sharded in decode mode");

        uint32_t batch_size = input_tensor.padded_shape()[1];
        TT_FATAL(cos.padded_shape()[1] == batch_size, "Cos batch dim must match input");
        TT_FATAL(sin.padded_shape()[1] == batch_size, "Sin batch dim must match input");
        TT_FATAL(cos.padded_shape()[2] == 1, "Cos seq_len must be 1 in decode mode");
        TT_FATAL(sin.padded_shape()[2] == 1, "Sin seq_len must be 1 in decode mode");
    } else {
        // Prefill mode: input [1, num_heads, seq_len, head_dim], cos/sin [1, 1, seq_len, head_dim]
        uint32_t seq_len = input_tensor.padded_shape()[-2];
        TT_FATAL(cos.padded_shape()[1] == 1, "Cos must have batch dim = 1 in prefill mode");
        TT_FATAL(sin.padded_shape()[1] == 1, "Sin must have batch dim = 1 in prefill mode");
        TT_FATAL(cos.padded_shape()[-2] >= seq_len, "Cos seq_len must be >= input seq_len");
        TT_FATAL(sin.padded_shape()[-2] >= seq_len, "Sin seq_len must be >= input seq_len");

        if (input_tensor.is_sharded()) {
            TT_FATAL(
                input_tensor.memory_config().memory_layout() != tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                "Input cannot be WIDTH_SHARDED");
        }
    }

    if (args.output_mem_config.is_sharded()) {
        TT_FATAL(
            args.output_mem_config.memory_layout() != tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
            "Output cannot be WIDTH_SHARDED");
    }
}

tt::tt_metal::TensorSpec RotaryEmbeddingHfDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto shape = input_tensor.padded_shape();

    if (args.output_mem_config.is_sharded()) {
        tt::tt_metal::ShardSpec shard_spec{tt::tt_metal::CoreRangeSet(), {0, 0}};
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
            shard_spec.orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR;
        }
        auto mem_config = args.output_mem_config.with_shard_spec(shard_spec);
        return tt::tt_metal::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), mem_config));
    }

    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), args.output_mem_config));
}

tt::tt_metal::Tensor RotaryEmbeddingHfDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::create_device_tensor(
        compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t RotaryEmbeddingHfDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto program_factory = select_program_factory(args, tensor_args);
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<RotaryEmbeddingHfDeviceOperation>(
        args.is_decode_mode,
        args.output_mem_config,
        program_factory.index(),
        tensor_args.input_tensor,
        tensor_args.cos_cache,
        tensor_args.sin_cache);
    return hash;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

tt::tt_metal::Tensor rotary_embedding_hf(
    const tt::tt_metal::Tensor& input,
    const tt::tt_metal::Tensor& cos,
    const tt::tt_metal::Tensor& sin,
    bool is_decode_mode,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::RotaryEmbeddingHfDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .is_decode_mode = is_decode_mode,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input, .cos_cache = cos, .sin_cache = sin};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
