// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "update_cache_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

using namespace tt::constants;

UpdateKVCacheOperation::program_factory_t UpdateKVCacheOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& /*tensor_args*/) {
    if (args.op_type == UpdateCacheOpType::FILL) {
        return FillCacheMultiCoreProgramFactory{};
    }
    return UpdateCacheMultiCoreProgramFactory{};
}

void UpdateKVCacheOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE and cache_tensor.storage_type() == StorageType::DEVICE,
        "Operands to update_cache need to be on device!");
    TT_FATAL(input_tensor.device() == cache_tensor.device(), "Operands to update_cache need to be on the same device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr and cache_tensor.buffer() != nullptr,
        "Operands to update_cache need to be allocated in buffers on device!");
    TT_FATAL(
        (input_tensor.layout() == Layout::TILE && cache_tensor.layout() == Layout::TILE),
        "Inputs to update_cache must be tilized");
    TT_FATAL(
        input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16 ||
            input_tensor.dtype() == DataType::BFLOAT8_B,
        "Input tensor dtype must be FLOAT32, BFLOAT16, or BFLOAT8_B but got {}",
        input_tensor.dtype());
    TT_FATAL(
        cache_tensor.dtype() == DataType::FLOAT32 || cache_tensor.dtype() == DataType::BFLOAT16 ||
            cache_tensor.dtype() == DataType::BFLOAT8_B,
        "Cache tensor dtype must be FLOAT32, BFLOAT16, or BFLOAT8_B but got {}",
        cache_tensor.dtype());

    TT_FATAL(
        input_tensor.padded_shape()[-1] == cache_tensor.padded_shape()[-1],
        "Input tensor width ({}) must equal cache tensor width ({})",
        input_tensor.padded_shape()[-1],
        cache_tensor.padded_shape()[-1]);
    TT_FATAL(
        input_tensor.padded_shape()[0] == 1,
        "Input tensor batch size must be 1 but got {}",
        input_tensor.padded_shape()[0]);
    TT_FATAL(
        input_tensor.padded_shape()[1] == cache_tensor.padded_shape()[1],
        "Input tensor height ({}) must equal cache tensor height ({})",
        input_tensor.padded_shape()[1],
        cache_tensor.padded_shape()[1]);

    TT_FATAL(
        cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED ||
            (cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
             cache_tensor.memory_config().created_with_nd_shard_spec() &&
             cache_tensor.memory_config()
                 .is_dram()),  // ND_SHARDED layout can collapse to HEIGHT_SHARDED when each bank holds single shard
        "Cache tensor memory layout must be INTERLEAVED, ND_SHARDED, or HEIGHT_SHARDED (created from ND shard spec) "
        "but got {}",
        cache_tensor.memory_config().memory_layout());
    if (args.op_type == UpdateCacheOpType::FILL) {
        // TODO: If we want to support mixed precision like decode, we need to add simple compute kernel for conversion
        TT_FATAL(input_tensor.dtype() == cache_tensor.dtype(), "Input and cache tensors must have same dtype!");

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
            // Require even work division along seq_len and also only 1 head per core
            TT_FATAL(
                input_tensor.padded_shape()[-2] % input_tensor.shard_spec().value().shape[0] == 0,
                "Seq len must be divisible by shard height!");
        }

        TT_FATAL(
            args.batch_idx < cache_tensor.padded_shape()[0],
            "Batch index ({}) must be less than cache tensor batch size ({})",
            args.batch_idx,
            cache_tensor.padded_shape()[0]);
        TT_FATAL(
            args.update_idx % TILE_HEIGHT == 0,
            "Fill update_idx ({}) must be a multiple of TILE_HEIGHT ({})",
            args.update_idx,
            TILE_HEIGHT);
        TT_FATAL(
            args.update_idx <= cache_tensor.padded_shape()[-2] &&
                input_tensor.padded_shape()[-2] <= cache_tensor.padded_shape()[-2] - args.update_idx,
            "Fill update_idx ({}) + input tensor height ({}) must be <= cache tensor height ({})",
            args.update_idx,
            input_tensor.padded_shape()[-2],
            cache_tensor.padded_shape()[-2]);
    } else if (args.op_type == UpdateCacheOpType::UPDATE) {
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
        } else {
            TT_FATAL(
                input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Input tensor memory layout must be INTERLEAVED but got {}",
                input_tensor.memory_config().memory_layout());
        }
        TT_FATAL(
            cache_tensor.padded_shape()[0] <= input_tensor.padded_shape()[-2],
            "Cache tensor batch size ({}) must be <= input tensor height ({})",
            cache_tensor.padded_shape()[0],
            input_tensor.padded_shape()[-2]);
        // batch offset is only valid if num_user less than 32 and batch_offset + num_user <= 32
        if (cache_tensor.padded_shape()[0] < 32) {
            TT_FATAL(
                args.batch_offset + cache_tensor.padded_shape()[0] <= 32,
                "Batch offset ({}) + cache tensor batch size ({}) must be <= 32",
                args.batch_offset,
                cache_tensor.padded_shape()[0]);
        } else {
            TT_FATAL(args.batch_offset == 0, "Batch offset must be 0 when cache tensor batch size >= 32");
        }
    }
}

tt::tt_metal::TensorSpec UpdateKVCacheOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // Do nothing because it's an in-place operation. Cache Tensor is the output tensor.
    return tensor_args.cache.tensor_spec();
}

Tensor UpdateKVCacheOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // Do nothing because it's an in-place operation. Cache Tensor is the output tensor.
    return tensor_args.cache;
}

tt::tt_metal::operation::Hash UpdateKVCacheOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<UpdateKVCacheOperation>(
        args.op_type, std::vector<Tensor>{tensor_args.cache, tensor_args.input});
}

std::vector<tt::tt_metal::DynamicRuntimeArg> UpdateKVCacheOperation::get_dynamic_runtime_args(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The work-split (hence the core set) depends only on shapes, which ARE in the program hash, so
    // the active core set is identical on every cache hit — no freeze hazard from growing cores.
    std::vector<tt::tt_metal::DynamicRuntimeArg> dynamic_args;

    if (operation_attributes.op_type == UpdateCacheOpType::FILL) {
        // FILL: kernel push order in FillCacheMultiCoreProgramFactory::create_descriptor is
        // reader(0), writer(1). Only writer arg 2 (cache_start_id) is derived from the
        // hash-excluded batch_idx/update_idx; the reader args are shape-only.
        constexpr uint32_t kWriterKernelIdx = 1;
        constexpr uint32_t kCacheStartIdArgIdx = 2;
        const auto start_ids = compute_fill_cache_start_ids(operation_attributes, tensor_args);
        dynamic_args.reserve(start_ids.size());
        for (const auto& [core, cache_start_id] : start_ids) {
            dynamic_args.push_back({kWriterKernelIdx, core, kCacheStartIdArgIdx, cache_start_id});
        }
        return dynamic_args;
    }

    // UPDATE: kernel push order in UpdateCacheMultiCoreProgramFactory::create_descriptor is
    // reader(0), writer(1), compute(2), [optional compute(3)]. The hash-excluded values are:
    //   reader arg 8  = cache_start_id (per core)
    //   writer arg 7  = cache_start_id (per core)
    //   writer arg 10 = tile_update_offset (op-wide)
    //   writer arg 11 = batch_read_offset (op-wide)
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;
    constexpr uint32_t kReaderCacheStartIdArgIdx = 8;
    constexpr uint32_t kWriterCacheStartIdArgIdx = 7;
    constexpr uint32_t kWriterTileUpdateOffsetArgIdx = 10;
    constexpr uint32_t kWriterBatchReadOffsetArgIdx = 11;

    const auto dyn = compute_update_cache_dynamic_args(operation_attributes, tensor_args);
    dynamic_args.reserve(dyn.cache_start_ids.size() * 4);
    for (const auto& [core, cache_start_id] : dyn.cache_start_ids) {
        dynamic_args.push_back({kReaderKernelIdx, core, kReaderCacheStartIdArgIdx, cache_start_id});
        dynamic_args.push_back({kWriterKernelIdx, core, kWriterCacheStartIdArgIdx, cache_start_id});
        dynamic_args.push_back({kWriterKernelIdx, core, kWriterTileUpdateOffsetArgIdx, dyn.tile_update_offset});
        dynamic_args.push_back({kWriterKernelIdx, core, kWriterBatchReadOffsetArgIdx, dyn.batch_read_offset});
    }
    return dynamic_args;
}

Tensor update_cache(
    const Tensor& cache,
    const Tensor& input,
    const uint32_t batch_idx,
    const uint32_t update_index,
    const uint32_t batch_offset,
    const UpdateCacheOpType op_type,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::device_operation::launch<UpdateKVCacheOperation>(
        KvCacheParams{
            .batch_idx = batch_idx,
            .update_idx = update_index,
            .batch_offset = batch_offset,
            .op_type = op_type,
            .compute_kernel_config = compute_kernel_config},
        KvCacheInputs{.cache = cache, .input = input});
}

}  // namespace ttnn::prim
