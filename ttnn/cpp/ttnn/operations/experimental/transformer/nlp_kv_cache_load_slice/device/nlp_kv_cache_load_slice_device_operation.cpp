// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_kv_cache_load_slice_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::transformer::nlp_kv_cache_load_slice {

NlpKVCacheLoadSliceDeviceOperation::program_factory_t NlpKVCacheLoadSliceDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::NlpKVCacheLoadSliceProgramFactory{};
}

void NlpKVCacheLoadSliceDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void NlpKVCacheLoadSliceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor_a = tensor_args.input;
    TT_FATAL(
        input_tensor_a.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.layout() == tt::tt_metal::Layout::TILE,
        "Input tensor layout must be TILE but got {}",
        input_tensor_a.layout());

    for (uint32_t i = 0; i < input_tensor_a.padded_shape().rank(); i++) {
        TT_FATAL(
            args.output_tensor_start[i] < input_tensor_a.padded_shape()[i],
            "Output tensor start[{}] ({}) must be less than input tensor shape[{}] ({})",
            i,
            args.output_tensor_start[i],
            i,
            input_tensor_a.padded_shape()[i]);
        TT_FATAL(
            args.output_tensor_end[i] < input_tensor_a.padded_shape()[i],
            "Output tensor end[{}] ({}) must be less than input tensor shape[{}] ({})",
            i,
            args.output_tensor_end[i],
            i,
            input_tensor_a.padded_shape()[i]);

        // Check if start shape is <= end shape
        TT_FATAL(
            args.output_tensor_start[i] <= args.output_tensor_end[i],
            "Output tensor start[{}] ({}) must be <= output tensor end[{}] ({})",
            i,
            args.output_tensor_start[i],
            i,
            args.output_tensor_end[i]);
    }

    Shape output_tensor_shape = compute_output_specs(args, tensor_args).padded_shape();
    auto num_dims = input_tensor_a.padded_shape().rank();
    TT_FATAL(num_dims == 4, "Input tensor must be 4D");
    const auto& input_shape = input_tensor_a.padded_shape();
    auto dim0 = input_shape[0];
    auto dim1 = input_shape[1];
    auto fused_batch_heads = dim0 * dim1;
    auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
    // Need at least fused_batch_heads cores to unpad into sharded tensor
    TT_FATAL(
        fused_batch_heads <= core_grid.x * core_grid.y,
        "Fused batch heads ({}) must be <= total grid size ({})",
        fused_batch_heads,
        core_grid.x * core_grid.y);
    TT_FATAL(
        input_tensor_a.physical_volume() % TILE_HW == 0,
        "Input tensor physical volume ({}) must be divisible by TILE_HW ({})",
        input_tensor_a.physical_volume(),
        TILE_HW);
    TT_FATAL(
        (output_tensor_shape[-2] % TILE_HEIGHT == 0) && (args.output_tensor_start[-2] % TILE_HEIGHT == 0),
        "Can only unpad tilized tensor with full tiles");
    TT_FATAL(
        (output_tensor_shape[-1] % TILE_WIDTH == 0) && (args.output_tensor_start[-1] % TILE_WIDTH == 0),
        "Can only unpad tilized tensor with full tiles");
}

NlpKVCacheLoadSliceDeviceOperation::spec_return_value_t NlpKVCacheLoadSliceDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input;
    const auto& input_shape = input_tensor_a.padded_shape();

    SmallVector<uint32_t> out_shape;
    auto rank = input_shape.rank();
    out_shape.reserve(rank);
    for (uint32_t i = 0; i < rank; i++) {
        out_shape.push_back(args.output_tensor_end[i] - args.output_tensor_start[i] + 1);
    }

    auto dim0 = input_shape[0];
    auto dim1 = input_shape[1];
    auto unpad_length = args.output_tensor_end[2] - args.output_tensor_start[2] + 1;
    auto head_dim = input_shape[3];
    auto fused_batch_heads = dim0 * dim1;

    auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
    auto shard_grid = tt::tt_metal::num_cores_to_corerangeset(fused_batch_heads, core_grid, true);
    tt::tt_metal::ShardSpec shard_spec{shard_grid, {unpad_length, head_dim}};
    auto mem_config = tt::tt_metal::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1, shard_spec};

    return TensorSpec(
        Shape(out_shape),
        tt::tt_metal::TensorLayout(
            input_tensor_a.dtype(), tt::tt_metal::PageConfig(input_tensor_a.layout()), mem_config));
}

NlpKVCacheLoadSliceDeviceOperation::tensor_return_value_t NlpKVCacheLoadSliceDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::experimental::transformer::nlp_kv_cache_load_slice

namespace ttnn::prim {

ttnn::operations::experimental::transformer::nlp_kv_cache_load_slice::tensor_return_value_t nlp_kv_cache_load_slice(
    const Tensor& input_tensor,
    uint32_t seq_len_start,
    uint32_t seq_len_end,
    [[maybe_unused]] const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType =
        ttnn::operations::experimental::transformer::nlp_kv_cache_load_slice::NlpKVCacheLoadSliceDeviceOperation;

    auto input_tensor_shape = input_tensor.padded_shape();
    auto dim0 = input_tensor_shape[0];
    auto dim1 = input_tensor_shape[1];
    auto head_dim = input_tensor_shape[3];

    ttnn::Shape output_tensor_start({
        0,
        0,
        seq_len_start,
        0,
    });

    ttnn::Shape output_tensor_end({
        dim0 - 1,
        dim1 - 1,
        seq_len_end - 1,
        head_dim - 1,
    });

    auto operation_attributes = OperationType::operation_attributes_t{output_tensor_start, output_tensor_end};
    auto tensor_args = OperationType::tensor_args_t{input_tensor, preallocated_output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
