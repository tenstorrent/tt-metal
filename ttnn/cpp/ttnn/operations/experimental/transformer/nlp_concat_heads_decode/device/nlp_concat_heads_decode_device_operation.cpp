// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_decode_device_operation.hpp"
#include <algorithm>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::nlp_concat_heads_decode {

NLPConcatHeadsDecodeDeviceOperation::program_factory_t NLPConcatHeadsDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& /*tensor_args*/) {
    if (args.on_subcoregrids) {
        return program::NLPConcatHeadsDecodeSubcoregridsProgramFactory{};
    }
    return program::NLPConcatHeadsDecodeProgramFactory{};
}

void NLPConcatHeadsDecodeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void NLPConcatHeadsDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();

    // input tensor and shape
    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "Unsupported data format");
    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::TILE,
        "Input tensor layout must be TILE but got {}",
        input_tensor.layout());
    TT_FATAL(input_shape[0] == 1, "seqlen=1 for decode");
    TT_FATAL(input_shape[1] <= 32, "currently only support less than 32 users");
    TT_FATAL(input_shape[2] == 32, "currently only support 32 padded heads");
    TT_FATAL(input_shape[2] >= args.num_heads, "head_dim must be multiple of TILE_WIDTH");

    // input tensor shard spec
    TT_FATAL(input_tensor.is_sharded(), "Input tensor must be sharded");
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        "Input tensor memory layout must be HEIGHT_SHARDED but got {}",
        input_tensor.memory_config().memory_layout());
    auto shard_spec = input_tensor.shard_spec().value();
    TT_FATAL(
        shard_spec.shape[1] == input_tensor.padded_shape()[-1],
        "Shard spec shape[1] ({}) must match input tensor padded shape[-1] ({})",
        shard_spec.shape[1],
        input_tensor.padded_shape()[-1]);
    TT_FATAL(
        shard_spec.shape[0] == input_tensor.padded_shape()[-2],
        "Shard spec shape[0] ({}) must match input tensor padded shape[-2] ({})",
        shard_spec.shape[0],
        input_tensor.padded_shape()[-2]);
    auto num_cores = shard_spec.grid.num_cores();
    if (args.on_subcoregrids) {
        TT_FATAL(num_cores == input_shape[1], "Input core grid num_cores must be equal to num users");
        TT_FATAL(args.sub_core_grids.has_value(), "Subcoregrids must be provided if on_subcoregrids is true");
        TT_FATAL(
            args.sub_core_grids.value().num_cores() >= args.num_heads,
            "Subcoregrids must have at least num_heads cores");
    } else {
        TT_FATAL(num_cores == input_shape[1], "Input core grid num_cores must be equal to num users");
    }
}

spec_return_value_t NLPConcatHeadsDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();

    auto num_heads = args.num_heads;
    auto sequence_length = input_shape[0];
    auto batch = input_shape[1];
    auto head_dim = input_shape[3];
    // pad batch to 32 if necessary
    batch = std::max<uint32_t>(batch, 32);

    auto hidden_dim = num_heads * head_dim;

    Shape output_shape({sequence_length, 1, batch, hidden_dim});

    CoreRangeSet output_core_grid;
    if (args.on_subcoregrids) {
        const auto input_core_ranges = input_tensor.shard_spec().value().grid.ranges();
        CoreRangeSet input_core_grid = input_tensor.shard_spec().value().grid;
        const auto start_coord = input_core_ranges[0].start_coord;
        const auto& sub_core_grids = args.sub_core_grids;
        output_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
            start_coord, num_heads, sub_core_grids.value(), true);
    } else {
        output_core_grid = tt::tt_metal::num_cores_to_corerangeset(
            num_heads, input_tensor.device()->compute_with_storage_grid_size(), true);
    }

    tt::tt_metal::ShardSpec shard_spec{output_core_grid, {batch, head_dim}};
    auto mem_config = tt::tt_metal::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, tt::tt_metal::BufferType::L1, shard_spec};

    return TensorSpec(
        output_shape, tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::Layout::TILE, mem_config));
}

tensor_return_value_t NLPConcatHeadsDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::experimental::nlp_concat_heads_decode

namespace ttnn::prim {

ttnn::operations::experimental::nlp_concat_heads_decode::tensor_return_value_t nlp_concat_heads_decode(
    const Tensor& input_tensor,
    uint32_t num_heads,
    const std::optional<MemoryConfig>& /*memory_config*/,
    const std::optional<Tensor>& preallocated_output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::experimental::nlp_concat_heads_decode::NLPConcatHeadsDecodeDeviceOperation;

    bool on_subcoregrids = false;
    if (input_tensor.is_sharded()) {
        const auto& input_core_ranges = input_tensor.shard_spec().value().grid.ranges();
        if (input_core_ranges.size() > 1 || !(input_core_ranges[0].start_coord == CoreCoord{0, 0}) ||
            sub_core_grids.has_value()) {
            on_subcoregrids = true;
        }
    }

    auto operation_attributes = OperationType::operation_attributes_t{
        .num_heads = num_heads,
        .on_subcoregrids = on_subcoregrids,
        .sub_core_grids = sub_core_grids,
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input_tensor, .preallocated_output = preallocated_output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
