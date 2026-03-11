// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_bincount_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

void MaskedBincountDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& input_tensor) {
    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::UINT16, "Only UINT16 is supported for input!");
    TT_FATAL(input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for input!");
    const auto& input_shape = input_tensor.padded_shape();
    TT_FATAL(
        input_shape.size() == 2, "Input tensor must be 2D [sp_dim, topk_dim], got {} dimensions", input_shape.size());
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        "Input tensor must be height sharded!");
    TT_FATAL(input_tensor.shard_spec().has_value(), "Input tensor must have a shard spec!");
    TT_FATAL(args.n_routed_experts > 0, "n_routed_experts must be > 0");
}

MaskedBincountDeviceOperation::spec_return_value_t MaskedBincountDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& /*input_tensor*/) {
    ttnn::Shape output_shape({args.n_routed_experts});
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

tt::stl::hash::hash_t MaskedBincountDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& input_tensor) {
    const auto& input_shape = input_tensor.padded_shape();
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<MaskedBincountDeviceOperation>(
        args, input_tensor.dtype(), input_tensor.memory_config(), input_shape);
    return hash;
}

MaskedBincountDeviceOperation::tensor_return_value_t MaskedBincountDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& input_tensor) {
    return create_device_tensor(compute_output_specs(args, input_tensor), input_tensor.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor masked_bincount(const Tensor& input_tensor, uint32_t n_routed_experts) {
    using OperationType = ttnn::experimental::prim::MaskedBincountDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{.n_routed_experts = n_routed_experts};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, input_tensor);
}

}  // namespace ttnn::prim
