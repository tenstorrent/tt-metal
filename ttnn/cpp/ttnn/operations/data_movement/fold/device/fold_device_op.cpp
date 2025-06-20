// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"

namespace ttnn::operations::data_movement {

Fold::program_factory_t Fold::select_program_factory(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    if (op_attr.is_sharded) {
        return MultiCore{};
    } else if (op_attr.is_dram_interleaved) {
        return MultiCoreDRAMFold{};
    }
    return SingleCore{};
}

void validate_fold(
    const std::vector<Tensor>& input_tensors,
    bool is_sharded,
    bool is_dram_interleaved,
    uint32_t stride_h,
    uint32_t stride_w) {
    const Tensor& input_tensor = input_tensors.at(0);

    const auto& input_shape = input_tensor.padded_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Fold: Expect input tensor to be stored on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "Fold: Expect input tensor to be allocated on a device buffer.");
    if (is_sharded) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Fold: Only height-sharded input tensors are supported.");

        auto shard_shape = input_tensor.shard_spec().value().shape;
        TT_FATAL(shard_shape[0] % (input_shape[2] * stride_h * stride_w) == 0, "Error");
        TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "Fold: Expect sharded input tensor in row-major layout.");
        TT_FATAL(
            (input_shape[-1] * input_tensor.element_size()) % 16 == 0,
            "Fold: Expect input tensor's pages to be multiples of 16 bytes.");
    } else if (is_dram_interleaved) {
        TT_FATAL(input_shape[1] % stride_h == 0, "Fold: Input height must be divisible by stride_h.");
        TT_FATAL(input_shape[2] % stride_w == 0, "Fold: Input width must be divisible by stride_w.");
    } else {
        TT_FATAL(input_shape[1] % stride_h == 0, "Fold: Input height must be divisible by stride_h.");
        TT_FATAL(input_shape[2] % stride_w == 0, "Fold: Input width must be divisible by stride_w.");
        TT_FATAL(
            (input_shape[-1] * input_tensor.element_size()) % 16 == 0,
            "Fold: Expect input tensor's pages to be multiples of 16 bytes.");
    }
}

void Fold::validate_on_program_cache_miss(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return validate_fold(
        {tensors.input_tensor}, op_attr.is_sharded, op_attr.is_dram_interleaved, op_attr.stride_h, op_attr.stride_w);
}

void Fold::validate_on_program_cache_hit(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return validate_fold(
        {tensors.input_tensor}, op_attr.is_sharded, op_attr.is_dram_interleaved, op_attr.stride_h, op_attr.stride_w);
}

Fold::spec_return_value_t Fold::compute_output_specs(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    auto input_tensor = tensors.input_tensor;
    const ttnn::Shape& input_shape = input_tensor.logical_shape();
    // we concatenate (stride_h sticks in H-dim) * (stride_w in W-dim) into 1 stick along C-dim
    ttnn::Shape output_shape(
        {1,
         1,
         input_shape[0] * input_shape[1] * input_shape[2] / (op_attr.stride_h * op_attr.stride_w),
         input_shape[3] * op_attr.stride_h * op_attr.stride_w});

    if (op_attr.is_sharded) {
        auto shard_spec = input_tensor.shard_spec().value();
        shard_spec.shape[0] /= op_attr.stride_h * op_attr.stride_w;
        shard_spec.shape[1] *= op_attr.stride_h * op_attr.stride_w;
        auto mem_config = input_tensor.memory_config().with_shard_spec(shard_spec);

        return {TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), mem_config))};
    } else if (op_attr.is_dram_interleaved) {
        ttnn::Shape output_logical_shape({input_shape[0], input_shape[1], input_shape[2], input_shape[3]});
        return {TensorSpec(
            output_logical_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(),
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                input_tensor.memory_config()))};
    }

    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), input_tensor.memory_config()))};
}

Fold::tensor_return_value_t Fold::create_output_tensors(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return create_device_tensor(compute_output_specs(op_attr, tensors), tensors.input_tensor.device());
}

std::tuple<Fold::operation_attributes_t, Fold::tensor_args_t> Fold::invoke(
    const ttnn::Tensor& input_tensor,
    uint32_t stride_h,
    uint32_t stride_w,
    const std::optional<const ttnn::Shape>& output_shape,
    uint32_t pad_c,
    uint32_t pad_h,
    uint32_t pad_w) {
    bool is_sharded = input_tensor.is_sharded();
    bool is_dram_interleaved =
        input_tensor.storage_type() == StorageType::DEVICE && input_tensor.memory_config().is_dram();
    Fold::operation_attributes_t op_attr = {
        .stride_h = stride_h,
        .stride_w = stride_w,
        .is_sharded = is_sharded,
        .is_dram_interleaved = is_dram_interleaved};
    return {op_attr, Fold::tensor_args_t{.input_tensor = input_tensor}};
}

}  // namespace ttnn::operations::data_movement
