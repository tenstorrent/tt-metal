// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"


namespace ttnn::operations::data_movement {

Fold::program_factory_t Fold::select_program_factory(const operation_attributes_t &op_attr, const tensor_args_t &tensors) {
    if (op_attr.is_sharded) {
        return MultiCore{};
    }
    return SingleCore{};
}

void validate_fold(const std::vector<Tensor> &input_tensors, bool is_sharded, uint32_t stride_h, uint32_t stride_w) {
    const Tensor &input_tensor = input_tensors.at(0);

    const Shape &input_shape = Shape(input_tensor.get_legacy_shape());

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Fold: Expect input tensor to be stored on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "Fold: Expect input tensor to be allocated on a device buffer.");
    TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR, "Fold: Expect input tensor in row-major layout.");
    if (is_sharded) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
            "Fold: Only height-sharded input tensors are supported.");

        auto shard_shape = input_tensor.shard_spec().value().shape;
        TT_FATAL(shard_shape[0] % (input_shape[2] * stride_h * stride_w) == 0);
    } else {
        TT_FATAL(input_shape[1] % stride_h == 0);
        TT_FATAL(input_shape[2] % stride_w == 0);
    }
    TT_FATAL(
        (input_shape[-1] * input_tensor.element_size()) % 16 == 0,
        "Fold: Expect input tensor's pages to be multiples of 16 bytes.");
}

void Fold::validate_on_program_cache_miss(const operation_attributes_t &op_attr, const tensor_args_t &tensors) {
    return validate_fold({tensors.input_tensor}, op_attr.is_sharded, op_attr.stride_h, op_attr.stride_w);
}

void Fold::validate_on_program_cache_hit(const operation_attributes_t &op_attr, const tensor_args_t &tensors) {
    return validate_fold({tensors.input_tensor}, op_attr.is_sharded, op_attr.stride_h, op_attr.stride_w);
}

Fold::shape_return_value_t Fold::compute_output_shapes(const operation_attributes_t &op_attr, const tensor_args_t &tensors) {
    auto input_tensor = tensors.input_tensor;
    const Shape &input_shape = Shape(input_tensor.get_legacy_shape());
    // we concatenate (stride_h sticks in H-dim) * (stride_w in W-dim) into 1 stick along C-dim
    Shape output_shape = Shape(tt::tt_metal::Shape({1, 1, input_shape[0] * input_shape[1] * input_shape[2] / (op_attr.stride_h * op_attr.stride_w), input_shape[3] * op_attr.stride_h * op_attr.stride_w}));
    return output_shape;
}

Fold::tensor_return_value_t Fold::create_output_tensors(const operation_attributes_t &op_attr, const tensor_args_t &tensors) {
    const Tensor &input_tensor = tensors.input_tensor;
    DataType output_dtype = input_tensor.get_dtype();

    auto output_shape = compute_output_shapes(op_attr, tensors);

    if (op_attr.is_sharded) {
        MemoryConfig mem_config = input_tensor.memory_config();
        mem_config.shard_spec->shape[0] /= op_attr.stride_h * op_attr.stride_w;
        mem_config.shard_spec->shape[1] *= op_attr.stride_h * op_attr.stride_w;

        return {create_device_tensor(
            output_shape,
            output_dtype,
            input_tensor.get_layout(),
            input_tensor.device(),
            mem_config
            )};
    } else {
        return {create_device_tensor(output_shape, output_dtype, Layout::ROW_MAJOR, input_tensor.device(), input_tensor.memory_config())};
    }
}


} // namespace ttnn::operations::data_movement
