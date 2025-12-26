// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"
#include <tt-metalium/work_split.hpp>

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

        // auto shard_shape = input_tensor.shard_spec().value().shape;
        TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "Fold: Expect sharded input tensor in row-major layout.");
        // TT_FATAL(
        //     shard_shape[0] % (stride_h * stride_w) == 0,
        //     "Fold: Shard height must be divisible by stride_h * stride_w.",
        //     shard_shape[0],
        //     stride_h,
        //     stride_w);
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
    validate_fold(
        {tensors.input_tensor}, op_attr.is_sharded, op_attr.is_dram_interleaved, op_attr.stride_h, op_attr.stride_w);

    // Validate padded dimensions are divisible by strides for sharded path
    if (op_attr.is_sharded) {
        const auto& input_shape = tensors.input_tensor.logical_shape();
        uint32_t padded_h = input_shape[1] + op_attr.pad_top + op_attr.pad_bottom;
        uint32_t padded_w = input_shape[2] + op_attr.pad_left + op_attr.pad_right;
        TT_FATAL(
            padded_h % op_attr.stride_h == 0,
            "Fold: Padded height ({}) must be divisible by stride_h ({}).",
            padded_h,
            op_attr.stride_h);
        TT_FATAL(
            padded_w % op_attr.stride_w == 0,
            "Fold: Padded width ({}) must be divisible by stride_w ({}).",
            padded_w,
            op_attr.stride_w);
    }
}

void Fold::validate_on_program_cache_hit(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_fold(
        {tensors.input_tensor}, op_attr.is_sharded, op_attr.is_dram_interleaved, op_attr.stride_h, op_attr.stride_w);
}

Fold::spec_return_value_t Fold::compute_output_specs(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    auto input_tensor = tensors.input_tensor;
    const ttnn::Shape& input_shape = input_tensor.logical_shape();
    auto output_dtype = input_tensor.dtype();
    switch (input_tensor.dtype()) {
        case tt::tt_metal::DataType::FLOAT32: output_dtype = tt::tt_metal::DataType::FLOAT32; break;
        case tt::tt_metal::DataType::UINT16: output_dtype = tt::tt_metal::DataType::UINT16; break;
        default: output_dtype = tt::tt_metal::DataType::BFLOAT16; break;
    }

    // Calculate padded input dimensions
    uint32_t padded_h = input_shape[1] + op_attr.pad_top + op_attr.pad_bottom;
    uint32_t padded_w = input_shape[2] + op_attr.pad_left + op_attr.pad_right;

    // we concatenate (stride_h sticks in H-dim) * (stride_w in W-dim) into 1 stick along C-dim
    ttnn::Shape output_shape(
        {1,
         1,
         input_shape[0] * padded_h * padded_w / (op_attr.stride_h * op_attr.stride_w),
         input_shape[3] * op_attr.stride_h * op_attr.stride_w});

    if (op_attr.is_sharded) {
        auto input_shard_spec = input_tensor.shard_spec().value();

        // Calculate output dimensions with padding
        uint32_t total_output_height = input_shape[0] * padded_h * padded_w / (op_attr.stride_h * op_attr.stride_w);
        uint32_t output_width = input_shape[3] * op_attr.stride_h * op_attr.stride_w;

        // Use max available cores
        auto device = input_tensor.device();
        auto compute_grid = device->compute_with_storage_grid_size();
        uint32_t max_cores = compute_grid.x * compute_grid.y;

        // Use as many cores as possible (up to total_output_height)
        uint32_t output_num_cores = std::min(max_cores, total_output_height);

        // Ceiling division for shard height - last core may have fewer pixels
        uint32_t output_shard_height = (total_output_height + output_num_cores - 1) / output_num_cores;

        // Create output core grid
        CoreRangeSet output_grid = tt::tt_metal::num_cores_to_corerangeset(
            output_num_cores, compute_grid, input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

        tt::tt_metal::ShardSpec output_shard_spec(
            output_grid, {output_shard_height, output_width}, input_shard_spec.orientation);
        auto mem_config = MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, output_shard_spec);

        return {TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), mem_config))};
    } else if (op_attr.is_dram_interleaved) {
        ttnn::Shape output_logical_shape({input_shape[0], input_shape[1], input_shape[2], input_shape[3]});
        if (input_tensor.layout() == Layout::ROW_MAJOR) {
            output_logical_shape = ttnn::Shape(
                {input_shape[0],
                 input_shape[1] / op_attr.stride_h,
                 input_shape[2] / op_attr.stride_w,
                 input_shape[3] * op_attr.stride_h * op_attr.stride_w});
        }
        return {TensorSpec(
            output_logical_shape,
            tt::tt_metal::TensorLayout(
                output_dtype,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                input_tensor.memory_config()))};
    }

    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), input_tensor.memory_config()))};
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
    uint32_t pad_w,
    uint32_t pad_top,
    uint32_t pad_bottom,
    uint32_t pad_left,
    uint32_t pad_right) {
    bool is_sharded = input_tensor.is_sharded();
    bool is_dram_interleaved =
        input_tensor.storage_type() == StorageType::DEVICE && input_tensor.memory_config().is_dram();
    // Use explicit asymmetric padding parameters
    Fold::operation_attributes_t op_attr = {
        .stride_h = stride_h,
        .stride_w = stride_w,
        .is_sharded = is_sharded,
        .is_dram_interleaved = is_dram_interleaved,
        .pad_top = pad_top,
        .pad_bottom = pad_bottom,
        .pad_left = pad_left,
        .pad_right = pad_right};
    return {op_attr, Fold::tensor_args_t{.input_tensor = input_tensor}};
}

}  // namespace ttnn::operations::data_movement
