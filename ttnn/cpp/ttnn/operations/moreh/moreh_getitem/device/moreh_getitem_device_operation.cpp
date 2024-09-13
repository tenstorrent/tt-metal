// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_getitem_device_operation.hpp"

#include <cstdint>

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_getitem {
void MorehGetItemOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto input_layout = input_tensor.get_layout();
    const auto& index_tensors = tensor_args.index_tensors;
    const auto& output_tensor = tensor_args.output;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to getitem need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to getitem need to be allocated in buffers on device!");
    auto dtype = input_tensor.get_dtype();
    TT_FATAL(dtype == DataType::INT32 || dtype == DataType::BFLOAT16, "Input tensor must be of type INT32 or BFLOAT16!");

    // validate index tensors
    uint32_t index_size = index_tensors[0].get_shape()[-1];
    for (uint32_t i = 0; i < index_tensors.size(); i++) {
        auto& index_tensor = index_tensors[i];
        TT_FATAL(index_tensor.storage_type() == StorageType::DEVICE, "Operands to getitem need to be on device!");
        TT_FATAL(index_tensor.buffer() != nullptr, "Operands to getitem need to be allocated in buffers on device!");
        TT_FATAL(index_tensor.get_dtype() == DataType::INT32, "Index tensor must be of type INT32!");

        auto index_shape = index_tensor.get_shape();
        auto index_layout = index_tensor.get_layout();
        if (index_layout == Layout::ROW_MAJOR) {
            TT_FATAL(index_shape.rank() == 1, "Index tensor must be 1D for ROW_MAJOR layout!");
        } else if (index_layout == Layout::TILE) {
            TT_FATAL(index_shape.rank() == 5, "Index tensor must be 5D for TILE layout!");
        }
        TT_FATAL(
            !(input_layout == Layout::ROW_MAJOR && index_layout == Layout::TILE),
            "input layout ROW_MAJOR and index layout TILE not supported");
        TT_FATAL(index_size == index_shape[-1], "The shapes of all index tensors must be identical!");
    }

    if (input_layout == Layout::ROW_MAJOR) {
        for (auto dim : operation_attributes.index_dims) {
            TT_FATAL(dim != 4, "getitem for ROW_MAJOR layout not support W index tensor!");
        }
    }

    uint32_t dim_start = operation_attributes.index_dims.front();
    uint32_t i = 0;
    for (auto dim : operation_attributes.index_dims) {
        TT_FATAL(
            dim_start + i == dim,
            "The value of index_dims={} must be consecutive integers.", operation_attributes.index_dims);
        i++;
    }
    if (!output_tensor.has_value()) {
        // If the user decided to not use any optional output tensors, then this would be empty or would be a nullptr.
        return;
    }
    TT_ASSERT(output_tensor->buffer() != nullptr, "Must have 1 output tensor.");
    TT_FATAL(dtype == output_tensor.value().get_dtype(), "Output tensor must have the same dtype as input tensor!");
}
MorehGetItemOperation::program_factory_t MorehGetItemOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto& input_tensor = tensor_args.input;
    auto input_layout = input_tensor.get_layout();
    if (input_layout == Layout::ROW_MAJOR) {
        return MorehGetItemRmFactory();
    } else {
        return MorehGetItemTilizedFactory();
    }
}

void MorehGetItemOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehGetItemOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehGetItemOperation::shape_return_value_t MorehGetItemOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto index_dims = operation_attributes.index_dims;
    auto input_layout = input_tensor.get_layout();
    const auto& index_tensors = tensor_args.index_tensors;
    auto input_shape = input_tensor.get_shape();
    auto output_shape = input_shape;
    auto layout = input_tensor.get_layout();

    if (layout == Layout::TILE) {
        // compute output shape
        // ex)
        // input: (10, 20, 30, 40)
        // index_tensor: [(100), (100)]
        // index_dims = 1,2
        // output: (10, 1, 100, 40)
        auto dim_offset = 5 - input_shape.rank();
        auto dimensions_pads = std::vector<Padding::PadDimension>();
        std::vector<uint32_t> output_size_vec;
        for (int dim = 0; dim < output_shape.size(); dim++) {
            dimensions_pads.push_back(output_shape.value.padding()[dim]);
            output_size_vec.push_back(output_shape.value[dim]);
        }

        auto index = index_tensors[0];
        uint32_t index_size = index.get_shape()[-1];
        uint32_t index_size_without_padding = index.get_shape().value.without_padding()[-1];

        uint32_t last_dim = index_dims.back() + dim_offset;

        for (uint32_t i = 0; i < index_dims.size(); i++) {
            uint32_t out_put_dim = index_dims[i];
            uint32_t dim = out_put_dim + dim_offset;
            auto index = index_tensors[i];

            if (dim == 3 || dim == 4) {
                dimensions_pads[out_put_dim] = Padding::PadDimension{.front = 0, .back = 31};
                output_size_vec[out_put_dim] = 32;
            } else {
                output_size_vec[out_put_dim] = 1;
            }
        }

        if (last_dim == 3 || last_dim == 4) {
            output_size_vec[index_dims.back()] = round_up_to_mul32(index_size);
            uint32_t padding_back = round_up_to_mul32(index_size_without_padding) - index_size_without_padding;
            dimensions_pads[index_dims.back()] = Padding::PadDimension{.front = 0, .back = padding_back};
        } else {
            output_size_vec[index_dims.back()] = index_size_without_padding;
        }

        const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
        output_shape = Shape(tt::tt_metal::Shape(output_size_vec, padding));

    } else {
        // compute output shape
        // ex)
        // input: (10, 20, 30, 40)
        // index_tensor: [(100), (100)]
        // index_dims = 1,2
        // output: (10, 100, 40)
        std::vector<uint32_t> output_size_vec;

        auto input_shape = input_tensor.get_shape();
        uint32_t input_rank = input_shape.rank();

        auto index = index_tensors[0];
        uint32_t index_size = index.get_shape()[0];

        uint32_t start_dim = operation_attributes.index_dims.front();
        uint32_t last_dim = operation_attributes.index_dims.back();
        for (uint32_t input_dim = 0; input_dim < input_rank; input_dim++) {
            if (input_dim < start_dim) {
                output_size_vec.push_back(input_shape[input_dim]);
            } else if (start_dim == input_dim) {
                output_size_vec.push_back(index_size);
            } else if (last_dim < input_dim) {
                output_size_vec.push_back(input_shape[input_dim]);
            }
        }

        output_shape = Shape(output_size_vec);
    }
    return {output_shape};
};

MorehGetItemOperation::tensor_return_value_t MorehGetItemOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        log_debug(tt::LogOp, "{}:{} use output tensor", __func__, __LINE__);
        return {tensor_args.output.value()};
    }
    log_debug(tt::LogOp, "{}:{} create output tensor", __func__, __LINE__);
    const auto& output_shape = compute_output_shapes(operation_attributes, tensor_args);
    return create_device_tensor(
        output_shape,
        tensor_args.input.get_dtype(),
        tensor_args.input.get_layout(),
        tensor_args.input.device(),
        operation_attributes.output_memory_config);
};

std::tuple<MorehGetItemOperation::operation_attributes_t, MorehGetItemOperation::tensor_args_t>
MorehGetItemOperation::invoke(
    const Tensor& input,
    const std::vector<Tensor>& index_tensors,
    const std::vector<uint32_t> index_dims,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig> output_memory_config) {
    operation_attributes_t operation_attributes = {index_dims, output_memory_config.value_or(input.memory_config())};
    tensor_args_t tensor_args = {input, index_tensors, output};
    return {operation_attributes, tensor_args};
}
}  // namespace ttnn::operations::moreh::moreh_getitem
