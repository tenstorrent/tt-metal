// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_getitem_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <cstdint>

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_getitem {
void MorehGetItemOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto input_layout = input_tensor.layout();
    const auto& index_tensors = tensor_args.index_tensors;
    const auto& output_tensor = tensor_args.output;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to getitem need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to getitem need to be allocated in buffers on device!");
    auto dtype = input_tensor.dtype();
    TT_FATAL(
        dtype == DataType::INT32 || dtype == DataType::BFLOAT16, "Input tensor must be of type INT32 or BFLOAT16!");

    // validate index tensors
    uint32_t index_size = index_tensors[0].logical_shape()[-1];
    for (const auto& index_tensor : index_tensors) {
        TT_FATAL(index_tensor.storage_type() == StorageType::DEVICE, "Operands to getitem need to be on device!");
        TT_FATAL(index_tensor.buffer() != nullptr, "Operands to getitem need to be allocated in buffers on device!");
        TT_FATAL(index_tensor.dtype() == DataType::INT32, "Index tensor must be of type INT32!");

        auto index_shape = index_tensor.logical_shape();
        auto index_layout = index_tensor.layout();
        if (index_layout == Layout::ROW_MAJOR) {
            TT_FATAL(index_shape.rank() == 1, "Index tensor must be 1D for ROW_MAJOR layout!");
        } else {
            // nothing
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
            "The value of index_dims={} must be consecutive integers.",
            operation_attributes.index_dims);
        i++;
    }
    if (!output_tensor.has_value()) {
        // If the user decided to not use any optional output tensors, then this would be empty or would be a nullptr.
        return;
    }
    TT_ASSERT(output_tensor->buffer() != nullptr, "Must have 1 output tensor.");
    TT_FATAL(dtype == output_tensor.value().dtype(), "Output tensor must have the same dtype as input tensor!");
}
MorehGetItemOperation::program_factory_t MorehGetItemOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto input_layout = input_tensor.layout();
    if (input_layout == Layout::ROW_MAJOR) {
        return MorehGetItemRmFactory();
    }
    return MorehGetItemTilizedFactory();
}

void MorehGetItemOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehGetItemOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehGetItemOperation::spec_return_value_t MorehGetItemOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return {tensor_args.output->tensor_spec()};
    }

    const auto& input_tensor = tensor_args.input;
    const auto index_dims = operation_attributes.index_dims;
    const auto& index_tensors = tensor_args.index_tensors;
    auto input_shape = input_tensor.logical_shape();
    auto output_shape = input_shape;
    auto layout = input_tensor.layout();

    if (layout == Layout::TILE) {
        // compute output shape
        // ex)
        // input: (10, 20, 30, 40)
        // index_tensor: [(100), (100)]
        // index_dims = 1,2
        // output: (10, 1, 100, 40)
        SmallVector<uint32_t> output_size_vec;
        for (unsigned int dim : output_shape) {
            output_size_vec.push_back(dim);
        }

        auto index = index_tensors[0];
        uint32_t index_size = index.logical_shape()[-1];

        for (unsigned int out_put_dim : index_dims) {
            output_size_vec[out_put_dim] = 1;
        }
        output_size_vec[index_dims.back()] = index_size;

        output_shape = ttnn::Shape(std::move(output_size_vec));
    } else {
        // compute output shape
        // ex)
        // input: (10, 20, 30, 40)
        // index_tensor: [(100), (100)]
        // index_dims = 1,2
        // output: (10, 100, 40)
        SmallVector<uint32_t> output_size_vec;

        auto input_shape = input_tensor.logical_shape();
        uint32_t input_rank = input_shape.rank();

        auto index = index_tensors[0];
        uint32_t index_size = index.logical_shape()[0];

        uint32_t start_dim = operation_attributes.index_dims.front();
        uint32_t last_dim = operation_attributes.index_dims.back();
        for (uint32_t input_dim = 0; input_dim < input_rank; input_dim++) {
            if (input_dim < start_dim || last_dim < input_dim) {
                output_size_vec.push_back(input_shape[input_dim]);
            } else if (start_dim == input_dim) {
                output_size_vec.push_back(index_size);
            }
        }

        output_shape = ttnn::Shape(std::move(output_size_vec));
    }
    return TensorSpec(
        output_shape,
        TensorLayout(
            tensor_args.input.dtype(), PageConfig(tensor_args.input.layout()), operation_attributes.memory_config));
}

MorehGetItemOperation::tensor_return_value_t MorehGetItemOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        log_debug(tt::LogOp, "{}:{} use output tensor", __func__, __LINE__);
        return {tensor_args.output.value()};
    }
    log_debug(tt::LogOp, "{}:{} create output tensor", __func__, __LINE__);
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::moreh::moreh_getitem

namespace ttnn::prim {
ttnn::operations::moreh::moreh_getitem::MorehGetItemOperation::tensor_return_value_t moreh_getitem(
    const Tensor& input,
    const std::vector<Tensor>& index_tensors,
    const ttnn::SmallVector<uint32_t>& index_dims,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::moreh::moreh_getitem::MorehGetItemOperation;
    auto operation_attributes =
        OperationType::operation_attributes_t{index_dims, memory_config.value_or(input.memory_config())};
    auto tensor_args = OperationType::tensor_args_t{input, index_tensors, output};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
