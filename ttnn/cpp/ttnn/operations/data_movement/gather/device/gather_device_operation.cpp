// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::gather {

constexpr uint32_t WT_THRESHOLD = 60;

GatherDeviceOperation::program_factory_t GatherDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Calculate Wt to decide which program factory to use
    const auto input_tensor_shape = tensor_args.input_tensor.padded_shape();
    const auto input_index_tensor_shape = tensor_args.input_index_tensor.padded_shape();
    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const uint32_t Wt_input = input_tensor_shape[3] / tile_width;
    const uint32_t Wt_index = input_index_tensor_shape[3] / tile_width;

    if (Wt_input > WT_THRESHOLD || Wt_index > WT_THRESHOLD) {
        // Use GatherProgramFactorySingleRowMultiCore for larger Wt
        return gather::program::GatherProgramFactorySingleRowMultiCore{};
    }
    return gather::program::GatherProgramFactorySingleRowSingleCore{};
}

void GatherDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

void GatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // Validate shapes of input and output tensors
    const auto input_tensor_shape = tensor_args.input_tensor.logical_shape();
    const auto input_tensor_rank = input_tensor_shape.rank();
    const auto input_index_tensor_shape = tensor_args.input_index_tensor.logical_shape();
    const auto input_index_tensor_rank = input_index_tensor_shape.rank();

    TT_FATAL(
        input_tensor_rank == input_index_tensor_rank,
        "Input and index tensor must have the same number of dimensions. Got input dim: {} and index dim: {}",
        input_tensor_rank,
        input_index_tensor_rank);

    if (tensor_args.output_tensor.has_value()) {
        const auto output_tensor_shape = tensor_args.output_tensor.value().logical_shape();
        TT_FATAL(
            output_tensor_shape == input_index_tensor_shape,
            "Output tensor shape must be the same as index tensor shape. Got output tensor shape: {} and index "
            "tensor shape: {}",
            output_tensor_shape,
            input_index_tensor_shape);
    }
    TT_FATAL(
        tensor_args.input_index_tensor.dtype() == DataType::UINT32 ||
            tensor_args.input_index_tensor.dtype() == DataType::UINT16,
        "Index tensor must be of type UINT32 or UINT16. Got: {}",
        tensor_args.input_index_tensor.dtype());

    for (int i = 0; i < input_tensor_rank - 1; ++i) {
        // Validate all dimensions except the last one, as the tensor has been transposed
        // to move the gather dimension to the last position.
        // Improvement idea: Consider removing transposition and handling arbitrary dimensions directly in the kernel.
        TT_FATAL(
            input_index_tensor_shape[i] <= input_tensor_shape[i],
            "Index tensor shape dimension {} must be less than or equal to input tensor shape dimension {}. Got "
            "index tensor shape: {} and input tensor shape: {}",
            i,
            i,
            input_index_tensor_shape[i],
            input_tensor_shape[i]);
    }
    TT_FATAL(
        attributes.output_mem_config.is_sharded() == false,
        "Sharded implementation not supported yet. Shard status: {}",
        attributes.output_mem_config.is_sharded());

    TT_FATAL(
        tensor_args.input_tensor.layout() == Layout::TILE,
        "The input tensor must be in tiled format. Current layout: {}",
        tensor_args.input_tensor.layout());
    TT_FATAL(
        tensor_args.input_index_tensor.layout() == Layout::TILE,
        "The input index tensor must be in tiled format. Current layout: {}",
        tensor_args.input_index_tensor.layout());

    TT_FATAL(
        (tensor_args.input_tensor.buffer() != nullptr) && (tensor_args.input_index_tensor.buffer() != nullptr),
        "Operands need to be allocated in buffers on the device. Buffer is null.");
    TT_FATAL(
        tensor_args.input_tensor.storage_type() == StorageType::DEVICE,
        "Operation requires input to be on Device. Input storage type: {}",
        tensor_args.input_tensor.storage_type());
    TT_FATAL(
        tensor_args.input_index_tensor.storage_type() == StorageType::DEVICE,
        "Operation requires input to be on Device. Input storage type: {}",
        tensor_args.input_index_tensor.storage_type());

    TT_FATAL(attributes.sparse_grad == false, "Sparse gradient is not supported yet.");
}

GatherDeviceOperation::spec_return_value_t GatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value().tensor_spec();
    }
    // Create output tensor specs
    const auto input_index_tensor_shape = tensor_args.input_index_tensor.logical_shape();
    const auto tensor_specs = TensorSpec(
        input_index_tensor_shape,
        TensorLayout(tensor_args.input_tensor.dtype(), PageConfig(Layout::TILE), attributes.output_mem_config));
    return tensor_specs;
}

GatherDeviceOperation::tensor_return_value_t GatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }
    const auto output_specs = compute_output_specs(attributes, tensor_args);
    return create_device_tensor(output_specs, tensor_args.input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<GatherDeviceOperation::tensor_return_value_t>
GatherDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*op_attr*/, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::operations::data_movement::gather

namespace ttnn::prim {

ttnn::operations::data_movement::gather::tensor_return_value_t gather(
    const Tensor& input_tensor,
    const int8_t dim,
    const Tensor& input_index_tensor,
    const bool sparse_grad,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& output_tensors,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::data_movement::gather::GatherDeviceOperation;

    auto operation_attributes =
        OperationType::operation_attributes_t{dim, sparse_grad, output_memory_config, sub_core_grids};
    auto tensor_args = OperationType::tensor_args_t{input_tensor, input_index_tensor, output_tensors};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
