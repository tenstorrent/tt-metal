// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"

namespace ttnn::operations::reduction::generic {

ReduceDeviceOperation::program_factory_t ReduceDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    const auto& input_tensor = tensor_args.input_tensor;
    auto parallelization_strategy = detail::get_parallelization_strategy(input_tensor, operation_attributes.dim);

    switch (parallelization_strategy) {
        case ReduceOpParallelizationStrategy::MULTI_CORE_H: return program::ReduceMultiCoreHProgramFactory{};
        case ReduceOpParallelizationStrategy::MULTI_CORE_W: return program::ReduceMultiCoreWProgramFactory{};
        case ReduceOpParallelizationStrategy::MULTI_CORE_HW:
        case ReduceOpParallelizationStrategy::SINGLE_CORE_HW: return program::ReduceSingleCoreHwProgramFactory{};
        default: TT_THROW("Unsupported parallelization strategy");
    }
}

void ReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Operands to reduce need to be on device! Got storage type: {}",
        input_tensor.storage_type());
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to reduce need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.layout() == Layout::TILE), "Inputs to reduce must be tilized");
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32 ||
            input_tensor.dtype() == DataType::BFLOAT8_B || input_tensor.dtype() == DataType::UINT32,
        "Only FLOAT32, BFLOAT16, BFLOAT8_B, and UINT32 are supported for generic reduction - got {}",
        input_tensor.dtype());
}

void ReduceDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

ReduceDeviceOperation::spec_return_value_t ReduceDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto output_shape = input_tensor.logical_shape();
    switch (operation_attributes.dim) {
        case tt::tt_metal::ReduceOpDim::H: output_shape[2] = 1; break;
        case tt::tt_metal::ReduceOpDim::W: output_shape[3] = 1; break;
        case tt::tt_metal::ReduceOpDim::HW:
            output_shape[2] = 1;
            output_shape[3] = 1;
            break;
    }

    TensorSpec tensor_spec(
        output_shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(Layout::TILE),
            MemoryConfig(operation_attributes.output_mem_config.buffer_type())));

    if (operation_attributes.output_mem_config.nd_shard_spec().has_value()) {
        if (operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            const auto& nd_shard_spec = *operation_attributes.output_mem_config.nd_shard_spec();
            return tensor_spec.width_sharded(nd_shard_spec.grid, nd_shard_spec.orientation);
        }

        auto nd_shard_spec = *operation_attributes.output_mem_config.nd_shard_spec();
        if (operation_attributes.dim == tt::tt_metal::ReduceOpDim::W ||
            operation_attributes.dim == tt::tt_metal::ReduceOpDim::HW) {
            nd_shard_spec.shard_shape[-1] = 1;
        }
        if ((operation_attributes.dim == tt::tt_metal::ReduceOpDim::H ||
             operation_attributes.dim == tt::tt_metal::ReduceOpDim::HW) &&
            nd_shard_spec.shard_shape.rank() > 1) {
            nd_shard_spec.shard_shape[-2] = tt::div_up(nd_shard_spec.shard_shape[-2], input_tensor.logical_shape()[-2]);
        }
        return tensor_spec.sharded(std::move(nd_shard_spec), tt::tt_metal::TensorSpec::ShardShapeAlignment::REQUIRED);
    }

    return tensor_spec;
}

ReduceDeviceOperation::tensor_return_value_t ReduceDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t ReduceDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<ReduceDeviceOperation>(
        operation_attributes.math_op,
        operation_attributes.dim,
        operation_attributes.scaler,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        operation_attributes.compute_kernel_config,
        operation_attributes.sub_core_grids,
        program_factory.index(),
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.padded_shape());
}

}  // namespace ttnn::operations::reduction::generic

namespace ttnn::prim {
ttnn::Tensor reduce(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scaler,
    const MemoryConfig& output_mem_config,
    const std::optional<DataType>& output_dtype,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::reduction::generic::ReduceDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            reduce_math,
            reduce_dim,
            scaler,
            output_mem_config,
            output_dtype.value_or(input_tensor.dtype()),
            compute_kernel_config,
            sub_core_grids},
        OperationType::tensor_args_t{input_tensor});
}
}  // namespace ttnn::prim
