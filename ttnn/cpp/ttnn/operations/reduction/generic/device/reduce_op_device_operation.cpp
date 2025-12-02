// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"

namespace ttnn::operations::reduction::generic {

std::tuple<ReduceDeviceOperation::operation_attributes_t, ReduceDeviceOperation::tensor_args_t>
ReduceDeviceOperation::invoke(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scaler,
    const MemoryConfig& output_mem_config,
    const std::optional<DataType>& output_dtype,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return {
        operation_attributes_t{
            reduce_math,
            reduce_dim,
            scaler,
            output_mem_config,
            output_dtype.value_or(input_tensor.dtype()),
            compute_kernel_config,
            sub_core_grids},
        tensor_args_t{input_tensor}};
}

ReduceDeviceOperation::program_factory_t ReduceDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    uint32_t num_tiles = input_tensor.physical_volume() / tt::constants::TILE_HW;

    if (operation_attributes.dim == tt::tt_metal::ReduceOpDim::H) {
        return program::ReduceMultiCoreHProgramFactory{};
    } else if (operation_attributes.dim == tt::tt_metal::ReduceOpDim::W) {
        return program::ReduceMultiCoreWProgramFactory{};
    } else if (operation_attributes.dim == tt::tt_metal::ReduceOpDim::HW) {
        if (num_tiles > 1) {
            // MultiCoreHW is mapped to SingleCoreHW in the legacy code switch case
            // case ReduceOpParallelizationStrategy::MULTI_CORE_HW:
            // case ReduceOpParallelizationStrategy::SINGLE_CORE_HW:
            //    return reduce_single_core_hw(...)
            // Wait, get_parallelization_strategy returns MULTI_CORE_HW if num_tiles > 1.
            // But create_program calls reduce_single_core_hw for both.
            // So we should return ReduceSingleCoreHwProgramFactory for both?
            // But the factories are named differently?
            // No, I created ReduceSingleCoreHwProgramFactory.
            // I did NOT create ReduceMultiCoreHwProgramFactory.
            // The legacy code used reduce_single_core_hw for MULTI_CORE_HW as well?
            // Let's check reduce_op.cpp:
            // case ReduceOpParallelizationStrategy::MULTI_CORE_HW:
            // case ReduceOpParallelizationStrategy::SINGLE_CORE_HW:
            //    return reduce_single_core_hw(...)
            // Yes.
            return program::ReduceSingleCoreHwProgramFactory{};
        } else {
            return program::ReduceSingleCoreHwProgramFactory{};
        }
    } else {
        TT_THROW("Unsupported reduce dim");
    }
}

void ReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
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

    if (input_tensor.nd_shard_spec().has_value()) {
        if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            const auto& nd_shard_spec = *input_tensor.nd_shard_spec();
            return tensor_spec.width_sharded(nd_shard_spec.grid, nd_shard_spec.orientation);
        }

        auto nd_shard_spec = *input_tensor.nd_shard_spec();
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

    return tt::tt_metal::operation::hash_operation<ReduceDeviceOperation>(
        operation_attributes.math_op,
        operation_attributes.dim,
        operation_attributes.scaler,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        operation_attributes.compute_kernel_config,
        operation_attributes.sub_core_grids,
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.padded_shape());
}

}  // namespace ttnn::operations::reduction::generic
