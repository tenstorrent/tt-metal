// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"

namespace ttnn::prim {

ReduceDeviceOperation::program_factory_t ReduceDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    auto parallelization_strategy = get_parallelization_strategy(tensor_args, operation_attributes.dim);

    switch (parallelization_strategy) {
        case ReduceOpParallelizationStrategy::MULTI_CORE_H: return ReduceMultiCoreHProgramFactory{};
        case ReduceOpParallelizationStrategy::MULTI_CORE_W: return ReduceMultiCoreWProgramFactory{};
        case ReduceOpParallelizationStrategy::MULTI_CORE_HW:
        case ReduceOpParallelizationStrategy::SINGLE_CORE_HW: return ReduceSingleCoreHwProgramFactory{};
        default: TT_THROW("Unsupported parallelization strategy");
    }
}

void ReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.storage_type() == StorageType::DEVICE,
        "Operands to reduce need to be on device! Got storage type: {}",
        tensor_args.storage_type());
    TT_FATAL(tensor_args.buffer() != nullptr, "Operands to reduce need to be allocated in buffers on device!");
    TT_FATAL((tensor_args.layout() == Layout::TILE), "Inputs to reduce must be tilized");
    TT_FATAL(
        tensor_args.dtype() == DataType::BFLOAT16 || tensor_args.dtype() == DataType::FLOAT32 ||
            tensor_args.dtype() == DataType::BFLOAT8_B || tensor_args.dtype() == DataType::UINT32,
        "Only FLOAT32, BFLOAT16, BFLOAT8_B, and UINT32 are supported for generic reduction - got {}",
        tensor_args.dtype());
}

ReduceDeviceOperation::spec_return_value_t ReduceDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_shape = tensor_args.logical_shape();
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

    TensorMemoryLayout mem_layout = operation_attributes.output_mem_config.memory_layout();

    if (mem_layout == TensorMemoryLayout::WIDTH_SHARDED || mem_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
        mem_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        // Grid and orientation are identical in both spec formats (nd_shard_spec and shard_spec)
        // when both are populated. Pick whichever is available.
        const auto& nd = operation_attributes.output_mem_config.nd_shard_spec();
        const auto& legacy = operation_attributes.output_mem_config.shard_spec();
        TT_FATAL(
            nd.has_value() || legacy.has_value(),
            "Sharded memory layout {} requires either nd_shard_spec or shard_spec to be set",
            mem_layout);
        const CoreRangeSet& grid = nd ? nd->grid : legacy->grid;
        ShardOrientation orientation = nd ? nd->orientation : legacy->orientation;

        // For width/height/block sharding modes, the output shard shape is fully determined
        // by the output physical shape and the core grid. Just delegate to the
        // appropriate TensorSpec builder.
        if (mem_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            return tensor_spec.width_sharded(grid, orientation);
        }
        if (mem_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            return tensor_spec.height_sharded(grid, orientation);
        }
        TT_FATAL(
            grid.ranges().size() == 1,
            "Block sharding requires a single CoreRange, got {} ranges",
            grid.ranges().size());
        return tensor_spec.block_sharded(grid.bounding_box(), orientation);
    }

    // ND sharding: adjust per-logical-dimension shard shape for reduced dims.
    if (mem_layout == TensorMemoryLayout::ND_SHARDED) {
        const auto& nd_shard_spec = operation_attributes.output_mem_config.nd_shard_spec();
        TT_FATAL(nd_shard_spec.has_value(), "ND_SHARDED memory layout requires nd_shard_spec to be set");
        auto nd_shard_spec_copy = *nd_shard_spec;
        if (operation_attributes.dim == tt::tt_metal::ReduceOpDim::W ||
            operation_attributes.dim == tt::tt_metal::ReduceOpDim::HW) {
            nd_shard_spec_copy.shard_shape[-1] = 1;
        }
        if ((operation_attributes.dim == tt::tt_metal::ReduceOpDim::H ||
             operation_attributes.dim == tt::tt_metal::ReduceOpDim::HW) &&
            nd_shard_spec_copy.shard_shape.rank() > 1) {
            nd_shard_spec_copy.shard_shape[-2] = 1;
        }
        return tensor_spec.sharded(
            std::move(nd_shard_spec_copy), tt::tt_metal::TensorSpec::ShardShapeAlignment::REQUIRED);
    }

    // Guard against unexpected new memory layouts.
    TT_FATAL(mem_layout == TensorMemoryLayout::INTERLEAVED, "Unexpected memory layout: {}", mem_layout);
    // Interleaved tensor: tensor_spec already has everything we need.
    return tensor_spec;
}

ReduceDeviceOperation::tensor_return_value_t ReduceDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.device());
}

ttsl::hash::hash_t ReduceDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<ReduceDeviceOperation>(
        operation_attributes.math_op,
        operation_attributes.dim,
        operation_attributes.scaler,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        operation_attributes.compute_kernel_config,
        operation_attributes.sub_core_grids,
        operation_attributes.negate,
        program_factory.index(),
        tensor_args.dtype(),
        tensor_args.memory_config(),
        tensor_args.padded_shape(),
        tensor_args.tensor_spec().tile());
}

ttnn::Tensor reduce(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scaler,
    const MemoryConfig& output_mem_config,
    const std::optional<DataType>& output_dtype,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    bool negate) {
    return ttnn::device_operation::launch<ReduceDeviceOperation>(
        ReduceParams{
            reduce_math,
            reduce_dim,
            scaler,
            output_mem_config,
            output_dtype.value_or(input_tensor.dtype()),
            compute_kernel_config,
            sub_core_grids,
            negate},
        input_tensor);
}

}  // namespace ttnn::prim
