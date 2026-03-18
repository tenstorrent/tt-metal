// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "welford_reduce_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"

namespace ttnn::prim {

WelfordReduceDeviceOperation::program_factory_t WelfordReduceDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    return WelfordReduceProgramFactory{};
}

void WelfordReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.storage_type() == StorageType::DEVICE,
        "Operands to Std/Var reductions need to be on device! Got storage type: {}",
        tensor_args.storage_type());
    TT_FATAL(tensor_args.buffer() != nullptr, "Operands to Std/Var reductions need to be allocated in buffers on device!");
    TT_FATAL((tensor_args.layout() == Layout::TILE), "Inputs to Std/Var reductions must be tilized");
    TT_FATAL(
        tensor_args.dtype() == DataType::BFLOAT16 || tensor_args.dtype() == DataType::FLOAT32 ||
            tensor_args.dtype() == DataType::BFLOAT8_B,
        "Only FLOAT32, BFLOAT16, and BFLOAT8_B are supported for Std/Var reduction - got {}",
        tensor_args.dtype());
    TT_FATAL(
        tensor_args.logical_shape().rank() >= 2,
        "Welford reduce only supports tensors with at least 2 dimensions, got rank: {}",
        tensor_args.logical_shape().rank());
}

WelfordReduceDeviceOperation::spec_return_value_t WelfordReduceDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_shape = tensor_args.logical_shape();
    // Always reduce the H dimension for Welford reduce.
    output_shape[-2] = 1;

    TensorSpec tensor_spec(
        output_shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(Layout::TILE),
            MemoryConfig(operation_attributes.output_mem_config.buffer_type())));

    // TODO: Add support for sharded outputs.
    // if (operation_attributes.output_mem_config.nd_shard_spec().has_value()) {
    //     if (operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
    //         const auto& nd_shard_spec = *operation_attributes.output_mem_config.nd_shard_spec();
    //         return tensor_spec.width_sharded(nd_shard_spec.grid, nd_shard_spec.orientation);
    //     }

    //     auto nd_shard_spec = *operation_attributes.output_mem_config.nd_shard_spec();
    //     if (operation_attributes.dim == tt::tt_metal::ReduceOpDim::W ||
    //         operation_attributes.dim == tt::tt_metal::ReduceOpDim::HW) {
    //         nd_shard_spec.shard_shape[-1] = 1;
    //     }
    //     if ((operation_attributes.dim == tt::tt_metal::ReduceOpDim::H ||
    //          operation_attributes.dim == tt::tt_metal::ReduceOpDim::HW) &&
    //         nd_shard_spec.shard_shape.rank() > 1) {
    //         nd_shard_spec.shard_shape[-2] = tt::div_up(nd_shard_spec.shard_shape[-2], tensor_args.logical_shape()[-2]);
    //     }
    //     return tensor_spec.sharded(std::move(nd_shard_spec), tt::tt_metal::TensorSpec::ShardShapeAlignment::REQUIRED);
    // }

    return tensor_spec;
}

WelfordReduceDeviceOperation::tensor_return_value_t WelfordReduceDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.device());
}

ttsl::hash::hash_t WelfordReduceDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    //TODO: Add hash for any paremeters added later.

    return tt::tt_metal::operation::hash_operation<WelfordReduceDeviceOperation>(
        operation_attributes.math_op,
        operation_attributes.scaler,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        operation_attributes.compute_kernel_config,
        operation_attributes.sub_core_grids,
        program_factory.index(),
        tensor_args.dtype(),
        tensor_args.memory_config(),
        tensor_args.padded_shape());
}

ttnn::Tensor welford_reduce(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    float scaler,
    const MemoryConfig& output_mem_config,
    const std::optional<DataType>& output_dtype,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    ) {
    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(ttnn::init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        std::nullopt,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true));

    return ttnn::device_operation::launch<WelfordReduceDeviceOperation>(
        WelfordReduceParams{
            reduce_math,
            scaler,
            output_mem_config,
            output_dtype.value_or(input_tensor.dtype()),
            config,
            sub_core_grids},
        input_tensor);
}

}  // namespace ttnn::prim
