// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include "softmax/device/softmax_types.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization::softmax {

SoftmaxDeviceOperation::program_factory_t SoftmaxDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Determine if we should use sharded multi-core program factory
    const auto input_tensor_shape = tensor_args.input_tensor.padded_shape();
    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto rank = input_tensor_shape.size();

    if (operation_attributes.softmax_type == SoftmaxOperationType::SoftmaxInPlace ||
        operation_attributes.softmax_type == SoftmaxOperationType::ScaleMaskSoftmaxInPlace ||
        operation_attributes.softmax_type == SoftmaxOperationType::ScaleCausalMaskHWSoftmaxInPlace ||
        operation_attributes.softmax_type == SoftmaxOperationType::ScaleCausalMaskHWSoftmaxInPlace) {
        return program::SoftmaxProgramFactoryAttentionOptimized{};
    } else if (
        operation_attributes.softmax_type == SoftmaxOperationType::Softmax && operation_attributes.dim == rank - 1 &&
        rank == 4) {
        return program::SoftmaxProgramFactoryAttentionOptimized{};
    }
    return program::SoftmaxProgramFactoryGeneral{};
}

void SoftmaxDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return validate_on_program_cache_miss(attributes, tensor_args);
}

void SoftmaxDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensors_args) {
    // TODO:
}

SoftmaxDeviceOperation::spec_return_value_t SoftmaxDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if ((attributes.softmax_type == SoftmaxOperationType::SoftmaxInPlace ||
         attributes.softmax_type == SoftmaxOperationType::ScaleMaskSoftmaxInPlace ||
         attributes.softmax_type == SoftmaxOperationType::ScaleCausalMaskHWSoftmaxInPlace) &&
        attributes.inplace) {
        return tensor_args.input_tensor.tensor_spec();
    }
    return {TensorSpec(
        tensor_args.input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_args.input_tensor.dtype(),
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            attributes.output_mem_config))};
}

SoftmaxDeviceOperation::tensor_return_value_t SoftmaxDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // Inplace config
    if ((attributes.softmax_type == SoftmaxOperationType::SoftmaxInPlace ||
         attributes.softmax_type == SoftmaxOperationType::ScaleMaskSoftmaxInPlace ||
         attributes.softmax_type == SoftmaxOperationType::ScaleCausalMaskHWSoftmaxInPlace) &&
        attributes.inplace) {
        return tensor_args.input_tensor;
    }

    // Standard
    return {create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input_tensor.device())};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<SoftmaxDeviceOperation::tensor_return_value_t>
SoftmaxDeviceOperation::create_op_performance_model(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, const Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    int ideal_dev_clock_cycles = data_movement::common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}

std::tuple<SoftmaxDeviceOperation::operation_attributes_t, SoftmaxDeviceOperation::tensor_args_t>
SoftmaxDeviceOperation::invoke(
    SoftmaxOperationType softmax_type,
    const Tensor& input_tensor,
    int8_t dim,
    const std::optional<const Tensor>& mask,
    std::optional<float> scale,
    bool inplace,
    tt::tt_metal::MemoryConfig output_mem_config,
    SoftmaxProgramConfig program_config,
    bool is_causal_mask,
    DeviceComputeKernelConfig compute_kernel_config,
    bool is_scale_causal_mask_hw_dims_softmax,
    bool numeric_stable) {
    return {
        operation_attributes_t{
            softmax_type,
            dim,
            scale,
            inplace,
            output_mem_config,
            program_config,
            is_causal_mask,
            compute_kernel_config,
            is_scale_causal_mask_hw_dims_softmax,
            numeric_stable},
        tensor_args_t{input_tensor, mask}};
}

Tensor softmax(
    QueueId queue_id,
    const Tensor& input_tensor,
    int8_t dim,
    tt::tt_metal::MemoryConfig output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto is_fp32 = input_tensor.dtype() == DataType::FLOAT32;
    const auto compute_kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32, false);

    // Operation
    return ttnn::prim::softmax(
        queue_id,
        SoftmaxOperationType::Softmax,
        /*input_tensor=*/input_tensor,
        /*dim=*/dim,
        /*mask=*/std::nullopt,
        /*scale=*/std::nullopt,
        /*inplace=*/false,
        /*output_mem_config=*/output_mem_config,
        /*program_config=*/SoftmaxDefaultProgramConfig{},
        /*is_causal_mask=*/false,
        /*compute_kernel_config=*/
        compute_kernel_config_val,
        /*is_scale_causal_mask_hw_dims_softmax=*/false,
        /*numeric_stable=*/numeric_stable);
}

Tensor scale_mask_softmax(
    QueueId queue_id,
    const Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    tt::tt_metal::MemoryConfig output_mem_config,
    bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto is_fp32 = input_tensor.dtype() == DataType::FLOAT32;
    const auto compute_kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32, false);

    // TODO: Transpose if not dim = -1

    // Operation
    return ttnn::prim::softmax(
        queue_id,
        SoftmaxOperationType::ScaleMaskSoftmax,
        /*input_tensor=*/input_tensor,
        /*dim=*/-1,
        /*mask=*/mask,
        /*scale=*/scale,
        /*inplace=*/false,
        /*output_mem_config=*/output_mem_config,
        /*program_config=*/SoftmaxDefaultProgramConfig{},
        /*is_causal_mask=*/is_causal_mask,
        /*compute_kernel_config=*/compute_kernel_config_val,
        /*is_scale_causal_mask_hw_dims_softmax=*/false,
        /*numeric_stable=*/numeric_stable);
}

Tensor softmax_in_place(
    QueueId queue_id,
    Tensor& input_tensor,
    int8_t dim,
    SoftmaxProgramConfig program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // TODO: Inplace currently available only for last dim and rank 4
    // Constants
    const auto is_fp32 = input_tensor.dtype() == DataType::FLOAT32;
    const auto compute_kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32, false);

    // Operation
    return ttnn::prim::softmax(
        queue_id,
        SoftmaxOperationType::SoftmaxInPlace,
        /*input_tensor=*/input_tensor,
        /*dim=*/dim,
        /*mask=*/std::nullopt,
        /*scale=*/std::nullopt,
        /*inplace=*/false,
        /*output_mem_config=*/input_tensor.memory_config(),
        /*program_config=*/program_config,
        /*is_causal_mask=*/false,
        /*compute_kernel_config=*/compute_kernel_config_val,
        /*is_scale_causal_mask_hw_dims_softmax=*/false,
        /*numeric_stable=*/numeric_stable);
}

Tensor scale_mask_softmax_in_place(
    QueueId queue_id,
    Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    SoftmaxProgramConfig program_config,
    bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto is_fp32 = input_tensor.dtype() == DataType::FLOAT32;
    const auto compute_kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32, false);

    // Operation
    return ttnn::prim::softmax(
        queue_id,
        SoftmaxOperationType::ScaleMaskSoftmaxInPlace,
        /*input_tensor=*/input_tensor,
        /*dim=*/-1,
        /*mask=*/mask,
        /*scale=*/scale,
        /*inplace=*/false,
        /*output_mem_config=*/input_tensor.memory_config(),
        /*program_config=*/program_config,
        /*is_causal_mask=*/is_causal_mask,
        /*compute_kernel_config=*/compute_kernel_config_val,
        /*is_scale_causal_mask_hw_dims_softmax=*/false,
        /*numeric_stable=*/numeric_stable);
}

Tensor scale_causal_mask_hw_dims_softmax_in_place(
    QueueId queue_id,
    Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    SoftmaxProgramConfig program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto is_fp32 = input_tensor.dtype() == DataType::FLOAT32;
    const auto compute_kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32, false);

    // Operation
    return ttnn::prim::softmax(
        queue_id,
        SoftmaxOperationType::ScaleCausalMaskHWSoftmaxInPlace,
        /*input_tensor=*/input_tensor,
        /*dim=*/-1,
        /*mask=*/mask,
        /*scale=*/scale,
        /*inplace=*/false,
        /*output_mem_config=*/input_tensor.memory_config(),
        /*program_config=*/program_config,
        /*is_causal_mask=*/false,
        /*compute_kernel_config=*/compute_kernel_config_val,
        /*is_scale_causal_mask_hw_dims_softmax=*/false,
        /*numeric_stable=*/numeric_stable);
}
}  // namespace ttnn::operations::normalization::softmax
