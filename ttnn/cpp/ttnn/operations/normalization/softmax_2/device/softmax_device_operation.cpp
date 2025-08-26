// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization::softmax {

SoftmaxDeviceOperation::program_factory_t SoftmaxDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Determine if we should use sharded multi-core program factory
    const auto input_tensor_shape = tensor_args.input_tensor.padded_shape();
    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const uint32_t Wt_input = input_tensor_shape[3] / tile_width;

    // TODO:
}

void SoftmaxDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return validate_on_program_cache_miss(attributes, tensor_args);
}

void SoftmaxDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensors_args) {}

SoftmaxDeviceOperation::spec_return_value_t SoftmaxDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return {};
}

SoftmaxDeviceOperation::tensor_return_value_t SoftmaxDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return {};
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
    return {operation_attributes_t{}, tensor_args_t{}};
}

Tensor softmax(
    const Tensor& input_tensor,
    int8_t dim,
    tt::tt_metal::MemoryConfig output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {}

Tensor scale_mask_softmax(
    const Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    tt::tt_metal::MemoryConfig output_mem_config,
    bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {}

Tensor softmax_in_place(
    Tensor& input_tensor,
    int8_t dim,
    SoftmaxProgramConfig program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // TODO: Inplace currently available only for last dim and rank 4
}

Tensor scale_mask_softmax_in_place(
    Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    SoftmaxProgramConfig program_config,
    bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {}

Tensor scale_causal_mask_hw_dims_softmax_in_place(
    Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    SoftmaxProgramConfig program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {}
}  // namespace ttnn::operations::normalization::softmax
