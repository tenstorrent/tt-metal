// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/device/ilwt_1d_device_operation.hpp"

#include "ttnn/operations/wavelet/device/wavelet_1d_operation_impl.hpp"
#include "ttnn/operations/wavelet/device/wavelet_l1_budget.hpp"

namespace ttnn::prim {

void Ilwt1DDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    detail::validate_ilwt_1d(operation_attributes, tensor_args);
}

Ilwt1DDeviceOperation::spec_return_value_t Ilwt1DDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return detail::compute_ilwt_1d_output_spec(operation_attributes, tensor_args);
}

Ilwt1DDeviceOperation::tensor_return_value_t Ilwt1DDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return detail::create_ilwt_1d_output_tensor(operation_attributes, tensor_args);
}

Tensor ilwt(
    const Tensor& approximation,
    const Tensor& detail,
    const operations::wavelet::SchemeId scheme_id,
    const operations::wavelet::BoundaryMode boundary_mode,
    const uint32_t original_length,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output) {
    return device_operation::launch<Ilwt1DDeviceOperation>(
        Ilwt1DParams{
            .scheme_id = scheme_id,
            .boundary_mode = boundary_mode,
            .original_length = original_length,
            .available_l1_bytes = detail::quantized_available_l1_bytes(approximation.device()),
            .output_memory_config = output_memory_config,
        },
        Ilwt1DInputs{
            .approximation = approximation,
            .detail = detail,
            .preallocated_output = preallocated_output,
        });
}

}  // namespace ttnn::prim
