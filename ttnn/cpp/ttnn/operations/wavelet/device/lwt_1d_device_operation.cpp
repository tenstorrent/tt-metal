// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/device/lwt_1d_device_operation.hpp"

#include "ttnn/operations/wavelet/device/wavelet_1d_operation_impl.hpp"
#include "ttnn/operations/wavelet/device/wavelet_l1_budget.hpp"

namespace ttnn::prim {

void Lwt1DDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    detail::validate_lwt_1d(operation_attributes, tensor_args);
}

Lwt1DDeviceOperation::spec_return_value_t Lwt1DDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return detail::compute_lwt_1d_output_specs(operation_attributes, tensor_args);
}

Lwt1DDeviceOperation::tensor_return_value_t Lwt1DDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return detail::create_lwt_1d_output_tensors(operation_attributes, tensor_args);
}

Lwt1DOutputs lwt(
    const Tensor& input,
    const operations::wavelet::SchemeId scheme_id,
    const operations::wavelet::BoundaryMode boundary_mode,
    const MemoryConfig& output_memory_config,
    const std::optional<Lwt1DOutputs>& preallocated_outputs) {
    return device_operation::launch<Lwt1DDeviceOperation>(
        Lwt1DParams{
            .scheme_id = scheme_id,
            .boundary_mode = boundary_mode,
            .available_l1_bytes = detail::quantized_available_l1_bytes(input.device()),
            .output_memory_config = output_memory_config,
        },
        Lwt1DInputs{
            .input = input,
            .preallocated_outputs = preallocated_outputs,
        });
}

}  // namespace ttnn::prim
