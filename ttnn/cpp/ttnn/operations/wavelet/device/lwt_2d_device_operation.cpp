// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/device/lwt_2d_device_operation.hpp"

#include "ttnn/operations/wavelet/device/wavelet_2d_operation_impl.hpp"
#include "ttnn/operations/wavelet/device/wavelet_l1_budget.hpp"

namespace ttnn::prim {

void Lwt2DDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    detail::validate_lwt_2d(operation_attributes, tensor_args);
}

Lwt2DDeviceOperation::spec_return_value_t Lwt2DDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return detail::compute_lwt_2d_output_specs(operation_attributes, tensor_args);
}

Lwt2DDeviceOperation::tensor_return_value_t Lwt2DDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return detail::create_lwt_2d_output_tensors(operation_attributes, tensor_args);
}

Lwt2DOutputs lwt_2d(
    const Tensor& input,
    const operations::wavelet::SchemeId scheme_id,
    const operations::wavelet::BoundaryMode boundary_mode,
    const MemoryConfig& output_memory_config,
    const std::optional<std::array<Tensor, 4>>& preallocated_outputs) {
    return device_operation::launch<Lwt2DDeviceOperation>(
        Lwt2DParams{
            .scheme_id = scheme_id,
            .boundary_mode = boundary_mode,
            .available_l1_bytes = detail::quantized_available_l1_bytes(input.device()),
            .output_memory_config = output_memory_config,
        },
        Lwt2DInputs{
            .input = input,
            .preallocated_outputs = preallocated_outputs,
        });
}

}  // namespace ttnn::prim
