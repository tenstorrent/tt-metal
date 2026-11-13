// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/device/ilwt_2d_device_operation.hpp"

#include "ttnn/operations/wavelet/device/wavelet_2d_operation_impl.hpp"
#include "ttnn/operations/wavelet/device/wavelet_l1_budget.hpp"

namespace ttnn::prim {

void Ilwt2DDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    detail::validate_ilwt_2d(operation_attributes, tensor_args);
}

Ilwt2DDeviceOperation::spec_return_value_t Ilwt2DDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return detail::compute_ilwt_2d_output_spec(operation_attributes, tensor_args);
}

Ilwt2DDeviceOperation::tensor_return_value_t Ilwt2DDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return detail::create_ilwt_2d_output_tensor(operation_attributes, tensor_args);
}

Tensor ilwt_2d(
    const Tensor& ll,
    const Tensor& lh,
    const Tensor& hl,
    const Tensor& hh,
    const operations::wavelet::SchemeId scheme_id,
    const operations::wavelet::BoundaryMode boundary_mode,
    const uint32_t output_height,
    const uint32_t output_width,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output) {
    return device_operation::launch<Ilwt2DDeviceOperation>(
        Ilwt2DParams{
            .scheme_id = scheme_id,
            .boundary_mode = boundary_mode,
            .output_height = output_height,
            .output_width = output_width,
            .available_l1_bytes = detail::quantized_available_l1_bytes(ll.device()),
            .output_memory_config = output_memory_config,
        },
        Ilwt2DInputs{
            .ll = ll,
            .lh = lh,
            .hl = hl,
            .hh = hh,
            .preallocated_output = preallocated_output,
        });
}

}  // namespace ttnn::prim
