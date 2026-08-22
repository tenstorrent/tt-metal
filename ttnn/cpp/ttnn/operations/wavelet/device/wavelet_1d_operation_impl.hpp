// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/wavelet/device/wavelet_device_operation_types.hpp"

namespace ttnn::prim::detail {

tt::tt_metal::WorkloadDescriptor create_lwt_1d_workload(
    const Lwt1DParams& operation_attributes,
    const Lwt1DInputs& tensor_args,
    Lwt1DOutputs& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords);

void validate_lwt_1d(const Lwt1DParams& operation_attributes, const Lwt1DInputs& tensor_args);
Lwt1DOutputSpecs compute_lwt_1d_output_specs(const Lwt1DParams& operation_attributes, const Lwt1DInputs& tensor_args);
Lwt1DOutputs create_lwt_1d_output_tensors(const Lwt1DParams& operation_attributes, const Lwt1DInputs& tensor_args);

tt::tt_metal::WorkloadDescriptor create_ilwt_1d_workload(
    const Ilwt1DParams& operation_attributes,
    const Ilwt1DInputs& tensor_args,
    Tensor& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords);

void validate_ilwt_1d(const Ilwt1DParams& operation_attributes, const Ilwt1DInputs& tensor_args);
tt::tt_metal::TensorSpec compute_ilwt_1d_output_spec(
    const Ilwt1DParams& operation_attributes, const Ilwt1DInputs& tensor_args);
Tensor create_ilwt_1d_output_tensor(const Ilwt1DParams& operation_attributes, const Ilwt1DInputs& tensor_args);

}  // namespace ttnn::prim::detail
